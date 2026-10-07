"""Application service for the Fylit RAG query pipeline.

This module contains the application-level orchestration required to answer a
question. Keeping this logic outside the FastAPI route separates HTTP concerns
from the RAG workflow and makes the pipeline easier to test independently.
"""

from __future__ import annotations

import logging
import time

from fylit_rag.api.schemas import AskRequest, AskResponse, Diagnostics, Resource
from fylit_rag.config import settings
from fylit_rag.generation.llm import generate_answer
from fylit_rag.generation.memory import ConversationMemory
from fylit_rag.generation.prompts import DISCLAIMER
from fylit_rag.guardrails.input_guards import check_input
from fylit_rag.guardrails.output_guards import check_output
from fylit_rag.indexing.bootstrap import connect
from fylit_rag.indexing.embeddings import embed_texts
from fylit_rag.indexing.search import fetch_embeddings
from fylit_rag.retrieval.hybrid import retrieve
from fylit_rag.retrieval.query_expansion import (
    expand_query,
    has_normalisation,
)
from fylit_rag.retrieval.rerank import rerank
from fylit_rag.retrieval.temporal import prefer_current


log = logging.getLogger(__name__)

# In-process and bounded conversation memory.
memory = ConversationMemory()

# Passages retrieved before reranking, and passages kept as evidence for the
# prompt. From settings so K can be swept without rebuilding - see config.
SHORTLIST = settings.shortlist
EVIDENCE = settings.evidence_count


def _blocked(reason: str, guardrail: str) -> AskResponse:
    """Build a controlled refusal response."""
    return AskResponse(
        answer=reason,
        useful_resources=[],
        disclaimer=DISCLAIMER,
        diagnostics=Diagnostics(
            refused=True,
            guardrail=guardrail,
        ),
    )


def _merge_candidates(primary, secondary):
    """Interleave and deduplicate two retrieval result lists.

    The original FusedResult objects are preserved so their vector and keyword
    provenance remains available to downstream grounding checks.
    """
    merged = []
    seen = set()

    for index in range(max(len(primary), len(secondary))):
        for results in (primary, secondary):
            if index >= len(results):
                continue

            candidate = results[index]
            chunk_id = candidate.result.chunk_id

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            merged.append(candidate)

    return merged


def answer_question(body: AskRequest) -> AskResponse:
    """Run the RAG application pipeline for an API question.

    The service owns application orchestration: input guards, conversation
    context, retrieval, temporal filtering, reranking, generation, output
    guards, diagnostics, and response construction.

    HTTP-specific concerns such as routing and rate limiting remain in
    routes.py.
    """

    # Validate the question before performing expensive retrieval or generation.
    allowed, reason = check_input(body.question)

    if not allowed:
        log.info("input guard refused: %s", reason[:60])
        return _blocked(reason, "input_guard")

    # Add conversation context when a session identifier is available.
    question = memory.contextualise(
        body.session_id or "",
        body.question,
    )

    # Query expansion is used for retrieval and reranking only.
    # Generation still receives the natural contextualised question.
    retrieval_query = expand_query(question)

    if retrieval_query != question:
        log.debug(
            "expanded retrieval query: %r -> %r",
            question,
            retrieval_query,
        )

    # A year-scoped query may need historical/superseded guidance.
    filters = (
        {
            "financial_year": body.financial_year,
            "active": None,
        }
        if body.financial_year
        else None
    )

    # ------------------------------------------------------------------
    # Retrieval and reranking
    # ------------------------------------------------------------------

    started = time.perf_counter()

    try:
        with connect() as conn:
            # Some terminology needs explicit normalisation for retrieval.
            # For example, "Uber Eats" should not retrieve ride-sourcing
            # guidance merely because it contains the word "Uber".
            normalised_retrieval = has_normalisation(question)

            if settings.rerank_strategy == "mmr":
                # MMR requires explicit query and candidate embeddings.
                expanded_vector = embed_texts([retrieval_query])[0]

                if normalised_retrieval:
                    # For a known ambiguous term, retrieve only with the
                    # clarified query.
                    candidates = retrieve(
                        retrieval_query,
                        top_k=SHORTLIST,
                        filters=filters,
                        conn=conn,
                        query_vector=expanded_vector,
                    )

                else:
                    # Preserve both the taxpayer's natural wording and the
                    # expanded ATO-oriented vocabulary.
                    original_vector = embed_texts([question])[0]

                    original_candidates = retrieve(
                        question,
                        top_k=SHORTLIST,
                        filters=filters,
                        conn=conn,
                        query_vector=original_vector,
                    )

                    expanded_candidates = retrieve(
                        retrieval_query,
                        top_k=SHORTLIST,
                        filters=filters,
                        conn=conn,
                        query_vector=expanded_vector,
                    )

                    # Interleave instead of performing another RRF so the
                    # original vector/keyword provenance is preserved.
                    candidates = _merge_candidates(
                        original_candidates,
                        expanded_candidates,
                    )

                # Remove stale year-specific guidance for undated questions.
                # Explicit historical-year questions are preserved.
                candidates = prefer_current(
                    question,
                    candidates,
                )

                candidate_embeddings = fetch_embeddings(
                    conn,
                    [
                        candidate.result.chunk_id
                        for candidate in candidates
                    ],
                )

                evidence = rerank(
                    retrieval_query,
                    candidates,
                    top_n=EVIDENCE,
                    embeddings=candidate_embeddings,
                    query_vector=expanded_vector,
                )

            else:
                if normalised_retrieval:
                    # Use only the clarified query for known ambiguous terms.
                    candidates = retrieve(
                        retrieval_query,
                        top_k=SHORTLIST,
                        filters=filters,
                        conn=conn,
                    )

                else:
                    # Retrieve using both the natural contextualised question
                    # and the expanded ATO-oriented query.
                    original_candidates = retrieve(
                        question,
                        top_k=SHORTLIST,
                        filters=filters,
                        conn=conn,
                    )

                    expanded_candidates = retrieve(
                        retrieval_query,
                        top_k=SHORTLIST,
                        filters=filters,
                        conn=conn,
                    )

                    candidates = _merge_candidates(
                        original_candidates,
                        expanded_candidates,
                    )

                # Remove stale annual guidance for undated questions.
                candidates = prefer_current(
                    question,
                    candidates,
                )

                evidence = rerank(
                    retrieval_query,
                    candidates,
                    top_n=EVIDENCE,
                )

    except Exception:
        # Infrastructure failures should not leak internal exceptions through
        # the public API and should not appear as insufficient ATO evidence.
        log.exception("retrieval failed")

        return _blocked(
            "Something went wrong looking that up. Please try again shortly.",
            "retrieval_failed",
        )

    retrieval_ms = (time.perf_counter() - started) * 1000

    # ------------------------------------------------------------------
    # Generation and output guardrails
    # ------------------------------------------------------------------

    started = time.perf_counter()

    result = check_output(
        generate_answer(
            question,
            evidence,
            max_evidence=EVIDENCE,
        )
    )

    generation_ms = (time.perf_counter() - started) * 1000

    # Store successful conversational turns only.
    if not result["refused"]:
        memory.add_turn(
            body.session_id or "",
            body.question,
            result["answer"],
        )

    # Record retrieved passages containing instruction-like content.
    findings = len(result.get("injection_findings") or [])

    if findings:
        log.warning(
            "neutralised instruction-like text in %d retrieved passage(s)",
            findings,
        )

    # ------------------------------------------------------------------
    # API response
    # ------------------------------------------------------------------

    return AskResponse(
        answer=result["answer"],
        useful_resources=[
            Resource(
                title=source["title"],
                url=source["url"],
            )
            for source in result["sources"]
            if source.get("url")
        ],
        disclaimer=result.get("disclaimer") or DISCLAIMER,
        diagnostics=Diagnostics(
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=round(generation_ms, 1),
            chunks_considered=len(candidates),
            refused=result["refused"],
            guardrail=result.get("guardrail"),
            injection_findings=findings,
        ),
    )