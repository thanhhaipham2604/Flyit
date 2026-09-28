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
from fylit_rag.retrieval.rerank import rerank


log = logging.getLogger(__name__)

# In-process and bounded conversation memory.
memory = ConversationMemory()

# Number of passages retrieved before reranking and number retained as evidence.
SHORTLIST = 30
EVIDENCE = 5


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


def answer_question(body: AskRequest) -> AskResponse:
    """Run the RAG application pipeline for an API question.

    The service owns application orchestration: input guards, conversation
    context, retrieval, reranking, generation, output guards, diagnostics,
    and response construction.

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
            if settings.rerank_strategy == "mmr":
                # MMR requires the query embedding and candidate embeddings.
                query_vector = embed_texts([question])[0]

                candidates = retrieve(
                    question,
                    top_k=SHORTLIST,
                    filters=filters,
                    conn=conn,
                    query_vector=query_vector,
                )

                candidate_embeddings = fetch_embeddings(
                    conn,
                    [candidate.chunk_id for candidate in candidates],
                )

                evidence = rerank(
                    question,
                    candidates,
                    top_n=EVIDENCE,
                    embeddings=candidate_embeddings,
                    query_vector=query_vector,
                )

            else:
                candidates = retrieve(
                    question,
                    top_k=SHORTLIST,
                    filters=filters,
                    conn=conn,
                )

                evidence = rerank(
                    question,
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