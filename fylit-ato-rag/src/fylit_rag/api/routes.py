"""Query endpoints.

POST /ask pipeline: validate -> input guards -> (memory contextualise) ->
hybrid retrieve -> rerank -> grounding confidence check -> generate or refuse
-> output guards -> respond with answer + Useful Resources + diagnostics.

Every stage can end the request, and the ones that end it early are the cheap
ones - an input guard costs a regex, the grounding gate costs nothing beyond the
retrieval already done, and neither calls the model. Only a question that gets
past both is worth paying for.

The endpoint is a plain `def`, not `async def`. Retrieval and generation are
blocking calls into psycopg and the OpenAI SDK; declaring the handler async
would run that blocking work on the event loop and stall every other request.
FastAPI runs a sync handler in a threadpool, which is what this needs.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from fylit_rag.api.rate_limit import ask_limit, limiter
from fylit_rag.api.schemas import AskRequest, AskResponse, Diagnostics, Resource
from fylit_rag.config import settings
from fylit_rag.generation.llm import generate_answer
from fylit_rag.generation.memory import ConversationMemory
from fylit_rag.generation.prompts import DISCLAIMER
from fylit_rag.guardrails.input_guards import check_input
from fylit_rag.guardrails.output_guards import check_output
from fylit_rag.indexing.bootstrap import connect
from fylit_rag.retrieval.hybrid import retrieve
from fylit_rag.retrieval.rerank import rerank

log = logging.getLogger(__name__)
router = APIRouter()

# In-process and bounded, matching what `generation.memory` documents. A scaled
# deployment needs shared storage; a single instance does not.
memory = ConversationMemory()

# How many passages retrieval brings back before reranking trims them.
SHORTLIST = 30
EVIDENCE = 5


def _blocked(reason: str, guardrail: str) -> AskResponse:
    """An input guard ended the request. The reason is the answer - it explains
    what we will not do and what we can do instead."""
    return AskResponse(
        answer=reason,
        useful_resources=[],
        disclaimer=DISCLAIMER,
        diagnostics=Diagnostics(refused=True, guardrail=guardrail),
    )


@router.post("/ask", response_model=AskResponse)
@limiter.limit(ask_limit)
def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer a general Australian tax question from official ATO content.

    Returns a grounded answer with its sources, or a controlled refusal. A
    refusal is a correct outcome, not an error, so the status code stays 200 and
    `diagnostics.refused` carries the fact.
    """
    allowed, reason = check_input(body.question)
    if not allowed:
        # Logged because a rise in these is the signal that someone is probing.
        log.info("input guard refused: %s", reason[:60])
        return _blocked(reason, "input_guard")

    question = memory.contextualise(body.session_id or "", body.question)

    filters = {"financial_year": body.financial_year} if body.financial_year else None

    started = time.perf_counter()
    try:
        with connect() as conn:
            candidates = retrieve(question, top_k=SHORTLIST, filters=filters, conn=conn)
            evidence = rerank(question, candidates, top_n=EVIDENCE)
    except Exception:
        # A database or embedding failure must not leak a stack trace to a
        # public endpoint, and must not look like "no ATO content covers this".
        log.exception("retrieval failed")
        return _blocked(
            "Something went wrong looking that up. Please try again shortly.",
            "retrieval_failed",
        )
    retrieval_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    result = check_output(generate_answer(question, evidence))
    generation_ms = (time.perf_counter() - started) * 1000

    if not result["refused"]:
        memory.add_turn(body.session_id or "", body.question, result["answer"])

    findings = len(result.get("injection_findings") or [])
    if findings:
        log.warning("neutralised instruction-like text in %d retrieved passage(s)", findings)

    return AskResponse(
        answer=result["answer"],
        useful_resources=[
            Resource(title=s["title"], url=s["url"])
            for s in result["sources"]
            if s.get("url")
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


@router.get("/config")
def config() -> dict:
    """What this instance is running. Useful in a demo, and safe to expose:
    model names and limits, never the API key or the database URL."""
    return {
        "embedding_model": settings.embedding_model,
        "generation_model": settings.generation_model,
        "rerank_strategy": settings.rerank_strategy,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
    }
