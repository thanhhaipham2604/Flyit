"""LLM client wrapper (OpenAI chat completions).

Produces either a grounded answer or a controlled refusal - never
free-form knowledge. Timeouts and retries live here.

The order of operations is the safety design, and it is deliberate:

1. **Refuse before generating** when the evidence is thin (`grounding`). Free,
   deterministic, and it removes the temptation rather than detecting it later.
2. **Sanitise the passages** (`guardrails.injection`) so retrieved web content is
   delimited data, never instruction.
3. **Generate** with a system prompt that states the rules.
4. **Check the answer** (`guardrails.output_guards`) and fail closed.

Steps 1 and 4 are not decoration. Step 3's rules are advisory - a model can be
argued out of a prompt - so every rule that matters is enforced by code either
side of the call.

Sources come from the retrieved chunks, not from the model. Asking a model to
report its own citations invites it to invent one that looks right; the chunks we
actually put in the prompt are already the answer to "what did this come from".
"""

from __future__ import annotations

import time

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from fylit_rag.config import settings
from fylit_rag.generation import prompts
from fylit_rag.generation.grounding import evidence_strength, has_sufficient_evidence
from fylit_rag.guardrails.injection import sanitise_evidence
from fylit_rag.openai_client import client

# The answer path is user-facing, so it waits far less than the indexing path.
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 2
TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

# Enough passages to answer from, few enough that the model reads all of them.
MAX_EVIDENCE = 5


def _sources(evidence) -> list[dict]:
    """Citations, deduplicated by URL, in the order the evidence was ranked."""
    seen: set[str] = set()
    sources = []
    for candidate in evidence:
        result = getattr(candidate, "result", candidate)
        url = result.source_url or ""
        if url in seen:
            continue
        seen.add(url)
        sources.append(
            {
                "title": result.source_title,
                "url": url,
                "version": result.version,
                "last_updated": (
                    result.last_updated.isoformat() if result.last_updated else None
                ),
            }
        )
    return sources


def _refusal(reason: str) -> dict:
    return {
        "answer": prompts.REFUSAL_MESSAGE,
        "refused": True,
        "sources": [],
        "disclaimer": prompts.DISCLAIMER,
        "guardrail": reason,
    }


def _complete(messages: list[dict]) -> str:
    """One chat completion, retrying only what is worth retrying."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client().chat.completions.create(
                model=settings.generation_model,
                temperature=0,
                timeout=REQUEST_TIMEOUT,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except TRANSIENT_ERRORS:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable: the loop either returns or raises")


def generate_answer(question: str, evidence, history=None, *, complete=None) -> dict:
    """Answer from the supplied passages, or refuse.

    Returns ``{answer, refused, sources, disclaimer, guardrail, evidence}``.
    The caller is still expected to run `output_guards.check_output` on this -
    the two layers are independent on purpose.

    `complete` is injectable so tests exercise the assembly and the failure
    paths without an API.
    """
    evidence = list(evidence or [])[:MAX_EVIDENCE]

    if not has_sufficient_evidence(question, evidence):
        strength = evidence_strength(evidence)
        return _refusal(
            f"insufficient_evidence(top={strength['top']}, supporting={strength['supporting']})"
        )

    sanitised = sanitise_evidence(evidence)
    history_text = ""
    if history:
        history_text = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in history)

    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompts.build_user_message(question, sanitised.block, history_text),
        },
    ]

    try:
        answer = (complete or _complete)(messages).strip()
    except Exception:  # noqa: BLE001 - an upstream failure must not leak a stack trace
        return _refusal("generation_failed")

    if not answer:
        return _refusal("empty_generation")

    # The model was told to reply with exactly this when the passages fall short.
    if answer.startswith(prompts.REFUSAL_MESSAGE[:40]):
        return _refusal("model_refused")

    return {
        "answer": answer,
        "refused": False,
        "sources": _sources(evidence),
        "disclaimer": prompts.DISCLAIMER,
        "guardrail": None,
        "injection_findings": sanitised.findings,
    }
