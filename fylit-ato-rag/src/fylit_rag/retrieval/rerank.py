"""Rerank the hybrid shortlist so the strongest evidence sits on top.

Requirement: compare at least two approaches (e.g. cross-encoder vs
LLM-as-reranker vs score fusion only) and record the outcome in an ADR
plus scripts/evaluate.py numbers.

Three strategies sit behind one interface, chosen by `settings.rerank_strategy`:

``fusion``
    Keep the reciprocal-rank-fusion order. The baseline every other strategy has
    to beat - free, deterministic, and already quite good, so it is what ships
    if the others do not earn their cost.

``mmr``
    Maximal marginal relevance (Carbonell & Goldstein, 1998). Trades a little
    relevance for diversity, using the embeddings already in the index. Free and
    local. It exists because this corpus repeats itself: the same guidance is
    restated across myTax 2021/2022/2023 pages, so an unreranked shortlist can
    spend all five slots on near-identical text and starve the answer of the one
    chunk that covers the rest of the question.

``llm``
    gpt-4o-mini scores each candidate against the question. The strongest signal
    and the only one that reads the question, but it costs an API call on the
    query path and adds latency to every request.

A cross-encoder was considered and rejected: it would pull in torch and
transformers (~2 GB) for a service whose dependency list is otherwise eleven
light packages. If reranking quality becomes the bottleneck, that is the first
thing to revisit.
"""

from __future__ import annotations

import json

from fylit_rag.config import settings
from fylit_rag.openai_client import client
from fylit_rag.retrieval.hybrid import FusedResult

STRATEGIES = ("fusion", "mmr", "llm")

# Balance in MMR between relevance and novelty. 0.7 keeps relevance clearly in
# charge; the point is to break up near-duplicates, not to reorder the shortlist.
MMR_LAMBDA = 0.7

# Only this many candidates are shown to the LLM. The fused shortlist is longer,
# but a reranker that reads 60 chunks per query costs more than the answer does.
LLM_CANDIDATES = 20

# Each candidate is truncated to this many characters in the prompt. Enough to
# judge relevance, short enough that 20 of them stay affordable.
LLM_SNIPPET_CHARS = 600


# ---------------------------------------------------------------- strategies


def _fusion(query: str, candidates: list[FusedResult], top_n: int, **_) -> list[FusedResult]:
    """Keep the fused order. The baseline, and deliberately a no-op."""
    return candidates[:top_n]


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _mmr(
    query: str,
    candidates: list[FusedResult],
    top_n: int,
    *,
    embeddings: dict[str, list[float]] | None = None,
    query_vector: list[float] | None = None,
    lambda_: float = MMR_LAMBDA,
    **_,
) -> list[FusedResult]:
    """Greedily pick relevant-but-not-redundant chunks.

    Falls back to fused order when embeddings are unavailable rather than
    failing the query: a degraded shortlist beats no answer.
    """
    if not embeddings or query_vector is None:
        return candidates[:top_n]

    pool = [c for c in candidates if c.chunk_id in embeddings]
    if not pool:
        return candidates[:top_n]

    relevance = {c.chunk_id: _cosine(query_vector, embeddings[c.chunk_id]) for c in pool}
    selected: list[FusedResult] = []

    while pool and len(selected) < top_n:
        best, best_score = None, None
        for candidate in pool:
            redundancy = max(
                (_cosine(embeddings[candidate.chunk_id], embeddings[s.chunk_id])
                 for s in selected),
                default=0.0,
            )
            score = lambda_ * relevance[candidate.chunk_id] - (1 - lambda_) * redundancy
            if best_score is None or score > best_score:
                best, best_score = candidate, score
        pool.remove(best)
        best.rerank_score = best_score
        selected.append(best)

    return selected


def _llm(
    query: str,
    candidates: list[FusedResult],
    top_n: int,
    *,
    score_fn=None,
    **_,
) -> list[FusedResult]:
    """Ask gpt-4o-mini how well each candidate answers the question.

    One call scoring all candidates together, not one call each: the model ranks
    better when it can compare, and 20 calls per query would be unaffordable.

    Any failure falls back to the fused order. A reranker is an improvement on a
    shortlist that is already usable, so it must never be able to fail a query.
    """
    shortlist = candidates[:LLM_CANDIDATES]
    if not shortlist:
        return []

    score_fn = score_fn or _llm_scores
    try:
        scores = score_fn(query, shortlist)
    except Exception:  # noqa: BLE001 - any failure must degrade to the fused order, not fail the query
        return candidates[:top_n]

    for i, candidate in enumerate(shortlist):
        candidate.rerank_score = float(scores.get(i, 0.0))

    ranked = sorted(shortlist, key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
    return ranked[:top_n]


def _llm_scores(query: str, shortlist: list[FusedResult]) -> dict[int, float]:
    """Return {candidate index: relevance 0-10} from one chat completion."""
    passages = "\n\n".join(
        f"[{i}] {c.result.source_title} - {' > '.join(c.result.heading_path[-2:])}\n"
        f"{c.result.text[:LLM_SNIPPET_CHARS]}"
        for i, c in enumerate(shortlist)
    )
    response = client().chat.completions.create(
        model=settings.generation_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You rank passages from Australian Taxation Office guidance by how well "
                    "they answer a question. Judge only what the passage says; do not use "
                    "outside knowledge and do not answer the question. Respond with JSON: "
                    '{"scores": [{"index": <int>, "score": <0-10>}]} covering every passage.'
                ),
            },
            {"role": "user", "content": f"Question: {query}\n\nPassages:\n{passages}"},
        ],
    )
    payload = json.loads(response.choices[0].message.content)
    return {int(s["index"]): float(s["score"]) for s in payload.get("scores", [])}


_IMPLEMENTATIONS = {"fusion": _fusion, "mmr": _mmr, "llm": _llm}


# ---------------------------------------------------------------- interface


def rerank(
    query: str,
    candidates,
    top_n: int = 5,
    *,
    strategy: str | None = None,
    **kwargs,
) -> list[FusedResult]:
    """Reorder a hybrid shortlist and return the best `top_n`.

    `strategy` defaults to `settings.rerank_strategy`, so the choice is
    configuration rather than code. Extra keyword arguments are passed to the
    strategy - `mmr` wants `embeddings` and `query_vector`, `llm` accepts a
    `score_fn` for testing.
    """
    strategy = strategy or settings.rerank_strategy
    if strategy not in _IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown rerank strategy {strategy!r}. Available: {', '.join(STRATEGIES)}"
        )
    candidates = list(candidates)
    if not candidates:
        return []
    return _IMPLEMENTATIONS[strategy](query, candidates, top_n, **kwargs)
