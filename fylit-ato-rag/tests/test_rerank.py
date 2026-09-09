"""Reranking tests: the interface contract and each strategy's behaviour.

No database and no API - candidates are built by hand and the LLM scorer is
injected, so these pin ranking logic rather than model output.
"""

import pytest

from fylit_rag.config import settings
from fylit_rag.indexing.search import SearchResult
from fylit_rag.retrieval.hybrid import FusedResult
from fylit_rag.retrieval.rerank import MMR_LAMBDA, STRATEGIES, rerank


def candidate(chunk_id, score=0.5, doc_id=None, text="Some guidance text."):
    return FusedResult(
        result=SearchResult(
            chunk_id=chunk_id,
            doc_id=doc_id or chunk_id.split("#")[0],
            text=text,
            score=score,
            source_title="A page",
            source_url="https://ato.gov.au/x",
        ),
        score=score,
    )


def shortlist(n=5):
    return [candidate(f"d{i}#0", score=1.0 - i / 10) for i in range(n)]


# ---------------------------------------------------------------- interface


def test_unknown_strategy_is_rejected():
    """A typo must not silently fall back to no reranking."""
    with pytest.raises(ValueError, match="Unknown rerank strategy"):
        rerank("q", shortlist(), strategy="crossencoder")


def test_strategy_defaults_to_configuration(monkeypatch):
    """'Make the choice configurable' - so the default comes from settings."""
    monkeypatch.setattr(settings, "rerank_strategy", "nonsense")
    with pytest.raises(ValueError, match="nonsense"):
        rerank("q", shortlist())


def test_every_advertised_strategy_runs():
    for strategy in STRATEGIES:
        out = rerank("q", shortlist(), top_n=2, strategy=strategy, score_fn=lambda *_: {})
        assert len(out) <= 2


def test_empty_candidates_return_empty():
    assert rerank("q", [], strategy="fusion") == []


def test_top_n_is_respected():
    assert len(rerank("q", shortlist(10), top_n=3, strategy="fusion")) == 3


def test_citation_fields_survive_reranking():
    out = rerank("q", shortlist(), top_n=1, strategy="fusion")
    assert out[0].citation()["url"] == "https://ato.gov.au/x"


# ---------------------------------------------------------------- fusion


def test_fusion_preserves_order():
    """The baseline is deliberately a no-op - it is what the others must beat."""
    cands = shortlist(4)
    out = rerank("q", cands, top_n=4, strategy="fusion")
    assert [c.chunk_id for c in out] == [c.chunk_id for c in cands]


# ---------------------------------------------------------------- mmr


def test_mmr_falls_back_when_embeddings_are_missing():
    """A degraded shortlist beats a failed query."""
    cands = shortlist(3)
    out = rerank("q", cands, top_n=3, strategy="mmr")
    assert [c.chunk_id for c in out] == [c.chunk_id for c in cands]


def test_mmr_prefers_a_novel_chunk_over_a_near_duplicate():
    """The reason mmr exists: this corpus restates the same guidance across
    myTax 2021/2022/2023 pages.

    The geometry has to be chosen with care. If the top-ranked chunk sits on top
    of the query, then every candidate's redundancy equals its relevance and the
    score collapses to (2*lambda - 1) * relevance - monotonic in relevance, so
    diversity can never change the order. Here `a` is offset from the query, so
    `c` can be slightly less relevant than `b` while being far less redundant.
    """
    query_vector = [1.0, 0.0]
    embeddings = {
        "a#0": [0.9, 0.4359],      # relevance 0.90 - the top pick
        "b#0": [0.9, 0.4359],      # identical to a: relevance 0.90, redundancy 1.00
        "c#0": [0.85, -0.5268],    # relevance 0.85, redundancy 0.54
    }
    cands = [candidate("a#0"), candidate("b#0"), candidate("c#0")]

    out = rerank("q", cands, top_n=2, strategy="mmr",
                 embeddings=embeddings, query_vector=query_vector)

    # b: 0.7*0.90 - 0.3*1.00 = 0.33      c: 0.7*0.85 - 0.3*0.54 = 0.43
    assert out[0].chunk_id == "a#0", "the most relevant chunk still goes first"
    assert out[1].chunk_id == "c#0", "the near-duplicate must lose to the novel chunk"


def test_mmr_with_lambda_one_ignores_redundancy():
    """At lambda=1 mmr degenerates to pure relevance - a useful sanity check
    that the diversity term is what changes the order."""
    query_vector = [1.0, 0.0]
    embeddings = {"a#0": [1.0, 0.0], "b#0": [0.99, 0.01], "c#0": [0.6, 0.8]}
    cands = [candidate("a#0"), candidate("b#0"), candidate("c#0")]

    out = rerank("q", cands, top_n=2, strategy="mmr", lambda_=1.0,
                 embeddings=embeddings, query_vector=query_vector)

    assert [c.chunk_id for c in out] == ["a#0", "b#0"]
    assert MMR_LAMBDA < 1.0, "the shipped default must actually apply the penalty"


# ---------------------------------------------------------------- llm


def test_llm_orders_by_returned_score():
    cands = [candidate("a#0"), candidate("b#0"), candidate("c#0")]

    out = rerank("q", cands, top_n=3, strategy="llm",
                 score_fn=lambda *_: {0: 1.0, 1: 9.0, 2: 5.0})

    assert [c.chunk_id for c in out] == ["b#0", "c#0", "a#0"]
    assert out[0].rerank_score == 9.0


def test_llm_failure_falls_back_to_the_fused_order():
    """A reranker improves an already-usable shortlist, so it must never be
    able to fail a query."""
    def boom(*_):
        raise RuntimeError("rate limited")

    cands = shortlist(3)
    out = rerank("q", cands, top_n=3, strategy="llm", score_fn=boom)

    assert [c.chunk_id for c in out] == [c.chunk_id for c in cands]


def test_llm_scores_missing_from_the_response_default_to_zero():
    """A model that skips a passage must not crash the query or promote it."""
    cands = [candidate("a#0"), candidate("b#0")]

    out = rerank("q", cands, top_n=2, strategy="llm", score_fn=lambda *_: {1: 7.0})

    assert out[0].chunk_id == "b#0"
    assert out[1].rerank_score == 0.0


def test_rerank_score_is_kept_separate_from_the_fused_score():
    """The fused score that produced the shortlist stays visible afterwards."""
    cands = [candidate("a#0", score=0.031)]

    out = rerank("q", cands, top_n=1, strategy="llm", score_fn=lambda *_: {0: 8.0})

    assert out[0].score == pytest.approx(0.031)
    assert out[0].rerank_score == 8.0
