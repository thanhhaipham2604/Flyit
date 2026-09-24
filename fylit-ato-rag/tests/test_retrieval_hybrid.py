"""Retrieval tests: the filter contract and the fusion maths.

The fusion tests are pure - no database, no embedding client - because ranking
is the part most likely to be quietly wrong. The search tests need a populated
index and skip without one.
"""

import pytest

from fylit_rag.indexing.search import SearchResult, build_where
from fylit_rag.retrieval.hybrid import RRF_K, reciprocal_rank_fusion, retrieve


def result(chunk_id, score=1.0, title="A page"):
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        text=f"text of {chunk_id}",
        score=score,
        source_title=title,
        source_url="https://ato.gov.au/x",
    )


# ---------------------------------------------------------------- filters


def test_active_only_by_default():
    """Retrieval answers with current guidance unless asked otherwise."""
    sql, params = build_where(None)
    assert "active = %s" in sql
    assert params == [True]


def test_active_none_searches_superseded_content_too():
    """A question scoped to a past year needs the content that is no longer current."""
    sql, params = build_where({"active": None})
    assert sql == "status <> %s"
    assert params == ["deleted"]


def test_deleted_content_is_excluded_by_default():
    """Deleted corpus files must not remain answerable on the normal path."""
    sql, params = build_where({})
    assert "active = %s" in sql
    assert params == [True]


def test_year_filter_keeps_evergreen_content():
    """74% of the corpus names no year; an equality filter would discard it."""
    sql, params = build_where({"financial_year": "2023-24"})
    assert "financial_year = ARRAY[]::text[]" in sql, "evergreen content must survive"
    assert "@>" in sql
    assert params == [True, "2023-24"]


def test_unknown_filter_is_rejected():
    """A filter that silently does nothing is how a query returns content it
    should never have seen."""
    with pytest.raises(ValueError, match="Unknown retrieval filter"):
        build_where({"catgory": "business"})


def test_filters_combine():
    sql, params = build_where({"financial_year": "2023-24", "version": 2, "doc_id": "d1"})
    assert sql.count("AND") == 3
    assert params == [True, "2023-24", 2, "d1"]


# ---------------------------------------------------------------- fusion


def test_agreement_between_halves_outranks_a_single_confident_half():
    """The whole point of fusing ranks: two halves agreeing beats one shouting."""
    fused = reciprocal_rank_fusion(
        {
            "vector": [result("a#1"), result("b#1")],
            "keyword": [result("b#1"), result("c#1")],
        }
    )
    assert fused[0].chunk_id == "b#1", "b appears in both lists and must win"
    assert set(fused[0].sources) == {"vector", "keyword"}


def test_scores_follow_the_rrf_formula():
    fused = reciprocal_rank_fusion({"vector": [result("a#1"), result("b#1")]})
    assert fused[0].score == pytest.approx(1 / (RRF_K + 1))
    assert fused[1].score == pytest.approx(1 / (RRF_K + 2))


def test_chunks_are_deduped_across_halves():
    fused = reciprocal_rank_fusion(
        {"vector": [result("a#1")], "keyword": [result("a#1")]}
    )
    assert len(fused) == 1
    assert fused[0].sources == {"vector": 1, "keyword": 1}


def test_raw_scores_do_not_affect_the_blend():
    """Cosine similarity and ts_rank_cd are not on the same scale - only rank
    should count, or the half with bigger numbers would always win."""
    modest = reciprocal_rank_fusion({"vector": [result("a#1", score=0.01)]})
    huge = reciprocal_rank_fusion({"vector": [result("a#1", score=999.0)]})
    assert modest[0].score == huge[0].score


def test_order_is_stable_for_equal_scores():
    """An unstable shortlist makes evaluation numbers impossible to compare."""
    lists = {"vector": [result("b#1"), result("a#1")], "keyword": [result("a#1"), result("b#1")]}
    first = [f.chunk_id for f in reciprocal_rank_fusion(lists)]
    second = [f.chunk_id for f in reciprocal_rank_fusion(lists)]
    assert first == second == sorted(first)


def test_empty_lists_fuse_to_nothing():
    assert reciprocal_rank_fusion({"vector": [], "keyword": []}) == []


def test_a_blank_query_never_reaches_the_database():
    """Guards the API's most likely bad input without paying for an embedding."""
    def explode(_):
        raise AssertionError("should not embed a blank query")

    assert retrieve("   ", embed=explode) == []


def test_citation_fields_survive_fusion():
    """Every result keeps title, URL and version - citations depend on it."""
    fused = reciprocal_rank_fusion({"vector": [result("a#1", title="Rental income")]})
    citation = fused[0].citation()
    assert citation["title"] == "Rental income"
    assert citation["url"] == "https://ato.gov.au/x"
    assert citation["chunk_id"] == "a#1"
    assert "version" in citation


# ---------------------------------------------------------------- against a real index


@pytest.fixture
def conn():
    psycopg = pytest.importorskip("psycopg")
    from fylit_rag.indexing.bootstrap import connect

    try:
        connection = connect()
    except psycopg.OperationalError as exc:
        pytest.skip(f"no database reachable: {exc}")
    with connection as c:
        if c.execute("SELECT to_regclass('chunks')").fetchone()[0] is None:
            pytest.skip("chunks table missing - run scripts/bootstrap.py")
        if c.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0] == 0:
            pytest.skip("index is empty - run scripts/ingest.py")
        yield c


def test_keyword_search_returns_ranked_results(conn):
    from fylit_rag.indexing.keyword_index import KeywordIndex

    hits = KeywordIndex(conn).search("capital gains tax", top_k=5)

    assert hits, "the corpus certainly mentions capital gains tax"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.retriever == "keyword" for h in hits)
    assert all(h.score >= 0 for h in hits)


def test_vector_search_returns_similarity_not_distance(conn):
    """Higher must mean better in both halves, or whoever reads the two lists
    next will compare them wrongly."""
    from fylit_rag.indexing.embeddings import embed_texts
    from fylit_rag.indexing.vector_index import VectorIndex

    vec = embed_texts(["capital gains tax on a rental property"])[0]
    hits = VectorIndex(conn).search(vec, top_k=5)

    assert hits
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score), "results must be best-first"


def test_the_year_filter_never_returns_another_year(conn):
    from fylit_rag.indexing.keyword_index import KeywordIndex

    hits = KeywordIndex(conn).search(
        "tax free threshold", top_k=100, filters={"financial_year": "2023-24"}
    )

    for h in hits:
        assert not h.financial_year or "2023-24" in h.financial_year
