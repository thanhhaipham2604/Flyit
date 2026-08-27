"""Schema tests: the invariants retrieval depends on.

Stdlib-only, no database - this is the part of Step 0 that stays verifiable
whether or not a container is running.
"""

import re
from datetime import date

import pytest

from fylit_rag.indexing.schema import (
    FILTERABLE_FIELDS,
    GENERATED_COLUMNS,
    MAX_INDEXABLE_DIMENSIONS,
    YEAR_FILTER_SQL,
    ChunkRecord,
    Status,
    ddl_statements,
    dimensions_for,
    parse_last_updated,
)


def make_chunk(**overrides) -> ChunkRecord:
    base = {
        "chunk_id": "gst-basics#3",
        "doc_id": "gst-basics",
        "chunk_ordinal": 3,
        "content_hash": "a" * 64,
        "text": "You must register for GST if your turnover is $75,000 or more.",
    }
    return ChunkRecord(**(base | overrides))


def ddl() -> str:
    return "\n".join(ddl_statements("chunks", 1536))


@pytest.mark.parametrize(
    ("status", "expected"),
    [(Status.ACTIVE, True), (Status.SUPERSEDED, False), (Status.DELETED, False)],
)
def test_active_is_derived_from_status(status, expected):
    """Only superseded/deleted content drops out of the active filter."""
    assert make_chunk(status=status).active is expected


def test_record_carries_every_filterable_field():
    """A field retrieval filters on but the record omits = a filter that
    silently matches nothing."""
    assert set(FILTERABLE_FIELDS) <= set(make_chunk().as_record())


def test_status_serialises_as_a_plain_string():
    """Stored in a text column with a CHECK constraint; an enum repr would
    fail the constraint rather than round-trip."""
    assert make_chunk(status=Status.SUPERSEDED).as_record()["status"] == "superseded"


def test_row_omits_generated_columns():
    """Postgres computes these. Including them in an INSERT is an error."""
    assert set(make_chunk().as_row()).isdisjoint(GENERATED_COLUMNS)


def test_row_is_otherwise_the_whole_record():
    """as_row() should drop generated columns and nothing else."""
    record = make_chunk().as_record()
    assert set(make_chunk().as_row()) == set(record) - set(GENERATED_COLUMNS)


def test_every_ddl_statement_is_idempotent():
    """bootstrap re-runs on every `docker compose up`; a statement without
    IF NOT EXISTS would turn the second run into a crash."""
    assert all("IF NOT EXISTS" in stmt for stmt in ddl_statements("chunks", 1536))


def test_ddl_creates_both_retrieval_halves():
    """One table, two indexes - lose either and hybrid retrieval is no longer
    hybrid, it just gets quietly worse."""
    sql = ddl()
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    assert "USING gin (search_vector)" in sql


def test_ddl_indexes_the_filter_columns():
    """Unindexed filters still return correct rows - by scanning the table.

    financial_year is an array so it needs GIN; version/active are scalars and
    stay on btree."""
    sql = ddl()
    assert "USING gin (financial_year)" in sql
    assert "(version, active)" in sql


def test_financial_year_is_a_list_not_a_scalar():
    """74% of the corpus names no year, and rate tables name several. A scalar
    column cannot express either case."""
    assert make_chunk().as_record()["financial_year"] == []
    assert make_chunk(financial_year=["2023-24", "2024-25"]).as_record()[
        "financial_year"
    ] == ["2023-24", "2024-25"]


def test_year_filter_keeps_evergreen_content_visible():
    """The whole point: filtering to a year must not discard the 74% of
    chunks that carry no year at all."""
    assert "financial_year = ARRAY[]::text[]" in YEAR_FILTER_SQL  # evergreen
    assert "@>" in YEAR_FILTER_SQL  # or tagged with the requested year
    assert " OR " in YEAR_FILTER_SQL


def test_ddl_bakes_in_the_vector_width():
    assert "vector(1536)" in ddl()


@pytest.mark.parametrize("column", GENERATED_COLUMNS)
def test_each_generated_column_is_generated_in_the_ddl(column):
    """The derivations must be enforced by the database, not by convention -
    and by *each* column's own definition, not just somewhere in the file."""
    assert re.search(rf"\b{column}\s+\w+\s+GENERATED ALWAYS AS", ddl())


def test_new_taxonomy_and_recency_columns_exist():
    """category/topic scope a query; last_updated is the recency signal that
    financial_year cannot be, since 74% of pages carry no year."""
    sql = ddl()
    for column in ("category", "topic", "last_updated"):
        assert column in make_chunk().as_record()
        assert column in sql
    assert "(category, topic)" in sql  # indexed as a pair


@pytest.mark.parametrize(
    ("display", "expected"),
    [
        ("18 February 2026", date(2026, 2, 18)),
        ("  4 August 2021  ", date(2021, 8, 4)),
        ("not a date", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_last_updated(display, expected):
    """An unreadable date must leave the column empty, never guess."""
    assert parse_last_updated(display) == expected


def test_dimensions_for_known_model():
    assert dimensions_for("text-embedding-3-small") == 1536


def test_dimensions_for_unknown_model_names_the_fix():
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        dimensions_for("some-future-model")


def test_ddl_rejects_vectors_pgvector_cannot_index():
    """text-embedding-3-large is 3072-d; HNSW stops at 2000. Silently leaving
    the column unindexed would look fine until retrieval got slow."""
    too_big = MAX_INDEXABLE_DIMENSIONS + 1
    with pytest.raises(ValueError, match="HNSW"):
        ddl_statements("chunks", too_big)


def test_the_large_embedding_model_is_the_concrete_case():
    with pytest.raises(ValueError, match="HNSW"):
        ddl_statements("chunks", dimensions_for("text-embedding-3-large"))
