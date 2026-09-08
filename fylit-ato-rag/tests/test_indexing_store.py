"""Store tests: the incremental-indexing contract, against a real Postgres.

Skipped when no database is reachable, so the suite still runs before anyone has
a container up - the same constraint the schema tests are written to.

These exist because the interesting behaviour is in SQL, not in Python: whether
an embedding survives an update is decided by an ON CONFLICT clause, and a mock
connection would only prove that we wrote the string we meant to write.
"""

from dataclasses import replace

import pytest

from fylit_rag.indexing import store
from fylit_rag.indexing.keyword_index import KeywordIndex
from fylit_rag.indexing.schema import GENERATED_COLUMNS, ChunkRecord, Status
from fylit_rag.indexing.vector_index import VectorIndex

DOC = "_test_store_doc"
VECTOR = str([0.1] * 1536)


@pytest.fixture
def conn():
    """An open connection, or skip. Rows for DOC are removed either side."""
    psycopg = pytest.importorskip("psycopg")
    from fylit_rag.indexing.bootstrap import connect

    try:
        connection = connect()
    except psycopg.OperationalError as exc:
        pytest.skip(f"no database reachable: {exc}")

    with connection as c:
        if c.execute("SELECT to_regclass('chunks')").fetchone()[0] is None:
            pytest.skip("chunks table missing - run scripts/bootstrap.py")
        c.execute("DELETE FROM chunks WHERE doc_id = %s", (DOC,))
        c.commit()
        yield c
        c.execute("DELETE FROM chunks WHERE doc_id = %s", (DOC,))
        c.commit()


def make_chunks(n=3, suffix=""):
    return [
        ChunkRecord(
            chunk_id=f"{DOC}#{i}",
            doc_id=DOC,
            chunk_ordinal=i,
            content_hash=f"hash-{i}{suffix}",
            text=f"Chunk {i} about withholding tax obligations.{suffix}",
            heading_path=["Doc", f"Section {i}"],
            source_title="A document",
            source_url="https://ato.gov.au/x",
            financial_year=["2023-24"],
        )
        for i in range(n)
    ]


def count(conn, where="", params=()):
    sql = f"SELECT count(*) FROM chunks WHERE doc_id = %s {where}"
    return conn.execute(sql, (DOC, *params)).fetchone()[0]


def fill_embeddings(conn):
    conn.execute("UPDATE chunks SET embedding = %s WHERE doc_id = %s", (VECTOR, DOC))


def test_upsert_inserts_rows(conn):
    report = store.upsert_chunks(conn, make_chunks())
    assert report.written == 3
    assert count(conn) == 3


def test_new_rows_have_no_embedding_yet(conn):
    """run_embeddings finds work by looking for NULL, so a fresh row must be NULL."""
    report = store.upsert_chunks(conn, make_chunks())
    assert report.embeddings_cleared == 3
    assert count(conn, "AND embedding IS NULL") == 3


def test_reindexing_unchanged_content_keeps_every_vector(conn):
    """The point of incremental indexing: re-running costs nothing at the API."""
    store.upsert_chunks(conn, make_chunks())
    fill_embeddings(conn)

    report = store.upsert_chunks(conn, make_chunks())

    assert report.written == 3
    assert report.embeddings_cleared == 0
    assert count(conn, "AND embedding IS NOT NULL") == 3


def test_editing_one_chunk_invalidates_only_that_vector(conn):
    """A stale embedding is silently wrong retrieval, so a changed content_hash
    must clear the vector - and an unchanged one must not."""
    chunks = make_chunks()
    store.upsert_chunks(conn, chunks)
    fill_embeddings(conn)

    edited = list(chunks)
    edited[1] = replace(edited[1], text="Rewritten body.", content_hash="hash-1-new")
    report = store.upsert_chunks(conn, edited)

    assert report.embeddings_cleared == 1
    assert count(conn, "AND embedding IS NULL") == 1
    null_id = conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id = %s AND embedding IS NULL", (DOC,)
    ).fetchone()[0]
    assert null_id == f"{DOC}#1"


def test_reindexing_updates_the_text_it_kept_the_vector_for(conn):
    """Metadata changes must still land even when the embedding is preserved."""
    store.upsert_chunks(conn, make_chunks())
    retitled = [replace(c, source_title="Renamed") for c in make_chunks()]
    store.upsert_chunks(conn, retitled)

    titles = {r[0] for r in conn.execute(
        "SELECT source_title FROM chunks WHERE doc_id = %s", (DOC,)
    ).fetchall()}
    assert titles == {"Renamed"}


def test_generated_columns_are_maintained_by_postgres(conn):
    """`active` and `search_vector` are never written by us - the database
    derives them, which is what stops them drifting from their source."""
    store.upsert_chunks(conn, make_chunks())

    active, has_tsv = conn.execute(
        "SELECT active, search_vector IS NOT NULL FROM chunks WHERE chunk_id = %s",
        (f"{DOC}#0",),
    ).fetchone()
    assert active is True
    assert has_tsv is True
    assert not set(GENERATED_COLUMNS) & set(store._columns())


def test_the_keyword_half_needs_no_write_of_its_own(conn):
    """One row serves both halves: writing text makes it findable by keyword."""
    KeywordIndex(conn).upsert(make_chunks())

    hits = count(
        conn,
        "AND search_vector @@ plainto_tsquery('english', %s)",
        ("withholding tax",),
    )
    assert hits == 3


def test_supersede_drops_content_out_of_active_without_losing_it(conn):
    """Last year's guidance stays citable for last year - it just stops being
    current advice."""
    store.upsert_chunks(conn, make_chunks())

    changed = store.supersede_document(conn, DOC, "doc-new")

    assert changed == 3
    assert count(conn, "AND active") == 0
    assert count(conn) == 3
    replacement = conn.execute(
        "SELECT DISTINCT superseded_by FROM chunks WHERE doc_id = %s", (DOC,)
    ).fetchone()[0]
    assert replacement == "doc-new"


def test_delete_by_doc_is_a_soft_delete(conn):
    store.upsert_chunks(conn, make_chunks())

    VectorIndex(conn).delete_by_doc(DOC)

    assert count(conn) == 3, "rows must survive a delete"
    assert count(conn, "AND active") == 0
    status = conn.execute(
        "SELECT DISTINCT status FROM chunks WHERE doc_id = %s", (DOC,)
    ).fetchone()[0]
    assert status == str(Status.DELETED)


def test_purge_really_removes_rows(conn):
    store.upsert_chunks(conn, make_chunks())

    removed = store.purge_document(conn, DOC)

    assert removed == 3
    assert count(conn) == 0


def test_stale_chunk_ids_finds_the_tail_a_shrinking_document_leaves(conn):
    """A document that loses a section would otherwise leave orphan chunks
    still answering queries."""
    store.upsert_chunks(conn, make_chunks(5))

    keep = [c.chunk_id for c in make_chunks(2)]
    orphans = store.stale_chunk_ids(conn, DOC, keep)

    assert sorted(orphans) == [f"{DOC}#2", f"{DOC}#3", f"{DOC}#4"]


def test_vector_index_refuses_to_write_vectors(conn):
    """Two writers on one column would eventually disagree with the text."""
    with pytest.raises(ValueError, match="does not write vectors"):
        VectorIndex(conn).upsert(make_chunks(), vectors=[[0.0] * 1536])
