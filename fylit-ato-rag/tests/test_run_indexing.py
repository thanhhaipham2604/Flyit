"""Wiring tests: the incremental behaviour the demo has to show.

Exercises `index_documents` against a real database, but without running
preprocessing or the embedding API - the documents are handed in directly, so
these stay fast and cost nothing.

Skipped when no database is reachable, like the store tests.
"""

import pytest

from fylit_rag.indexing import store
from fylit_rag.indexing.run_indexing import index_documents, load_enriched

DOC_A = "_test_wiring_a"
DOC_B = "_test_wiring_b"


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
        c.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", ([DOC_A, DOC_B],))
        c.commit()
        yield c
        c.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", ([DOC_A, DOC_B],))
        c.commit()


def document(doc_id, sections=3, extra=""):
    body = "".join(
        f"## Section {i}\n\n" + ("Body text about withholding obligations. " * 12) + "\n\n"
        for i in range(sections)
    )
    return {
        "id": doc_id,
        "title": "A test document",
        "source_url": "https://ato.gov.au/test",
        "financial_years": [],
        "cleaned_content": f"# A test document\n\n{body}{extra}",
    }


def fill_embeddings(conn):
    conn.execute("UPDATE chunks SET embedding = %s WHERE embedding IS NULL", (str([0.1] * 1536),))


def count(conn, doc_id, where=""):
    return conn.execute(
        f"SELECT count(*) FROM chunks WHERE doc_id = %s {where}", (doc_id,)
    ).fetchone()[0]


def test_a_new_document_is_indexed_at_version_one(conn):
    docs = {DOC_A: document(DOC_A)}
    report = index_documents(conn, docs, [DOC_A], [])

    assert report.chunks_written > 0
    assert report.embeddings_invalidated == report.chunks_written, "new chunks need vectors"
    assert store.current_version(conn, DOC_A) == 1


def test_reindexing_unchanged_content_reuses_every_vector(conn):
    """The claim the whole pipeline rests on: a re-run costs nothing."""
    docs = {DOC_A: document(DOC_A)}
    index_documents(conn, docs, [DOC_A], [])
    fill_embeddings(conn)

    report = index_documents(conn, docs, [DOC_A], [])

    assert report.embeddings_invalidated == 0
    assert count(conn, DOC_A, "AND embedding IS NULL") == 0


def test_editing_a_document_re_embeds_only_what_moved(conn):
    """Appending a section must not invalidate the sections above it."""
    docs = {DOC_A: document(DOC_A)}
    index_documents(conn, docs, [DOC_A], [])
    fill_embeddings(conn)
    before = count(conn, DOC_A)

    edited = {DOC_A: document(DOC_A, extra="## New section\n\n" + ("Fresh body text. " * 20))}
    report = index_documents(conn, edited, [], [DOC_A])

    assert report.chunks_written > before, "the new section should add a chunk"
    assert report.embeddings_invalidated < report.chunks_written, "untouched chunks kept vectors"


def test_a_changed_document_gets_a_new_version(conn):
    docs = {DOC_A: document(DOC_A)}
    index_documents(conn, docs, [DOC_A], [])
    assert store.current_version(conn, DOC_A) == 1

    index_documents(conn, {DOC_A: document(DOC_A, extra="More.")}, [], [DOC_A])

    assert store.current_version(conn, DOC_A) == 2


def test_a_full_rebuild_does_not_inflate_versions(conn):
    """--full-rebuild reports every document as new; that is not a content
    change, so the version must stay put."""
    docs = {DOC_A: document(DOC_A)}
    index_documents(conn, docs, [DOC_A], [])

    index_documents(conn, docs, [DOC_A], [])          # reported new again

    assert store.current_version(conn, DOC_A) == 1


def test_a_shrinking_document_leaves_no_orphan_chunks(conn):
    """A removed section must stop answering queries."""
    index_documents(conn, {DOC_A: document(DOC_A, sections=6)}, [DOC_A], [])
    before = count(conn, DOC_A)

    report = index_documents(conn, {DOC_A: document(DOC_A, sections=2)}, [], [DOC_A])

    assert report.orphans_removed > 0
    assert count(conn, DOC_A) < before
    assert count(conn, DOC_A) == report.chunks_written


def test_one_bad_document_does_not_stop_the_run(conn):
    """A diff naming a document the corpus no longer holds is reported, not
    raised - one bad document must not block thousands of good ones."""
    docs = {DOC_A: document(DOC_A)}
    report = index_documents(conn, docs, [DOC_A, "_missing_doc"], [])

    assert report.chunks_written > 0
    assert any("_missing_doc" in e for e in report.errors)


def test_documents_are_indexed_independently(conn):
    docs = {DOC_A: document(DOC_A), DOC_B: document(DOC_B)}
    index_documents(conn, docs, [DOC_A, DOC_B], [])
    fill_embeddings(conn)

    edited = dict(docs)
    edited[DOC_A] = document(DOC_A, extra="Changed.")
    index_documents(conn, edited, [], [DOC_A])

    assert count(conn, DOC_B, "AND embedding IS NULL") == 0, "B must be untouched"


def test_load_enriched_reads_jsonl(tmp_path):
    path = tmp_path / "enriched_corpus.jsonl"
    path.write_text('{"id": "a", "cleaned_content": "x"}\n\n{"id": "b"}\n', encoding="utf-8")

    docs = load_enriched(path)

    assert set(docs) == {"a", "b"}
    assert docs["a"]["cleaned_content"] == "x"
