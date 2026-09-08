"""Write chunks into Postgres - the one place rows are created or changed.

Both retrieval halves live in the same row (ADR-0002), so there is one writer,
not two. `search_vector` is a GENERATED column, which means the keyword half is
maintained by the database the moment the text lands; nothing here writes it,
and nothing can let it drift from the text it indexes.

The embedding column is the opposite case - it is expensive and external, so it
is preserved across an upsert and cleared only when the chunk's own
`content_hash` changes. That single CASE expression is what makes indexing
incremental: re-running over an unchanged corpus rewrites metadata, keeps every
vector, and leaves `run_embeddings` with nothing to do.

Status is how content leaves the index. `deleted` and `superseded` are states,
not row removal, because the schema keeps last year's guidance citable for last
year (see `schema.Status`). `purge_document` is the only thing that truly
removes rows, and it exists for corpus mistakes, not for lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from fylit_rag.config import settings
from fylit_rag.indexing.schema import GENERATED_COLUMNS, ChunkRecord, Status


@dataclass(slots=True)
class UpsertReport:
    """What one upsert actually did, for the incremental-run log."""

    written: int = 0
    embeddings_cleared: int = 0

    def summary(self) -> str:
        return (
            f"{self.written} chunks written, {self.embeddings_cleared} embeddings invalidated "
            f"({self.written - self.embeddings_cleared} vectors kept)"
        )


def _columns() -> list[str]:
    """The columns an INSERT may set: every ChunkRecord field bar the generated
    ones. Derived from the record itself so adding a field cannot silently miss
    the writer."""
    probe = ChunkRecord(chunk_id="", doc_id="", chunk_ordinal=0, content_hash="", text="")
    return [c for c in probe.as_row() if c not in GENERATED_COLUMNS]


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: list[ChunkRecord],
    *,
    table: str | None = None,
) -> UpsertReport:
    """Insert or update chunk rows, preserving embeddings that are still valid.

    An existing row keeps its embedding unless its `content_hash` differs from
    the incoming one, in which case the vector is set to NULL so
    `indexing.run_embeddings` will recompute it. This is the contract that
    module's docstring asks for.

    Returns counts rather than printing: the caller decides what to report.
    """
    if not chunks:
        return UpsertReport()

    # Table name comes from our own settings, never from user input, so
    # interpolating it here is safe - psycopg cannot parameterise identifiers.
    table = table or settings.chunks_table
    cols = _columns()
    col_sql = ", ".join(f'"{c}"' for c in cols)
    marks = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c != "chunk_id")

    statement = f"""
        INSERT INTO {table} ({col_sql}) VALUES ({marks})
        ON CONFLICT (chunk_id) DO UPDATE SET
            {updates},
            embedding = CASE
                WHEN {table}.content_hash IS DISTINCT FROM EXCLUDED.content_hash THEN NULL
                ELSE {table}.embedding
            END
        RETURNING (xmax = 0) AS inserted, embedding IS NULL AS needs_embedding
    """

    report = UpsertReport()
    with conn.cursor() as cur:
        for chunk in chunks:
            row = chunk.as_row()
            cur.execute(statement, [row[c] for c in cols])
            _, needs_embedding = cur.fetchone()
            report.written += 1
            if needs_embedding:
                report.embeddings_cleared += 1
    return report


def set_document_status(
    conn: psycopg.Connection,
    doc_id: str,
    status: Status,
    *,
    superseded_by: str | None = None,
    table: str | None = None,
) -> int:
    """Move every chunk of a document to `status`. Returns rows affected.

    The generated `active` column follows `status`, so this is all retrieval
    needs to stop surfacing a document as current guidance - while the rows
    remain readable for a query scoped to the year they belong to.
    """
    table = table or settings.chunks_table
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET status = %s, superseded_by = %s WHERE doc_id = %s",
            (str(status), superseded_by, doc_id),
        )
        return cur.rowcount


def supersede_document(
    conn: psycopg.Connection,
    doc_id: str,
    superseded_by: str | None = None,
    *,
    table: str | None = None,
) -> int:
    """Mark a document's chunks superseded, optionally naming the replacement."""
    return set_document_status(
        conn, doc_id, Status.SUPERSEDED, superseded_by=superseded_by, table=table
    )


def mark_deleted(conn: psycopg.Connection, doc_id: str, *, table: str | None = None) -> int:
    """Mark a document's chunks deleted - it is gone from the corpus, not wrong."""
    return set_document_status(conn, doc_id, Status.DELETED, table=table)


def purge_document(conn: psycopg.Connection, doc_id: str, *, table: str | None = None) -> int:
    """Actually remove a document's rows. For corpus mistakes, not lifecycle.

    Deliberately separate from `mark_deleted`: losing the row loses the ability
    to answer a question about the year that content covered.
    """
    table = table or settings.chunks_table
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE doc_id = %s", (doc_id,))
        return cur.rowcount


def current_version(conn: psycopg.Connection, doc_id: str, *, table: str | None = None) -> int:
    """The version currently indexed for a document, or 0 if it is not indexed.

    Versions are read back from the index rather than from a side file on
    purpose: the index is what actually holds a version, so a manifest that
    disagreed with it would be the thing that is wrong. `manifest.json` from the
    preprocessing pipeline is a run report, not a version store.
    """
    table = table or settings.chunks_table
    with conn.cursor() as cur:
        cur.execute(f"SELECT max(version) FROM {table} WHERE doc_id = %s", (doc_id,))
        return cur.fetchone()[0] or 0


def purge_chunks(
    conn: psycopg.Connection,
    chunk_ids: list[str],
    *,
    table: str | None = None,
) -> int:
    """Remove specific chunk rows. Used for orphans, not for lifecycle.

    An orphan is a chunk the document no longer produces - a section that was
    deleted, so the text behind it is simply gone. Unlike a superseded document
    there is nothing left to cite, so the row goes rather than lingering in a
    status that still answers queries.
    """
    if not chunk_ids:
        return 0
    table = table or settings.chunks_table
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE chunk_id = ANY(%s)", (chunk_ids,))
        return cur.rowcount


def stale_chunk_ids(
    conn: psycopg.Connection,
    doc_id: str,
    keep: list[str],
    *,
    table: str | None = None,
) -> list[str]:
    """Chunk ids currently stored for `doc_id` that the new chunking no longer
    produces.

    A document that shrinks - a section removed, so it now yields 7 chunks where
    it yielded 12 - would otherwise leave chunks 7..11 behind as orphans that
    still answer queries. The caller decides whether to delete or mark them.
    """
    table = table or settings.chunks_table
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT chunk_id FROM {table} WHERE doc_id = %s AND NOT (chunk_id = ANY(%s))",
            (doc_id, keep),
        )
        return [r[0] for r in cur.fetchall()]
