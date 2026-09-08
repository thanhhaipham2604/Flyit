"""Vector half of retrieval: the `embedding` column (pgvector, HNSW/cosine).

Schema lives in `schema.py` - this reads and writes `chunks` rows, it does not
define them. Keep the interface thin so the backend can be swapped (ADR-0002
records what we gave up to get here).

Writes go through `indexing.store`, which is the single writer for the shared
row. Vectors themselves are filled in afterwards by `indexing.run_embeddings`,
not here: embedding is a slow, paid, retryable step, and keeping it out of the
row write means an interrupted embedding run never leaves the index half-written.
"""

from __future__ import annotations

import psycopg

from fylit_rag.indexing.schema import ChunkRecord
from fylit_rag.indexing.search import (
    RESULT_COLUMNS,
    SearchResult,
    build_where,
    table_name,
)
from fylit_rag.indexing.store import UpsertReport, mark_deleted, upsert_chunks


def _as_vector(query_vector) -> str:
    """pgvector accepts its text form, which works whether or not the caller has
    run `register_vector` on this connection."""
    if isinstance(query_vector, str):
        return query_vector
    return "[" + ",".join(repr(float(v)) for v in query_vector) + "]"


class VectorIndex:
    """The vector half. Construct with an open connection."""

    def __init__(self, conn: psycopg.Connection, table: str | None = None):
        self.conn = conn
        self.table = table

    def upsert(self, chunks: list[ChunkRecord], vectors=None) -> UpsertReport:
        """Write chunk rows, preserving embeddings whose text has not changed.

        `vectors` is accepted for interface compatibility and must be None:
        embeddings are attached by `run_embeddings`, which finds rows whose
        embedding is NULL. Passing vectors here would put the same column under
        two writers, and the second one would eventually disagree with the text.
        """
        if vectors is not None:
            raise ValueError(
                "VectorIndex.upsert does not write vectors; run indexing.run_embeddings "
                "after upserting, which embeds exactly the rows whose text changed."
            )
        return upsert_chunks(self.conn, chunks, table=self.table)

    def search(
        self,
        query_vector,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Nearest chunks by cosine distance, newest-guidance-only by default.

        Scores are cosine *similarity* (1 - distance) so that higher is better
        in both halves of retrieval - the fusion in `retrieval.hybrid` ranks, not
        thresholds, but a score that means opposite things in the two lists is a
        trap waiting for whoever reads them next.

        Rows without an embedding are excluded rather than ranked last: a chunk
        awaiting `run_embeddings` has no position in vector space at all.
        """
        where, params = build_where(filters)
        table = table_name(self.table)
        columns = ", ".join(f'"{c}"' for c in RESULT_COLUMNS)

        sql = (
            f"SELECT {columns}, embedding <=> %s::vector AS distance "
            f"FROM {table} "
            f"WHERE embedding IS NOT NULL AND {where} "
            f"ORDER BY distance LIMIT %s"
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, [_as_vector(query_vector), *params, top_k])
            return [
                SearchResult.from_row(row[:-1], 1.0 - float(row[-1]), rank, "vector")
                for rank, row in enumerate(cur.fetchall(), start=1)
            ]

    def delete_by_doc(self, doc_id: str) -> int:
        """Mark a document's chunks deleted.

        A soft delete: `schema.Status` keeps `deleted` distinct from row removal
        so content stays citable for the year it covered. Use
        `store.purge_document` when rows must genuinely go.
        """
        return mark_deleted(self.conn, doc_id, table=self.table)
