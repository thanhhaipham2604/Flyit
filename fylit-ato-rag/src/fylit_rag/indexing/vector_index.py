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
from fylit_rag.indexing.store import UpsertReport, mark_deleted, upsert_chunks


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

    def search(self, query_vector, top_k: int = 20, filters: dict | None = None):
        """TODO: filter by financial_year / version / active."""
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> int:
        """Mark a document's chunks deleted.

        A soft delete: `schema.Status` keeps `deleted` distinct from row removal
        so content stays citable for the year it covered. Use
        `store.purge_document` when rows must genuinely go.
        """
        return mark_deleted(self.conn, doc_id, table=self.table)
