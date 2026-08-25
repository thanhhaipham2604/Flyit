"""Vector half of retrieval: the `embedding` column (pgvector, HNSW/cosine).

Schema lives in `schema.py` - this reads and writes `chunks` rows, it does not
define them. Keep the interface thin so the backend can be swapped (ADR-0002
records what we gave up to get here).
"""


class VectorIndex:
    def upsert(self, chunks, vectors) -> None:
        raise NotImplementedError

    def search(self, query_vector, top_k: int = 20, filters: dict | None = None):
        """TODO: filter by financial_year / version / active."""
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> None:
        raise NotImplementedError
