"""Vector index backend (default: Qdrant - see ADR-0002).

Documented schema per point: id, vector, payload {doc_id, chunk_id, source_title,
url, financial_year, version, active}. Keep the interface thin so the backend
can be swapped.
"""


class VectorIndex:
    def upsert(self, chunks, vectors) -> None:
        raise NotImplementedError

    def search(self, query_vector, top_k: int = 20, filters: dict | None = None):
        """TODO: filter by financial_year / version / active."""
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> None:
        raise NotImplementedError
