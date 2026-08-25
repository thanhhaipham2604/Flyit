"""Keyword half of retrieval: the `search_vector` column (tsvector, GIN).

Exact words and phrases: names, section numbers, precise terms. Same table as
the vector half, so filters are literally the same columns and cannot disagree
(ADR-0002).
"""


class KeywordIndex:
    def upsert(self, chunks) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 20, filters: dict | None = None):
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> None:
        raise NotImplementedError
