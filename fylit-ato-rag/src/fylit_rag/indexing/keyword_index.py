"""Keyword (BM25) index backend (default: OpenSearch - see ADR-0002).

Exact words and phrases: names, section numbers, precise terms.
Same metadata schema as the vector index so filters behave identically.
"""


class KeywordIndex:
    def upsert(self, chunks) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 20, filters: dict | None = None):
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> None:
        raise NotImplementedError
