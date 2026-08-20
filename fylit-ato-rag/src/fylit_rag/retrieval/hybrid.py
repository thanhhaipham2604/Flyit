"""Hybrid retrieval: vector + keyword search blended (e.g. RRF).

Always filtered by financial year, version, and active status.
Every result keeps source title, URL and document version attached -
citations depend on it.
"""


def retrieve(query: str, top_k: int = 20, filters: dict | None = None):
    """TODO: run both searches, blend with reciprocal rank fusion, dedupe by chunk."""
    raise NotImplementedError
