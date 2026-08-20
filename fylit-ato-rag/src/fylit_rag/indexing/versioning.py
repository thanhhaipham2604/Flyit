"""Index version manifest.

Tracks per-document: doc_id, content_hash, version, status
(active | superseded | deleted), indexed_at. This is what makes
incremental ingestion and 'active-only' retrieval filters possible.
"""


class Manifest:
    def diff(self, current_docs) -> dict:
        """TODO: return {new: [...], changed: [...], deleted: [...]}."""
        raise NotImplementedError

    def mark_superseded(self, doc_id: str, by_doc_id: str) -> None:
        raise NotImplementedError
