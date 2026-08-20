"""Cut cleaned documents into roughly paragraph-sized chunks.

Chunks are the unit of search AND citation, so each chunk carries:
source title, URL, financial year, heading path, and document version.
Never split a table or list mid-structure.
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_document(doc_id: str, cleaned_text: str, metadata: dict) -> list[Chunk]:
    """TODO: heading-aware chunking; keep tables/lists whole; attach metadata."""
    raise NotImplementedError
