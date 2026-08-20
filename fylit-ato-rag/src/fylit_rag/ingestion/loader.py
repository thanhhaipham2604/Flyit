"""Read and validate ATO Markdown files from a configurable folder.

Every document gets a stable ID and a content hash so the same file is
never processed twice (requirement: idempotent ingestion).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawDocument:
    doc_id: str          # stable ID derived from source path/URL
    content_hash: str    # sha256 of file contents
    path: Path
    text: str
    metadata: dict       # source title, URL, financial year, ...


def load_corpus(corpus_dir: Path) -> list[RawDocument]:
    """Walk `corpus_dir`, validate each .md file, return RawDocuments.

    TODO:
    - validate frontmatter / expected structure, reject malformed files with a report
    - derive stable doc_id (e.g. slug of source URL) and sha256 content hash
    - extract metadata: source title, URL, financial year
    """
    raise NotImplementedError
