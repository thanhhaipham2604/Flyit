"""Read and validate scraped ATO Markdown documents.

This module is responsible for the loading side of Zone A ingestion:

- discover Markdown source files
- parse the standard scraper header
- separate scraper metadata from page content
- load menu-tree breadcrumb metadata
- derive category/topic information from corpus paths
- generate deterministic document identifiers
- generate SHA-256 content hashes
- expose a reusable RawDocument representation

Cleaning, financial-year inference, chunking, embedding and indexing belong to
later ingestion stages and are intentionally kept out of this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Scraper-header patterns
# --------------------------------------------------------------------------- #

# Expected corpus structure:
#
#   # Page title | Australian Taxation Office
#
#   > **Source:** https://www.ato.gov.au/...
#   > **Scraped:** 2026-04-14 09:58:51
#   > **Menu Path:** Category > Topic > Page
#
#   **Description:** Optional page description
#
#   ---
#
#   Page body...
#
HEADER_RE = re.compile(
    r"^#\s*(?P<title>.+?)\s*\n"
    r".*?"
    r"^>\s*\*\*Source:\*\*\s*(?P<source>\S+)\s*$"
    r".*?"
    r"^>\s*\*\*Scraped:\*\*\s*(?P<scraped>.+?)\s*$"
    r".*?"
    r"^>\s*\*\*Menu Path:\*\*\s*(?P<menu_path>.+?)\s*$"
    r"(?:.*?^\*\*Description:\*\*\s*(?P<description>.+?)\s*$)?"
    r".*?"
    r"^---\s*$",
    re.MULTILINE | re.DOTALL,
)


# Remove the website suffix from the scraped H1 page title.
TITLE_SUFFIX_RE = re.compile(
    r"\s*\|\s*Australian Taxation Office\s*$"
)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class RawDocument:
    """Validated ATO Markdown source document.

    Attributes
    ----------
    doc_id:
        Deterministic SHA-1 identifier derived from the source URL, with a
        relative-path fallback when a URL is unavailable.

    content_hash:
        SHA-256 hash of the raw Markdown source text.

        The final version/index hash is calculated from cleaned content later
        in the ingestion pipeline so scraper-only changes can be handled
        separately.

    path:
        Original Markdown source path.

    text:
        Raw Markdown source text, including the scraper header.

    metadata:
        Parsed source metadata used by downstream preprocessing stages.
    """

    doc_id: str
    content_hash: str
    path: Path
    text: str
    metadata: dict = field(
        default_factory=dict
    )


# --------------------------------------------------------------------------- #
# Menu-tree lookup
# --------------------------------------------------------------------------- #


def load_menu_tree(
    path: Path | None,
) -> dict:
    """Load ``menu_tree.json`` for breadcrumb enrichment.

    The menu tree maps complete ATO source URLs to their hierarchy.

    Missing or malformed menu-tree data does not invalidate the Markdown
    corpus. In that case an empty dictionary is returned and ingestion can
    continue without breadcrumb enrichment.
    """

    if not path or not path.exists():
        log.warning(
            "menu_tree.json not found at %s - "
            "breadcrumb enrichment disabled",
            path,
        )

        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

    except (
        json.JSONDecodeError,
        OSError,
    ) as exc:
        log.warning(
            "Could not parse menu_tree.json (%s) - "
            "continuing without it",
            exc,
        )

        return {}

    if not isinstance(
        data,
        dict,
    ):
        log.warning(
            "menu_tree.json does not contain a JSON object - "
            "breadcrumb enrichment disabled"
        )

        return {}

    log.info(
        "Loaded menu_tree.json with %d URL entries",
        len(data),
    )

    return data


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #


def discover_md_files(
    root: Path,
) -> list[Path]:
    """Recursively discover Markdown files beneath ``root``.

    Results are sorted so repeated ingestion runs process the corpus in a
    deterministic order.
    """

    files = sorted(
        root.rglob("*.md")
    )

    log.info(
        "Discovered %d .md files under %s",
        len(files),
        root,
    )

    return files


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #


def parse_header_and_body(
    raw_text: str,
) -> tuple[
    dict | None,
    str,
]:
    """Split one scraped Markdown document into header metadata and body.

    Returns
    -------
    tuple
        ``(header, body)`` when the expected scraper header is found.

        ``(None, raw_text)`` when the file does not match the expected corpus
        structure.
    """

    match = HEADER_RE.search(
        raw_text
    )

    if not match:
        return (
            None,
            raw_text,
        )

    header = {
        "title":
            TITLE_SUFFIX_RE.sub(
                "",
                match.group("title"),
            ).strip(),

        "source_url":
            match.group(
                "source"
            ).strip(),

        "scraped_at":
            match.group(
                "scraped"
            ).strip(),

        "menu_path_text":
            match.group(
                "menu_path"
            ).strip(),

        "description":
            (
                match.group(
                    "description"
                )
                or ""
            ).strip()
            or None,
    }

    body = raw_text[
        match.end():
    ]

    return (
        header,
        body,
    )


# --------------------------------------------------------------------------- #
# Stable identifiers and hashes
# --------------------------------------------------------------------------- #


def stable_id(
    source_url: str | None,
    fallback_path: str,
) -> str:
    """Generate a deterministic document identifier.

    The ATO source URL is preferred because filenames can change independently
    of the source page.

    If a source URL is unavailable, the relative file path is used as a stable
    fallback.
    """

    key = (
        source_url
        if source_url
        else f"path:{fallback_path}"
    )

    return hashlib.sha1(
        key.encode(
            "utf-8"
        )
    ).hexdigest()


def content_hash(
    text: str,
) -> str:
    """Return the SHA-256 hash of ``text``.

    The helper is intentionally generic. The loader uses it for raw source
    content, while later pipeline stages can use the same function for cleaned
    content when performing incremental version detection.
    """

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Category / topic extraction
# --------------------------------------------------------------------------- #


def category_and_topic(
    file_path: Path,
    root: Path,
) -> tuple[
    str,
    str | None,
]:
    """Derive the corpus category and topic from a relative source path."""

    relative_parts = (
        file_path
        .relative_to(root)
        .parts
    )

    category = (
        relative_parts[0]
        if relative_parts
        else "unknown"
    )

    topic = (
        relative_parts[1]
        if len(relative_parts) > 2
        else None
    )

    return (
        category,
        topic,
    )


# --------------------------------------------------------------------------- #
# Corpus loader
# --------------------------------------------------------------------------- #


def load_corpus(
    corpus_dir: Path,
) -> list[RawDocument]:
    """Read and validate the Markdown corpus.

    This is a strict convenience loader for the package API.

    Every discovered ``.md`` document must contain the expected scraper
    header. A malformed source raises ``ValueError`` instead of being silently
    accepted.

    The more detailed pipeline layer can later use the lower-level functions
    in this module when it needs to collect invalid-file reports rather than
    fail immediately.
    """

    corpus_dir = Path(
        corpus_dir
    )

    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: "
            f"{corpus_dir}"
        )

    if not corpus_dir.is_dir():
        raise NotADirectoryError(
            f"Corpus path is not a directory: "
            f"{corpus_dir}"
        )

    files = discover_md_files(
        corpus_dir
    )

    if not files:
        return []

    menu_tree = load_menu_tree(
        corpus_dir
        / "menu_tree.json"
    )

    documents: list[
        RawDocument
    ] = []

    for file_path in files:
        relative_path = str(
            file_path.relative_to(
                corpus_dir
            )
        )

        try:
            raw_text = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except OSError as exc:
            raise OSError(
                "Could not read Markdown file "
                f"{relative_path}: {exc}"
            ) from exc

        header, _ = parse_header_and_body(
            raw_text
        )

        if header is None:
            raise ValueError(
                "Expected ATO scraper header "
                f"not found in: {relative_path}"
            )

        category, topic = (
            category_and_topic(
                file_path,
                corpus_dir,
            )
        )

        source_url = header[
            "source_url"
        ]

        menu_path_tree = (
            menu_tree.get(
                source_url
            )
            if source_url
            else None
        )

        doc_id = stable_id(
            source_url,
            relative_path,
        )

        raw_hash = content_hash(
            raw_text
        )

        metadata = {
            "file_path":
                relative_path,

            "category":
                category,

            "topic":
                topic,

            "title":
                header[
                    "title"
                ],

            "description":
                header.get(
                    "description"
                ),

            "source_url":
                source_url,

            "menu_path_text":
                header[
                    "menu_path_text"
                ],

            "menu_path_tree":
                menu_path_tree,

            "scraped_at":
                header[
                    "scraped_at"
                ],
        }

        documents.append(
            RawDocument(
                doc_id=doc_id,
                content_hash=raw_hash,
                path=file_path,
                text=raw_text,
                metadata=metadata,
            )
        )

    log.info(
        "Loaded %d valid ATO Markdown documents",
        len(documents),
    )

    return documents


__all__ = [
    "HEADER_RE",
    "TITLE_SUFFIX_RE",
    "RawDocument",
    "category_and_topic",
    "content_hash",
    "discover_md_files",
    "load_corpus",
    "load_menu_tree",
    "parse_header_and_body",
    "stable_id",
]