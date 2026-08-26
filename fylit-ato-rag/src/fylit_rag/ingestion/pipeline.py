"""Phase-1 ATO preprocessing pipeline.

This module orchestrates the validated Zone A preprocessing stages:

    source discovery
        ->
    scraper-header parsing
        ->
    conservative Markdown cleaning
        ->
    cleaned-content hashing
        ->
    duplicate detection
        ->
    incremental state comparison
        ->
    financial-year enrichment
        ->
    preprocessing reports

The pipeline deliberately stops before structure-aware chunking, embeddings,
PostgreSQL/pgvector indexing and retrieval. Those are later ingestion phases.

Incremental ingestion compares stable document IDs and cleaned-content hashes.
An unchanged document therefore does not need to be reprocessed by later
indexing stages.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import typer

from fylit_rag.indexing.versioning import (
    diff_against_state,
    find_content_duplicates,
    load_state,
)

from fylit_rag.ingestion.cleaner import clean_body

from fylit_rag.ingestion.loader import (
    category_and_topic,
    content_hash,
    discover_md_files,
    load_menu_tree,
    parse_header_and_body,
    stable_id,
)

from fylit_rag.ingestion.metadata import (
    process_corpus as enrich_corpus_year_metadata,
)


log = logging.getLogger(__name__)


app = typer.Typer(
    help="Fylit ATO corpus ingestion"
)


# --------------------------------------------------------------------------- #
# Phase-1 constants
# --------------------------------------------------------------------------- #

# Documents with fewer than this many useful characters after cleaning are
# considered malformed or empty.
MIN_BODY_CHARS = 40


# This is retained for the legacy ``financial_years`` field in
# cleaned_corpus.jsonl.
#
# The authoritative primary-year inference is performed later by
# ``ingestion.metadata``.
#
FINANCIAL_YEAR_RE = re.compile(
    r"\b(20\d{2})\s*[-\u2013\u2014]\s*(\d{2})\b"
)


# --------------------------------------------------------------------------- #
# Cleaned-document representation
# --------------------------------------------------------------------------- #


@dataclass
class CleanedDoc:
    """One validated and cleaned ATO source document.

    This representation intentionally matches the existing Phase-1
    cleaned_corpus.jsonl structure so the package refactor remains backward
    compatible with previously generated preprocessing outputs.
    """

    id: str
    content_hash: str
    file_path: str
    category: str
    topic: Optional[str]
    title: str
    description: Optional[str]
    source_url: Optional[str]
    menu_path_text: Optional[str]
    menu_path_tree: Optional[list]
    last_updated_display: Optional[str]
    published_display: Optional[str]
    scraped_at: Optional[str]
    financial_years: list = field(
        default_factory=list
    )
    char_count: int = 0
    word_count: int = 0
    cleaned_content: str = ""


# --------------------------------------------------------------------------- #
# Legacy financial-year mention field
# --------------------------------------------------------------------------- #


def extract_financial_years(
    *texts: str,
) -> list[str]:
    """Extract explicit YYYY-YY financial-year mentions.

    This function exists to preserve the historical ``financial_years`` field
    in ``cleaned_corpus.jsonl``.

    It is only a mention extractor. It does NOT determine which financial year
    applies to the page.

    Primary financial-year inference is handled by
    ``fylit_rag.ingestion.metadata``.
    """

    years: set[str] = set()

    for text in texts:
        if not text:
            continue

        for match in FINANCIAL_YEAR_RE.finditer(
            text
        ):
            years.add(
                f"{match.group(1)}-"
                f"{match.group(2)}"
            )

    return sorted(
        years
    )


# --------------------------------------------------------------------------- #
# Corpus-root resolution
# --------------------------------------------------------------------------- #


def resolve_corpus_root(
    corpus_dir: Path,
) -> Path:
    """Resolve the actual ATO corpus root.

    The repository may be configured with either:

        data/ato_corpus

    or directly with:

        data/ato_corpus/atoData

    If ``atoData`` exists beneath the supplied directory and contains the
    menu-tree file, that child directory is used automatically.
    """

    root = Path(
        corpus_dir
    )

    candidate = (
        root / "atoData"
    )

    if (
        not (
            root / "menu_tree.json"
        ).exists()
        and candidate.is_dir()
        and (
            candidate / "menu_tree.json"
        ).exists()
    ):
        return candidate

    return root


# --------------------------------------------------------------------------- #
# Individual source-file preparation
# --------------------------------------------------------------------------- #


def process_file(
    file_path: Path,
    root: Path,
    menu_tree: dict,
    invalid: list,
) -> Optional[CleanedDoc]:
    """Parse, clean, enrich and validate one Markdown source file.

    Invalid files are appended to ``invalid`` rather than stopping the entire
    corpus run.
    """

    relative_path = str(
        file_path.relative_to(
            root
        )
    )

    # ------------------------------------------------------------------ #
    # Read source
    # ------------------------------------------------------------------ #

    try:
        raw_text = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except OSError as exc:
        invalid.append(
            {
                "file_path":
                    relative_path,
                "reason":
                    f"read error: {exc}",
            }
        )

        return None

    # ------------------------------------------------------------------ #
    # Parse scraper header
    # ------------------------------------------------------------------ #

    header, body = parse_header_and_body(
        raw_text
    )

    if header is None:
        invalid.append(
            {
                "file_path":
                    relative_path,
                "reason":
                    "header pattern not found",
            }
        )

        return None

    # ------------------------------------------------------------------ #
    # Clean Markdown body
    # ------------------------------------------------------------------ #

    (
        cleaned,
        last_updated,
        published,
    ) = clean_body(
        body
    )

    if len(cleaned) < MIN_BODY_CHARS:
        invalid.append(
            {
                "file_path":
                    relative_path,
                "reason":
                    (
                        "cleaned body too short "
                        f"({len(cleaned)} chars)"
                    ),
            }
        )

        return None

    # ------------------------------------------------------------------ #
    # Category / topic
    # ------------------------------------------------------------------ #

    category, topic = category_and_topic(
        file_path,
        root,
    )

    # ------------------------------------------------------------------ #
    # Menu-tree enrichment
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Explicit FY mentions
    # ------------------------------------------------------------------ #

    financial_years = extract_financial_years(
        header["title"],
        header.get(
            "description"
        )
        or "",
        cleaned,
    )

    # ------------------------------------------------------------------ #
    # Stable identity and cleaned-content hash
    # ------------------------------------------------------------------ #

    doc_id = stable_id(
        source_url,
        relative_path,
    )

    doc_hash = content_hash(
        cleaned
    )

    # ------------------------------------------------------------------ #
    # Cleaned document
    # ------------------------------------------------------------------ #

    return CleanedDoc(
        id=doc_id,
        content_hash=doc_hash,
        file_path=relative_path,
        category=category,
        topic=topic,
        title=header["title"],
        description=header.get(
            "description"
        ),
        source_url=source_url,
        menu_path_text=header[
            "menu_path_text"
        ],
        menu_path_tree=menu_path_tree,
        last_updated_display=last_updated,
        published_display=published,
        scraped_at=header[
            "scraped_at"
        ],
        financial_years=financial_years,
        char_count=len(
            cleaned
        ),
        word_count=len(
            cleaned.split()
        ),
        cleaned_content=cleaned,
    )


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #


def _write_json(
    path: Path,
    value,
) -> None:
    """Write a human-readable UTF-8 JSON file."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            indent=2,
        )


# --------------------------------------------------------------------------- #
# Phase-1 orchestration
# --------------------------------------------------------------------------- #


def run_preprocessing(
    corpus_dir: Path = Path(
        "data/ato_corpus"
    ),
    output_dir: Path = Path(
        "data/processed"
    ),
    full_rebuild: bool = False,
) -> dict:
    """Run the complete validated Phase-1 preprocessing workflow.

    Parameters
    ----------
    corpus_dir:
        ATO Markdown corpus root. Both ``data/ato_corpus`` and
        ``data/ato_corpus/atoData`` layouts are supported.

    output_dir:
        Directory for generated preprocessing artefacts.

    full_rebuild:
        When ``True``, ignore the previous state and treat the current corpus
        as a fresh ingestion run.

    Returns
    -------
    dict
        Processing statistics and output locations.
    """

    input_root = resolve_corpus_root(
        Path(
            corpus_dir
        )
    )

    output_root = Path(
        output_dir
    )

    # ------------------------------------------------------------------ #
    # Validate input
    # ------------------------------------------------------------------ #

    if not input_root.exists():
        raise FileNotFoundError(
            "Input corpus folder does not exist: "
            f"{input_root}"
        )

    if not input_root.is_dir():
        raise NotADirectoryError(
            "Input corpus path is not a directory: "
            f"{input_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------ #
    # Menu-tree metadata
    # ------------------------------------------------------------------ #

    menu_tree = load_menu_tree(
        input_root
        / "menu_tree.json"
    )

    # ------------------------------------------------------------------ #
    # Discover source files
    # ------------------------------------------------------------------ #

    md_files = discover_md_files(
        input_root
    )

    if not md_files:
        raise ValueError(
            "No .md files found under "
            f"{input_root}"
        )

    # ------------------------------------------------------------------ #
    # Parse and clean corpus
    # ------------------------------------------------------------------ #

    invalid: list = []
    docs: list[CleanedDoc] = []

    for index, file_path in enumerate(
        md_files,
        start=1,
    ):
        doc = process_file(
            file_path=file_path,
            root=input_root,
            menu_tree=menu_tree,
            invalid=invalid,
        )

        if doc is not None:
            docs.append(
                doc
            )

        if (
            index % 500 == 0
            or index == len(
                md_files
            )
        ):
            log.info(
                "Processed %d / %d files "
                "(%d valid so far)",
                index,
                len(md_files),
                len(docs),
            )

    # ------------------------------------------------------------------ #
    # Duplicate cleaned content
    # ------------------------------------------------------------------ #

    duplicates = find_content_duplicates(
        docs
    )

    duplicate_ids = {
        doc_id
        for ids in duplicates.values()
        for doc_id in ids
    }

    # ------------------------------------------------------------------ #
    # Incremental state comparison
    # ------------------------------------------------------------------ #

    state_path = (
        output_root
        / "state.json"
    )

    if full_rebuild:
        previous_state: dict[str, str] = {}

    else:
        previous_state = load_state(
            state_path
        )

    diff = diff_against_state(
        docs,
        previous_state,
    )

    # ------------------------------------------------------------------ #
    # cleaned_corpus.jsonl
    # ------------------------------------------------------------------ #

    cleaned_corpus_path = (
        output_root
        / "cleaned_corpus.jsonl"
    )

    with cleaned_corpus_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for doc in docs:
            record = asdict(
                doc
            )

            record[
                "is_duplicate_content"
            ] = (
                doc.id
                in duplicate_ids
            )

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    log.info(
        "Wrote %d cleaned documents to %s",
        len(docs),
        cleaned_corpus_path,
    )

    # ------------------------------------------------------------------ #
    # state.json
    # ------------------------------------------------------------------ #

    new_state = {
        doc.id:
            doc.content_hash
        for doc in docs
    }

    _write_json(
        state_path,
        new_state,
    )

    # ------------------------------------------------------------------ #
    # invalid_files.json
    # ------------------------------------------------------------------ #

    invalid_path = (
        output_root
        / "invalid_files.json"
    )

    _write_json(
        invalid_path,
        invalid,
    )

    # ------------------------------------------------------------------ #
    # duplicate_content.json
    # ------------------------------------------------------------------ #

    duplicate_path = (
        output_root
        / "duplicate_content.json"
    )

    _write_json(
        duplicate_path,
        duplicates,
    )

    # ------------------------------------------------------------------ #
    # Counts by category
    # ------------------------------------------------------------------ #

    by_category: dict[
        str,
        int,
    ] = {}

    for doc in docs:
        by_category[
            doc.category
        ] = (
            by_category.get(
                doc.category,
                0,
            )
            + 1
        )

    # ------------------------------------------------------------------ #
    # manifest.json
    # ------------------------------------------------------------------ #

    manifest = {
        "input_root":
            str(
                input_root
            ),

        "total_files_discovered":
            len(
                md_files
            ),

        "valid_documents":
            len(
                docs
            ),

        "invalid_files":
            len(
                invalid
            ),

        "duplicate_content_groups":
            len(
                duplicates
            ),

        "documents_by_category":
            by_category,

        "incremental_diff_vs_previous_run":
            diff,
    }

    manifest_path = (
        output_root
        / "manifest.json"
    )

    _write_json(
        manifest_path,
        manifest,
    )

    # ------------------------------------------------------------------ #
    # Financial-year enrichment
    # ------------------------------------------------------------------ #

    enriched_corpus_path = (
        output_root
        / "enriched_corpus.jsonl"
    )

    year_summary_path = (
        output_root
        / "year_tagging_summary.json"
    )

    year_summary = (
        enrich_corpus_year_metadata(
            input_path=cleaned_corpus_path,
            output_path=enriched_corpus_path,
            summary_path=year_summary_path,
        )
    )

    # ------------------------------------------------------------------ #
    # Return programmatic summary
    # ------------------------------------------------------------------ #

    return {
        "input_root":
            str(
                input_root
            ),

        "output_root":
            str(
                output_root
            ),

        "total_files_discovered":
            len(
                md_files
            ),

        "valid_documents":
            len(
                docs
            ),

        "invalid_files":
            len(
                invalid
            ),

        "duplicate_content_groups":
            len(
                duplicates
            ),

        "incremental_diff":
            diff,

        "year_tagging":
            year_summary,

        "paths":
            {
                "cleaned_corpus":
                    str(
                        cleaned_corpus_path
                    ),

                "enriched_corpus":
                    str(
                        enriched_corpus_path
                    ),

                "state":
                    str(
                        state_path
                    ),

                "manifest":
                    str(
                        manifest_path
                    ),

                "invalid_files":
                    str(
                        invalid_path
                    ),

                "duplicate_content":
                    str(
                        duplicate_path
                    ),

                "year_tagging_summary":
                    str(
                        year_summary_path
                    ),
            },
    }


# --------------------------------------------------------------------------- #
# Typer CLI
# --------------------------------------------------------------------------- #


@app.command()
def run(
    corpus_dir: str = "data/ato_corpus",
    full_rebuild: bool = False,
    output_dir: str = "data/processed",
) -> None:
    """Run Phase-1 ATO corpus preprocessing.

    Phase 1 performs loading, validation, cleaning, metadata enrichment,
    duplicate detection and incremental change analysis.

    Chunking, embeddings and database indexing are intentionally handled by
    later pipeline stages.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)-7s | "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    try:
        result = run_preprocessing(
            corpus_dir=Path(
                corpus_dir
            ),
            output_dir=Path(
                output_dir
            ),
            full_rebuild=full_rebuild,
        )

    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as exc:
        typer.echo(
            f"ERROR: {exc}",
            err=True,
        )

        raise typer.Exit(
            code=1
        ) from exc

    diff = result[
        "incremental_diff"
    ]

    year_summary = result[
        "year_tagging"
    ]

    typer.echo(
        ""
    )

    typer.echo(
        "Phase-1 preprocessing complete."
    )

    typer.echo(
        "Valid documents: "
        f"{result['valid_documents']}"
    )

    typer.echo(
        "Invalid files: "
        f"{result['invalid_files']}"
    )

    typer.echo(
        "Duplicate-content groups: "
        f"{result['duplicate_content_groups']}"
    )

    typer.echo(
        "Incremental diff: "
        f"new={diff['new']} "
        f"changed={diff['changed']} "
        f"unchanged={diff['unchanged']} "
        f"deleted={diff['deleted']}"
    )

    typer.echo(
        "Primary FY assigned: "
        f"{year_summary[
            'documents_with_primary_financial_year'
        ]}"
    )

    typer.echo(
        "FY coverage: "
        f"{year_summary['coverage_pct']}%"
    )

    typer.echo(
        "Outputs: "
        f"{result['output_root']}"
    )


def cli() -> None:
    """Launch the command-line interface."""

    app()


if __name__ == "__main__":
    cli()


__all__ = [
    "CleanedDoc",
    "FINANCIAL_YEAR_RE",
    "MIN_BODY_CHARS",
    "extract_financial_years",
    "process_file",
    "resolve_corpus_root",
    "run",
    "run_preprocessing",
]