"""Compatibility CLI for Fylit's Phase-1 ATO preprocessing pipeline.

The reusable preprocessing implementation now lives under:

    src/fylit_rag/

This root-level script is intentionally kept small so existing development
commands such as:

    python datacleaning.py

continue to work without maintaining a second copy of the ingestion logic.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fylit_rag.ingestion.pipeline import (
    run_preprocessing,
)

log = logging.getLogger("datacleaning")


def build_parser() -> argparse.ArgumentParser:
    """Build the legacy-compatible command-line parser."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/ato_corpus/atoData"
        ),
        help=(
            "Root folder of the ATO Markdown corpus. "
            "The parent data/ato_corpus folder is also accepted."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed"
        ),
        help=(
            "Folder used for generated Phase-1 preprocessing "
            "outputs."
        ),
    )

    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help=(
            "Ignore the previous state and treat all current "
            "documents as new for this run."
        ),
    )

    return parser


def main() -> int:
    """Run the canonical package preprocessing pipeline."""

    parser = build_parser()
    args = parser.parse_args()

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
            corpus_dir=args.input,
            output_dir=args.output,
            full_rebuild=args.full_rebuild,
        )

    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as exc:
        log.error(
            "%s",
            exc,
        )

        return 1

    diff = result[
        "incremental_diff"
    ]

    year_summary = result[
        "year_tagging"
    ]

    print()
    print(
        "Phase-1 preprocessing complete."
    )
    print(
        f"Valid documents: "
        f"{result['valid_documents']}"
    )
    print(
        f"Invalid files: "
        f"{result['invalid_files']}"
    )
    print(
        "Duplicate-content groups: "
        f"{result['duplicate_content_groups']}"
    )
    print(
        "Incremental diff: "
        f"new={diff['new']} "
        f"changed={diff['changed']} "
        f"unchanged={diff['unchanged']} "
        f"deleted={diff['deleted']}"
    )
    print(
        "Primary FY assigned: "
        f"{year_summary[
            'documents_with_primary_financial_year'
        ]}"
    )
    print(
        "FY coverage: "
        f"{year_summary['coverage_pct']}%"
    )
    print(
        f"Outputs: "
        f"{result['output_root']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )