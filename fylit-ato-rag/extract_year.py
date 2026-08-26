#!/usr/bin/env python3
"""CLI wrapper for Fylit financial-year metadata enrichment.

The reusable financial-year inference logic lives in:

    fylit_rag.ingestion.metadata

This script exists only as a convenient command-line entry point for the
Phase-1 preprocessing workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fylit_rag.ingestion.metadata import process_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively enrich the cleaned Fylit ATO corpus "
            "with financial-year metadata."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/cleaned_corpus.jsonl"
        ),
        help="Input cleaned corpus JSONL.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/enriched_corpus.jsonl"
        ),
        help="Output enriched corpus JSONL.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "data/processed/year_tagging_summary.json"
        ),
        help="Output JSON summary of financial-year tagging.",
    )

    args = parser.parse_args()

    summary = process_corpus(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
    )

    print(
        f"Processing complete: "
        f"{summary['total_documents']} documents processed."
    )

    print(
        "Primary financial year assigned: "
        f"{summary['documents_with_primary_financial_year']}"
    )

    print(
        "Left null: "
        f"{summary['documents_with_null_primary_financial_year']}"
    )

    print(
        f"Coverage: "
        f"{summary['coverage_pct']}%"
    )

    print(
        "Documents containing explicit FY mentions: "
        f"{summary['documents_with_financial_year_mentions']}"
    )

    print(
        "Null documents retaining FY mentions: "
        f"{summary['null_documents_with_financial_year_mentions']}"
    )

    print(
        f"Saved enriched corpus to: "
        f"{args.output}"
    )

    print(
        f"Saved summary to: "
        f"{args.summary}"
    )


if __name__ == "__main__":
    main()