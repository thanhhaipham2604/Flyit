"""CLI entry point: run the ingestion pipeline.

Usage: python scripts/ingest.py [--corpus-dir data/ato_corpus] [--full-rebuild]
"""

from fylit_rag.ingestion.pipeline import cli

if __name__ == "__main__":
    cli()
