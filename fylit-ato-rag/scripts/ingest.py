"""CLI entry point: take the corpus all the way to a populated, embedded index.

Usage: python scripts/ingest.py [--corpus-dir data/ato_corpus] [--full-rebuild]
                                [--no-embed]

This runs preprocessing *and* indexing. Preprocessing on its own - cleaning,
hashing, duplicate detection, year enrichment, no database - is still available
as `python -m fylit_rag.ingestion.pipeline`.
"""

from fylit_rag.indexing.run_indexing import app

if __name__ == "__main__":
    app()
