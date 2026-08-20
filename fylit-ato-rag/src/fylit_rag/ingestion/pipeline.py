"""Incremental, versioned ingestion pipeline (Zone A orchestrator).

Handles new, changed, deleted and superseded content: compares content
hashes against the index manifest and only (re)processes what changed.
Adding a new financial year must NOT force a full rebuild.
"""

import typer

app = typer.Typer(help="Fylit ATO corpus ingestion")


@app.command()
def run(corpus_dir: str = "data/ato_corpus", full_rebuild: bool = False) -> None:
    """Load -> validate -> clean -> chunk -> embed -> index (incrementally).

    TODO:
    - diff corpus against stored manifest (doc_id + content_hash)
    - process only new/changed docs; tombstone deleted/superseded ones
    - bump index version; record ingestion report
    """
    raise NotImplementedError


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
