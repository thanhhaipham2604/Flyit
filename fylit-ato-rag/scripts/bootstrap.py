"""CLI entry point: create the Postgres schema for both retrieval halves.

Usage: python scripts/bootstrap.py [--recreate]

Safe to re-run. `--recreate` drops the chunks table first and is exactly as
destructive as it sounds.
"""

import typer

from fylit_rag.indexing.bootstrap import bootstrap

app = typer.Typer(help="Create the chunks table, its indexes and the vector extension")


@app.command()
def run(
    recreate: bool = typer.Option(
        False, "--recreate", help="Drop and rebuild the chunks table. Deletes all indexed chunks."
    ),
) -> None:
    if recreate:
        typer.confirm("This deletes every indexed chunk. Continue?", abort=True)
    typer.echo(bootstrap(recreate=recreate).summary())


if __name__ == "__main__":
    app()
