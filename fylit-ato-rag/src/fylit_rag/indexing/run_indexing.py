"""Take the corpus all the way to a populated, embedded index.

    preprocess -> diff -> chunk changed docs -> upsert -> embed

This is the wiring between `ingestion.pipeline`, which deliberately stops
before chunking, and the `chunks` table. It is also where "incremental" stops
being a property of individual functions and becomes a property of the system:

- Unchanged documents are never chunked, never written and never embedded. The
  preprocessing diff decides that from content hashes, so a re-run over an
  untouched corpus does no API work at all.
- A changed document is re-chunked in full and upserted. Only the chunks whose
  own text moved lose their embedding (see `store.upsert_chunks`), so editing
  one paragraph of a long page re-embeds one chunk, not the page.
- A document that lost a section leaves orphan chunks behind. Those are found
  by comparing the ids we just wrote against what the index holds, and removed -
  otherwise deleted text keeps answering queries.
- A document that left the corpus is marked deleted, not erased: `schema.Status`
  keeps it citable for the year it covered.

Version numbers come from the index itself and increment when content changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import typer

from fylit_rag.indexing import store
from fylit_rag.indexing.bootstrap import connect, wait_for_postgres
from fylit_rag.ingestion.chunker import chunk_document
from fylit_rag.ingestion.pipeline import run_preprocessing


@dataclass
class IndexReport:
    """What one indexing run did, in the terms the guide asks us to demonstrate."""

    new: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks_written: int = 0
    orphans_removed: int = 0
    embeddings_invalidated: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> list[str]:
        reused = self.chunks_written - self.embeddings_invalidated
        return [
            (
                f"documents : {self.new} new, {self.changed} changed, "
                f"{self.unchanged} unchanged, {self.deleted} deleted"
            ),
            f"chunks    : {self.chunks_written} written, {self.orphans_removed} orphans removed",
            f"embeddings: {self.embeddings_invalidated} to compute ({reused} vectors reused)",
        ]


def load_enriched(path: Path) -> dict[str, dict]:
    """Read the enriched corpus JSONL into {doc_id: document}."""
    docs: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docs[doc["id"]] = doc
    return docs


def index_documents(
    conn,
    docs: dict[str, dict],
    new_ids: list[str],
    changed_ids: list[str],
) -> IndexReport:
    """Chunk and upsert the named documents, cleaning up orphan chunks.

    `new_ids` and `changed_ids` are kept apart because only a content change
    earns a new version number. A `--full-rebuild` reports every document as
    new even though nothing changed, so incrementing on anything else would
    inflate versions on a run that altered nothing.

    Takes an open connection so a caller can wrap a whole run in one
    transaction, and so tests can hand it a throwaway database.
    """
    report = IndexReport()
    changed = set(changed_ids)

    for doc_id in [*new_ids, *changed_ids]:
        doc = docs.get(doc_id)
        if doc is None:
            # The diff named a document the corpus no longer contains. Report it
            # rather than crashing the run - one bad document should not stop
            # 5,000 good ones being indexed.
            report.errors.append(f"{doc_id}: named by the diff but absent from the corpus")
            continue

        indexed = store.current_version(conn, doc_id)
        version = indexed + 1 if doc_id in changed else (indexed or 1)
        chunks = chunk_document(
            doc_id,
            doc.get("cleaned_content") or "",
            doc,
            version=version,
        )
        if not chunks:
            report.errors.append(f"{doc_id}: produced no chunks")
            continue

        result = store.upsert_chunks(conn, chunks)
        report.chunks_written += result.written
        report.embeddings_invalidated += result.embeddings_cleared

        orphans = store.stale_chunk_ids(conn, doc_id, [c.chunk_id for c in chunks])
        report.orphans_removed += store.purge_chunks(conn, orphans)

    return report


def run(
    corpus_dir: str = "data/ato_corpus",
    output_dir: str = "data/processed",
    full_rebuild: bool = False,
    embed: bool = True,
) -> IndexReport:
    """Preprocess, index the changed documents, then fill in the embeddings."""
    result = run_preprocessing(
        corpus_dir=Path(corpus_dir),
        output_dir=Path(output_dir),
        full_rebuild=full_rebuild,
    )
    diff = result["incremental_diff"]
    docs = load_enriched(Path(result["paths"]["enriched_corpus"]))

    wait_for_postgres()
    with connect() as conn:
        report = index_documents(conn, docs, diff["new_ids"], diff["changed_ids"])
        report.new = diff["new"]
        report.changed = diff["changed"]
        report.unchanged = diff["unchanged"]

        for doc_id in diff["deleted_ids"]:
            store.mark_deleted(conn, doc_id)
        report.deleted = diff["deleted"]

        conn.commit()

    for line in report.summary():
        print(line)
    for err in report.errors:
        print(f"  ! {err}")

    if embed and report.embeddings_invalidated:
        # Deliberately after the commit: embedding is slow, paid and external,
        # so an interruption there must leave a complete index with some vectors
        # missing - which the next run finishes - not a half-written index.
        from fylit_rag.indexing.run_embeddings import run as run_embeddings

        run_embeddings()
    elif embed:
        print("No embeddings needed - every chunk already has a current vector.")

    return report


app = typer.Typer(help="Preprocess the corpus, index the changed documents, embed what moved")


@app.command()
def cli(
    corpus_dir: str = typer.Option("data/ato_corpus", help="ATO Markdown corpus root"),
    output_dir: str = typer.Option("data/processed", help="Where preprocessing artefacts go"),
    full_rebuild: bool = typer.Option(
        False, "--full-rebuild", help="Ignore previous state and re-chunk every document"
    ),
    no_embed: bool = typer.Option(
        False, "--no-embed", help="Index only; leave embeddings for a later run"
    ),
) -> None:
    """Take the corpus to a populated, embedded index."""
    report = run(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        full_rebuild=full_rebuild,
        embed=not no_embed,
    )
    if report.errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
