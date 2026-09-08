"""Build a labelled question set for retrieval evaluation.

Usage: python scripts/build_eval_set.py [--n 150] [--out data/eval/questions.jsonl]

Method: sample chunks from the index, and for each one ask gpt-4o-mini to write
a question that chunk answers. The chunk it came from is then a known-correct
answer, which gives recall@k and MRR without anyone hand-labelling thousands of
pairs.

**This set is synthetic, and it flatters retrieval.** A question written *from* a
chunk shares that chunk's vocabulary, so lexical and semantic search both find it
more easily than they would find the answer to a question a taxpayer typed cold.
Treat the absolute numbers as an upper bound. What it is good for is *comparing*
retrieval configurations, because the bias applies equally to all of them - which
is exactly what choosing a reranker needs.

Two guards against measuring the wrong thing:

- chunks are sampled from distinct documents, so one repetitive page cannot
  dominate the set;
- very short chunks are skipped, because a question generated from two lines of
  boilerplate has no informative answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from fylit_rag.config import settings
from fylit_rag.indexing.bootstrap import connect

app = typer.Typer(help="Generate a synthetic labelled question set from the index")

MIN_CHUNK_CHARS = 400
PROMPT = (
    "You write realistic questions that Australian taxpayers ask the ATO. "
    "Given a passage of ATO guidance, write ONE natural question that this passage "
    "answers. Write it the way a taxpayer would - plain words, no ATO jargon, no "
    "reference to 'the passage' or 'this page'. Respond with JSON: "
    '{"question": "..."}'
)


def _client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    return OpenAI(api_key=settings.openai_api_key or None)


@app.command()
def build(
    n: int = typer.Option(150, help="How many questions to generate"),
    out: str = typer.Option("data/eval/questions.jsonl", help="Where to write the set"),
    seed: int = typer.Option(42, help="Sampling seed, so the set is reproducible"),
) -> None:
    """Sample chunks and generate one question each."""
    client = _client()
    rows = []

    with connect() as conn:
        # DISTINCT ON keeps one chunk per document; the ordering makes the
        # sample reproducible for a given seed rather than whatever the
        # planner happens to return.
        candidates = conn.execute(
            """
            SELECT DISTINCT ON (doc_id) chunk_id, doc_id, "text", source_title
            FROM chunks
            WHERE active AND length("text") >= %s AND embedding IS NOT NULL
            ORDER BY doc_id, md5(chunk_id || %s)
            """,
            (MIN_CHUNK_CHARS, str(seed)),
        ).fetchall()

    import random

    random.Random(seed).shuffle(candidates)
    candidates = candidates[:n]
    typer.echo(f"Sampled {len(candidates)} chunks from distinct documents")

    for i, (chunk_id, doc_id, text, title) in enumerate(candidates, start=1):
        try:
            response = client.chat.completions.create(
                model=settings.generation_model,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
            )
            question = json.loads(response.choices[0].message.content)["question"].strip()
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not lose the run
            typer.echo(f"  ! {chunk_id}: {type(exc).__name__}")
            continue

        rows.append(
            {
                "question": question,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source_title": title,
            }
        )
        if i % 25 == 0:
            typer.echo(f"  {i}/{len(candidates)}")

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row) + "\n" for row in rows)
    typer.echo(f"Wrote {len(rows)} questions to {path}")


if __name__ == "__main__":
    app()
