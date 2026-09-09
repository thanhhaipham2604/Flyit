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
from fylit_rag.openai_client import client

app = typer.Typer(help="Generate a synthetic labelled question set from the index")

MIN_CHUNK_CHARS = 400
PROMPT = (
    "You write realistic questions that Australian taxpayers ask the ATO. "
    "Given a passage of ATO guidance, write ONE natural question that this passage "
    "answers. Write it the way a taxpayer would - plain words, no ATO jargon, no "
    "reference to 'the passage' or 'this page'. Respond with JSON: "
    '{"question": "..."}'
)


@app.command()
def build(
    n: int = typer.Option(150, help="How many questions to generate"),
    out: str = typer.Option("data/eval/questions.jsonl", help="Where to write the set"),
    seed: int = typer.Option(42, help="Sampling seed, so the set is reproducible"),
) -> None:
    """Sample chunks and generate one question each."""
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




# Questions with no answer in an ATO corpus. Two kinds, because they fail
# differently and a refusal metric that only sees one kind is easy to game:
#
#   off-topic      - obviously not tax. Retrieval returns nothing similar, so
#                    the grounding gate catches these on similarity alone.
#   out-of-scope   - genuinely tax-shaped, and genuinely not in this corpus:
#                    other countries' regimes, invented schemes, professional
#                    advice the ATO does not publish. These are the hard ones -
#                    the vocabulary overlaps, so retrieval finds *something*.
UNANSWERABLE = [
    # off-topic
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "Who won the 2019 Melbourne Cup?",
    "Write me a poem about the ocean.",
    "How do I fix a leaking tap?",
    "What are the rules of cricket?",
    "How do I train for a marathon?",
    "Recommend a good science fiction novel.",
    # out-of-scope but tax-shaped
    "How does the United Kingdom's IR35 off-payroll rule work?",
    "What is the corporate tax rate in Singapore for 2025?",
    "How do I file a US Form 1040-NR as an Australian resident?",
    "What is New Zealand's GST registration threshold?",
    "How does Canada's TFSA contribution room carry forward?",
    "What are the ATO's internal audit selection algorithms?",
    "Which tax agent in Sydney should I hire?",
    "What will the tax-free threshold be in 2035?",
    "How much revenue did the ATO collect from cryptocurrency audits in 2026?",
    "What is the penalty under section 999-99 of the ITAA 1997?",
]


@app.command()
def unanswerable(
    out: str = typer.Option("data/eval/questions.jsonl", help="Question set to append to"),
) -> None:
    """Append the unanswerable questions, so refusal rate can be measured.

    Without these the eval set is all answerable and a system that never refuses
    scores perfectly - which is the failure mode the guide calls out, since
    refusing when the evidence is thin is a feature rather than a shortfall.
    """
    path = Path(out)
    existing = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            existing = [json.loads(line) for line in fh if line.strip()]

    already = {r["question"] for r in existing if not r.get("answerable", True)}
    added = [
        {"question": q, "answerable": False, "chunk_id": None, "doc_id": None,
         "source_title": None}
        for q in UNANSWERABLE
        if q not in already
    ]

    # Anything already in the file predates this flag and is answerable.
    for row in existing:
        row.setdefault("answerable", True)

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row) + "\n" for row in existing + added)

    answerable_n = sum(1 for r in existing if r.get("answerable", True))
    typer.echo(
        f"Wrote {len(existing) + len(added)} questions to {path} "
        f"({answerable_n} answerable, {len(already) + len(added)} unanswerable)"
    )

if __name__ == "__main__":
    app()
