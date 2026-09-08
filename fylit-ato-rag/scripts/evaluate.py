"""Retrieval + answer quality evaluation.

Tracks the guide's quality targets:
- retrieval: does the right chunk land near the top? (recall@k, MRR)
- grounding: do answer claims trace to retrieved passages?
- refusals: high refusal rate on genuinely unanswerable questions is a feature
- responsiveness: latency under concurrent load

Usage: python scripts/evaluate.py [--questions data/eval/questions.jsonl]
                                  [--strategies fusion,mmr,llm] [--top-n 5]

Currently implemented: the retrieval half. Grounding, refusal and load numbers
arrive with generation and the API.

Two metrics are reported, and the gap between them matters:

``chunk recall``
    Did the exact chunk the question was written from make the top N? Strict,
    and pessimistic on this corpus - the same guidance is restated across myTax
    2021/2022/2023 pages, so an equally correct chunk from a sibling page counts
    as a miss.

``doc recall``
    Did any chunk of the right document make the top N? Closer to what an answer
    actually needs, since a citation names a page.

MRR is over chunk rank, so a configuration that puts the right chunk first beats
one that merely gets it into the list.

The question set is synthetic (see scripts/build_eval_set.py) and flatters
retrieval in absolute terms. It is used here to *compare* configurations, where
that bias applies to all of them equally.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import typer

from fylit_rag.indexing.bootstrap import connect
from fylit_rag.indexing.embeddings import embed_texts
from fylit_rag.indexing.search import fetch_embeddings
from fylit_rag.retrieval.hybrid import retrieve
from fylit_rag.retrieval.rerank import STRATEGIES, rerank

app = typer.Typer(help="Compare retrieval configurations on a labelled question set")


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_strategy(conn, questions, strategy, top_n, pool):
    """Return metrics for one reranking strategy over the whole question set."""
    doc_outcomes: list[int] = []   # per question, for the paired comparison below
    chunk_hits, doc_hits, reciprocal_ranks, latencies = 0, 0, [], []

    for q in questions:
        # Retrieval is identical for every strategy, so it sits outside the
        # timer: what the ADR needs is the cost each reranker *adds* to a query.
        # (It would also be mistimed - the embedding cache means whichever
        # strategy runs first pays for every query vector and the rest do not.)
        candidates = retrieve(q["question"], top_k=pool, conn=conn)

        started = time.perf_counter()
        extra = {}
        if strategy == "mmr":
            extra = {
                "embeddings": fetch_embeddings(conn, [c.chunk_id for c in candidates]),
                "query_vector": embed_texts([q["question"]])[0],
            }
        top = rerank(q["question"], candidates, top_n=top_n, strategy=strategy, **extra)
        latencies.append(time.perf_counter() - started)

        chunk_ids = [c.chunk_id for c in top]
        doc_ids = [c.result.doc_id for c in top]

        if q["chunk_id"] in chunk_ids:
            chunk_hits += 1
            reciprocal_ranks.append(1 / (chunk_ids.index(q["chunk_id"]) + 1))
        else:
            reciprocal_ranks.append(0.0)
        hit = q["doc_id"] in doc_ids
        doc_hits += int(hit)
        doc_outcomes.append(int(hit))

    n = len(questions)
    return {
        "strategy": strategy,
        "chunk_recall": chunk_hits / n,
        "doc_recall": doc_hits / n,
        "mrr": statistics.mean(reciprocal_ranks),
        "median_latency": statistics.median(latencies),
        "doc_outcomes": doc_outcomes,
    }


@app.command()
def run(
    questions: str = typer.Option("data/eval/questions.jsonl", help="Labelled question set"),
    strategies: str = typer.Option(",".join(STRATEGIES), help="Comma-separated strategies"),
    top_n: int = typer.Option(5, help="How many results the answer would see"),
    pool: int = typer.Option(30, help="Shortlist size handed to the reranker"),
    limit: int = typer.Option(0, help="Evaluate only the first N questions (0 = all)"),
) -> None:
    """Score each reranking strategy and print a comparison table."""
    path = Path(questions)
    if not path.exists():
        typer.echo(f"No question set at {path}. Run scripts/build_eval_set.py first.")
        raise typer.Exit(code=1)

    items = load_questions(path)
    if limit:
        items = items[:limit]
    chosen = [s.strip() for s in strategies.split(",") if s.strip()]

    typer.echo(f"{len(items)} questions | top_n={top_n} | shortlist={pool}")
    typer.echo("Synthetic question set - absolute numbers are optimistic; compare across rows.\n")

    rows = []
    with connect() as conn:
        for strategy in chosen:
            typer.echo(f"  running {strategy}...")
            rows.append(evaluate_strategy(conn, items, strategy, top_n, pool))

    header = f"{'strategy':<10}{'chunk recall@' + str(top_n):>18}{'doc recall@' + str(top_n):>18}"
    header += f"{'MRR':>10}{'rerank latency':>17}"
    typer.echo("\n" + header)
    typer.echo("-" * len(header))
    for r in sorted(rows, key=lambda r: -r["doc_recall"]):
        typer.echo(
            f"{r['strategy']:<10}{r['chunk_recall']:>18.3f}{r['doc_recall']:>18.3f}"
            f"{r['mrr']:>10.3f}{r['median_latency'] * 1000:>15.0f}ms"
        )

    _paired_comparison(rows)


def _paired_comparison(rows: list[dict], baseline: str = "fusion") -> None:
    """Say whether a strategy's edge over the baseline is real or noise.

    Every strategy answers the same questions, so the comparison is paired and
    only the questions where the two disagree carry information (McNemar). With
    a question set this small a few points of recall is usually nothing, and an
    ADR that treats it as a result would be picking a reranker by coin flip.
    """
    base = next((r for r in rows if r["strategy"] == baseline), None)
    if base is None or len(rows) < 2:
        return

    typer.echo(f"\nPaired comparison against '{baseline}' (doc recall, McNemar):")
    for row in rows:
        if row["strategy"] == baseline:
            continue
        wins = sum(1 for a, b in zip(row["doc_outcomes"], base["doc_outcomes"], strict=True)
                   if a and not b)
        losses = sum(1 for a, b in zip(row["doc_outcomes"], base["doc_outcomes"], strict=True)
                     if b and not a)
        discordant = wins + losses
        if discordant == 0:
            typer.echo(f"  {row['strategy']:<8} identical on every question")
            continue
        # Two-sided exact binomial p under H0: a discordant pair is a coin flip.
        from math import comb
        k = min(wins, losses)
        p = min(1.0, 2 * sum(comb(discordant, i) for i in range(k + 1)) / 2 ** discordant)
        verdict = "significant" if p < 0.05 else "not significant"
        typer.echo(
            f"  {row['strategy']:<8} +{wins} / -{losses} of {discordant} disagreements, "
            f"p={p:.2f} ({verdict})"
        )


if __name__ == "__main__":
    app()
