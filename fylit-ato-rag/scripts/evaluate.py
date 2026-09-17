"""Retrieval + answer quality evaluation.

Tracks the guide's quality targets:
- retrieval: does the right chunk land near the top? (recall@k, MRR)
- grounding: do answer claims trace to retrieved passages?
- refusals: high refusal rate on genuinely unanswerable questions is a feature
- responsiveness: latency under concurrent load

Usage: python scripts/evaluate.py [--questions data/eval/questions.jsonl]
                                  [--strategies fusion,mmr,llm] [--top-n 5]

Four commands, one per target:

    run        retrieval quality - recall@k and MRR, comparing rerankers
    calibrate  where the refusal threshold belongs, from data
    answers    grounding, refusals and citations over the full answer path
    load       responsiveness under concurrent load, over HTTP

`run` and `calibrate` score retrieval, so they use only the answerable
questions - an unanswerable one has no correct chunk and would score as a miss
against every configuration equally, dragging recall down while telling us
nothing.

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

    # Unanswerable questions have no correct chunk; scoring recall against them
    # would penalise every configuration identically and mean nothing.
    items = [q for q in load_questions(path) if q.get("answerable", True)]
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


OFF_TOPIC = [
    "What is the airspeed velocity of an unladen swallow?",
    "Who won the 2019 Melbourne Cup?",
    "How do I bake sourdough bread?",
    "What is the capital of France?",
    "Write me a poem about the ocean.",
    "How do I fix a leaking tap?",
    "What are the rules of cricket?",
    "Recommend a good science fiction novel.",
    "How do I train for a marathon?",
    "What is the best programming language to learn?",
]


@app.command()
def calibrate(
    questions: str = typer.Option("data/eval/questions.jsonl", help="Answerable questions"),
    limit: int = typer.Option(60, help="How many answerable questions to sample"),
    pool: int = typer.Option(5, help="Shortlist size the grounding gate sees"),
) -> None:
    """Choose the refusal threshold from data rather than by feel.

    Measures the top vector similarity for answerable questions against
    deliberately off-topic ones, and reports what each candidate threshold would
    keep and refuse. The two populations should separate with a visible gap; the
    threshold belongs in that gap, nearer the off-topic end - the answerable set
    is synthetic and scores higher than real questions will.
    """
    from fylit_rag.generation.grounding import MIN_TOP_SIMILARITY, evidence_strength

    path = Path(questions)
    if not path.exists():
        typer.echo(f"No question set at {path}. Run scripts/build_eval_set.py first.")
        raise typer.Exit(code=1)

    answerable = [
        q["question"] for q in load_questions(path) if q.get("answerable", True)
    ][:limit]

    def tops(items, conn):
        out = []
        for q in items:
            strength = evidence_strength(retrieve(q, top_k=pool, conn=conn))
            if strength["top"] is not None:
                out.append(strength["top"])
        return out

    with connect() as conn:
        good, bad = tops(answerable, conn), tops(OFF_TOPIC, conn)

    if not good or not bad:
        typer.echo("Not enough measurements - is the index populated?")
        raise typer.Exit(code=1)

    pct = lambda xs, p: sorted(xs)[int(len(xs) * p)]
    typer.echo(
        f"answerable (n={len(good)}): min {min(good):.3f}  p5 {pct(good, 0.05):.3f}  "
        f"median {statistics.median(good):.3f}  max {max(good):.3f}"
    )
    typer.echo(
        f"off-topic  (n={len(bad)}): min {min(bad):.3f}  "
        f"median {statistics.median(bad):.3f}  max {max(bad):.3f}"
    )
    typer.echo(f"\ncurrent MIN_TOP_SIMILARITY = {MIN_TOP_SIMILARITY}\n")
    typer.echo(f"{'threshold':>10}{'answerable kept':>18}{'off-topic refused':>20}")
    for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        kept = sum(1 for x in good if x >= t) / len(good)
        refused = sum(1 for x in bad if x < t) / len(bad)
        marker = "  <- current" if abs(t - MIN_TOP_SIMILARITY) < 1e-9 else ""
        typer.echo(f"{t:>10.2f}{kept:>17.1%}{refused:>19.1%}{marker}")


# --------------------------------------------------------------------------- #
# Answer quality: grounding, refusals, citations
# --------------------------------------------------------------------------- #


@app.command()
def answers(
    questions: str = typer.Option("data/eval/questions.jsonl", help="Labelled question set"),
    limit: int = typer.Option(40, help="Answerable questions to sample (0 = all)"),
    pool: int = typer.Option(30, help="Shortlist size before reranking"),
    evidence_n: int = typer.Option(5, help="Passages handed to the model"),
) -> None:
    """Run the full answer path and score grounding, refusals and citations.

    Costs one generation per question, so it samples by default. Every
    unanswerable question is always included - they are the point of the run and
    there are only a handful.

    Four things are measured, and they pull against each other on purpose:

    - **refusal rate on unanswerable** should be high. Refusing when the evidence
      is thin is a feature, not a shortfall.
    - **false refusal rate on answerable** is what that costs. A system that
      refuses everything scores perfectly on the first metric and is useless.
    - **grounding** checks that figures in the answer appear in the passages.
      Prose is not checked - see `generation.grounding.verify_grounding`.
    - **citations and disclaimer** must be present on every non-refused answer;
      the guide makes both non-negotiable.
    """
    from fylit_rag.generation.grounding import verify_grounding
    from fylit_rag.generation.llm import generate_answer
    from fylit_rag.generation.prompts import DISCLAIMER
    from fylit_rag.guardrails.input_guards import check_input
    from fylit_rag.guardrails.output_guards import check_output

    path = Path(questions)
    if not path.exists():
        typer.echo(f"No question set at {path}. Run scripts/build_eval_set.py first.")
        raise typer.Exit(code=1)

    items = load_questions(path)
    answerable = [q for q in items if q.get("answerable", True)]
    unanswerable = [q for q in items if not q.get("answerable", True)]
    if limit:
        answerable = answerable[:limit]
    if not unanswerable:
        typer.echo("No unanswerable questions in the set - refusal rate cannot be measured.")
        typer.echo("Run: python scripts/build_eval_set.py unanswerable")

    typer.echo(f"{len(answerable)} answerable + {len(unanswerable)} unanswerable\n")

    stats = {
        "answerable": {"n": 0, "refused": 0, "grounded": 0, "cited": 0, "disclaimed": 0},
        "unanswerable": {"n": 0, "refused": 0, "grounded": 0, "cited": 0, "disclaimed": 0},
    }
    blocked_by_input_guard = 0
    latencies = []

    with connect() as conn:
        for group, batch in (("answerable", answerable), ("unanswerable", unanswerable)):
            typer.echo(f"  running {group}...")
            for q in batch:
                bucket = stats[group]
                bucket["n"] += 1

                allowed, _ = check_input(q["question"])
                if not allowed:
                    # An input guard refusal counts as a refusal: from the
                    # caller's side the system declined, whatever stopped it.
                    blocked_by_input_guard += 1
                    bucket["refused"] += 1
                    continue

                started = time.perf_counter()
                candidates = retrieve(q["question"], top_k=pool, conn=conn)
                evidence = rerank(q["question"], candidates, top_n=evidence_n)
                result = check_output(generate_answer(q["question"], evidence))
                latencies.append(time.perf_counter() - started)

                if result["refused"]:
                    bucket["refused"] += 1
                    continue
                if verify_grounding(result["answer"], evidence):
                    bucket["grounded"] += 1
                if result.get("sources"):
                    bucket["cited"] += 1
                if DISCLAIMER in result["answer"] or result.get("disclaimer"):
                    bucket["disclaimed"] += 1

    def rate(bucket, key):
        return bucket[key] / bucket["n"] if bucket["n"] else float("nan")

    good, bad = stats["answerable"], stats["unanswerable"]
    answered = good["n"] - good["refused"]

    typer.echo("\nREFUSALS  (high on unanswerable is the goal; high on answerable is the cost)")
    typer.echo(f"  unanswerable refused : {rate(bad, 'refused'):6.1%}  ({bad['refused']}/{bad['n']})")
    typer.echo(f"  answerable refused   : {rate(good, 'refused'):6.1%}  ({good['refused']}/{good['n']})"
               "   <- false refusals")
    typer.echo(f"  stopped by input guard: {blocked_by_input_guard}")

    typer.echo("\nANSWER QUALITY  (of the answerable questions actually answered)")
    if answered:
        typer.echo(f"  figures grounded in passages : {good['grounded'] / answered:6.1%}")
        typer.echo(f"  sources attached             : {good['cited'] / answered:6.1%}")
        typer.echo(f"  disclaimer present           : {good['disclaimed'] / answered:6.1%}")
    else:
        typer.echo("  nothing was answered - every answerable question was refused")

    leaked = bad["n"] - bad["refused"]
    if leaked:
        typer.echo(f"\n  WARNING: {leaked} unanswerable question(s) got an answer")

    if latencies:
        typer.echo(f"\nLATENCY (sequential)  median {statistics.median(latencies):.2f}s  "
                   f"p95 {sorted(latencies)[int(len(latencies) * 0.95)]:.2f}s")


# --------------------------------------------------------------------------- #
# Responsiveness under concurrent load
# --------------------------------------------------------------------------- #


@app.command()
def load(
    url: str = typer.Option("http://localhost:8000", help="A running API"),
    concurrency: int = typer.Option(10, help="Simultaneous callers"),
    requests_each: int = typer.Option(3, help="Requests per caller"),
    questions: str = typer.Option("data/eval/questions.jsonl", help="Questions to draw from"),
) -> None:
    """Hit a running API concurrently and report latency, throughput and errors.

    Needs the service up (`docker compose up -d`). Deliberately measured over
    HTTP rather than by calling the pipeline in-process: what matters is what a
    client experiences, including the rate limiter and the threadpool - neither
    of which exists when you call the functions directly.

    429s are counted separately and are not errors. Being turned away politely
    under load is the rate limiter working.
    """
    import random
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    path = Path(questions)
    if not path.exists():
        typer.echo(f"No question set at {path}.")
        raise typer.Exit(code=1)
    pool_of_questions = [
        q["question"] for q in load_questions(path) if q.get("answerable", True)
    ]

    try:
        health = httpx.get(f"{url}/health", timeout=5)
        health.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - a missing service is a message, not a traceback
        typer.echo(f"No API at {url} ({type(exc).__name__}). Start it with: docker compose up -d")
        raise typer.Exit(code=1) from None

    rng = random.Random(42)
    plan = [rng.choice(pool_of_questions) for _ in range(concurrency * requests_each)]
    results: list[tuple[int, float]] = []

    def call(question: str) -> tuple[int, float, bool]:
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{url}/ask", json={"question": question}, timeout=60
            )
            refused = False
            if response.status_code == 200:
                refused = bool(response.json().get("diagnostics", {}).get("refused"))
            return response.status_code, time.perf_counter() - started, refused
        except Exception:  # noqa: BLE001 - a timeout is a result, not a crash
            return 0, time.perf_counter() - started, False

    typer.echo(f"{len(plan)} requests, {concurrency} concurrent, against {url}\n")
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool_exec:
        results = list(pool_exec.map(call, plan))
    wall = time.perf_counter() - wall_start

    # Refusals are separated from answers because they skip generation entirely.
    # Averaging them together hides the number that matters: a service refusing
    # everything would post an excellent median latency and be useless.
    answered = [d for code, d, refused in results if code == 200 and not refused]
    refusals = [d for code, d, refused in results if code == 200 and refused]
    limited = [d for code, d, _ in results if code == 429]
    failed = [d for code, d, _ in results if code not in (200, 429)]
    ordered = sorted(answered)

    typer.echo(f"  wall clock      : {wall:.1f}s")
    typer.echo(f"  throughput      : {len(results) / wall:.2f} req/s")
    typer.echo(f"  answered (200)  : {len(answered)}")
    typer.echo(f"  refused (200)   : {len(refusals)}   (no generation call - much faster)")
    typer.echo(f"  rate limited    : {len(limited)}   (not errors - the limiter working)")
    typer.echo(f"  failed          : {len(failed)}")
    if ordered:
        typer.echo(
            f"  latency (answers): median {statistics.median(ordered):.2f}s  "
            f"p95 {ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]:.2f}s  "
            f"max {max(ordered):.2f}s"
        )
    else:
        typer.echo("  latency (answers): nothing was answered - every request was refused")
    if refusals:
        typer.echo(f"  latency (refusals): median {statistics.median(sorted(refusals)):.2f}s")
    if failed:
        typer.echo("\n  WARNING: requests failed outright - see the API logs")


if __name__ == "__main__":
    app()
