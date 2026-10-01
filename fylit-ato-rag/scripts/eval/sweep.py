"""Vary one setting and measure what it costs and buys.

    python scripts/eval/sweep.py k
    python scripts/eval/sweep.py k --values 3,5,8,15
    python scripts/eval/sweep.py rerank

Each run writes the setting into .env, recreates the api container (no
rebuild - docker-compose passes env_file through), confirms the container
actually picked it up, then asks the SWEEP question set.

Reported per value:
    answered    answered rather than refused
    correct     the answer contained an expected figure, where one is known
    sources     mean number of citations returned
    precision   share of citations whose title looks relevant to the question
    retrieval   mean retrieval time, ms
    total       mean end-to-end time, ms

Precision is keyword matching on source titles, so the absolute figure is
approximate. It is measured identically at every value, which is what makes
the comparison meaningful - trust the trend, not the number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ask, contains_any, container_env, get_env, outcome, restart, set_env, wait_ready,
)
from questions import SWEEP  # noqa: E402

DIMENSIONS = {
    # name: (env var, default values)
    "k": ("EVIDENCE_COUNT", ["3", "5", "8", "15", "20", "30"]),
    "rerank": ("RERANK_STRATEGY", ["fusion", "mmr", "llm"]),
    "shortlist": ("SHORTLIST", ["10", "20", "30", "50"]),
}


def measure() -> dict:
    answered = correct = scored = 0
    relevant = total_sources = 0
    source_counts: list[int] = []
    retrieval: list[float] = []
    total: list[float] = []
    notes: list[str] = []

    for _cat, question, _expect, figures, keywords, _note in SWEEP:
        started = time.time()
        body = ask(question)
        total.append((time.time() - started) * 1000)

        got, guardrail, answer = outcome(body)
        diag = body.get("diagnostics") or {}
        if diag.get("retrieval_ms"):
            retrieval.append(diag["retrieval_ms"])

        if got != "answered":
            notes.append(f"{guardrail or got}: {question[:44]}")
            continue
        answered += 1

        found = contains_any(answer, figures)
        if found is not None:
            scored += 1
            if found:
                correct += 1
            else:
                notes.append(f"wrong figure: {question[:44]}")

        sources = body.get("useful_resources") or []
        source_counts.append(len(sources))
        for s in sources:
            total_sources += 1
            title = (s.get("title") or "").lower()
            if keywords and any(kw in title for kw in keywords):
                relevant += 1

    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    return {
        "answered": answered, "of": len(SWEEP),
        "correct": correct, "scored": scored,
        "sources": mean(source_counts),
        "precision": (relevant / total_sources) if total_sources else 0.0,
        "retrieval_ms": mean(retrieval), "total_ms": mean(total),
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dimension", choices=sorted(DIMENSIONS))
    parser.add_argument("--values", help="comma separated, overrides the default")
    parser.add_argument("--json", dest="out", help="write full results here")
    args = parser.parse_args()

    var, default = DIMENSIONS[args.dimension]
    values = args.values.split(",") if args.values else default
    original = get_env(var)

    results = []
    try:
        for value in values:
            print(f"\n--- {var}={value} " + "-" * 34)
            set_env(var, value)
            restart()
            if not wait_ready():
                print("  api did not come up, skipping")
                continue
            actual = container_env(var)
            if actual != value:
                print(f"  WARNING container reports {var}={actual!r}, wanted {value!r}")
            row = {"value": value, **measure()}
            results.append(row)
            print(f"  answered {row['answered']}/{row['of']}"
                  f"  correct {row['correct']}/{row['scored']}"
                  f"  sources {row['sources']:.1f}"
                  f"  precision {row['precision']:.0%}"
                  f"  retrieval {row['retrieval_ms']:.0f}ms"
                  f"  total {row['total_ms']:.0f}ms")
    finally:
        if original is not None:
            set_env(var, original)
            restart()
            print(f"\nrestored {var}={original}")

    print("\n" + "=" * 76)
    print(f"{var:>16}  {'answered':>9}  {'correct':>8}  {'sources':>8}"
          f"  {'precision':>10}  {'retrieval':>10}  {'total':>8}")
    for r in results:
        print(f"{r['value']:>16}  {r['answered']:>4}/{r['of']:<4}  "
              f"{r['correct']:>3}/{r['scored']:<4}  {r['sources']:>8.1f}"
              f"  {r['precision']:>9.0%}  {r['retrieval_ms']:>8.0f}ms"
              f"  {r['total_ms']:>6.0f}ms")

    for r in results:
        if r["notes"]:
            print(f"\n{var}={r['value']} - what went wrong:")
            for n in r["notes"]:
                print(f"  {n}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
