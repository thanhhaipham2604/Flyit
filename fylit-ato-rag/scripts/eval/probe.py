"""Ask the running API a set of questions and report what it got wrong.

    python scripts/eval/probe.py realistic
    python scripts/eval/probe.py behaviour --json out.json
    python scripts/eval/probe.py realistic --repeat 3

Verdicts:
    OK      behaved as expected, and contained an expected value if one was given
    THIN    answered as expected, but none of the expected values appeared
    FAIL    answered when it should have refused, or refused when it should not
    REVIEW  marked "either" - a human decides
    ERROR   no usable response

THIN exists because "it answered" and "it answered correctly" are different
questions, and scoring only the first once hid an answer that gave rates from
the wrong financial year.

Use --repeat for anything you intend to quote. Generation is not
deterministic: the same set varies by about two questions run to run, so a
single run of 57 and a single run of 59 are the same result.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ask, contains_any, outcome, snippet  # noqa: E402
from questions import SETS  # noqa: E402


def verdict(expect: str, got: str, found: bool | None) -> str:
    if got == "error":
        return "ERROR"
    if expect == "either":
        return "REVIEW"
    if expect == "answer" and got == "answered":
        return "OK" if found is not False else "THIN"
    if expect == "refuse" and got == "refused":
        return "OK"
    return "FAIL"


def run(name: str) -> list[dict]:
    rows = []
    questions = SETS[name]
    for i, (cat, question, expect, contains, _kw, note) in enumerate(questions, 1):
        body = ask(question)
        got, guardrail, answer = outcome(body)
        found = contains_any(answer, contains)
        v = verdict(expect, got, found)
        rows.append({
            "category": cat, "question": question, "expect": expect,
            "got": got, "verdict": v, "guardrail": guardrail,
            "contains": contains or [], "note": note,
            "sources": [s.get("title") for s in (body.get("useful_resources") or [])],
            "answer": snippet(answer, 400),
        })
        print(f"[{i:2}/{len(questions)}] {v:6} {cat:16} {question[:46]}")
    return rows


def report(rows: list[dict]) -> None:
    print("\n" + "=" * 66)
    counts = collections.Counter(r["verdict"] for r in rows)
    for v in ("OK", "THIN", "FAIL", "REVIEW", "ERROR"):
        if counts.get(v):
            print(f"  {v:7} {counts[v]}")

    print("\nBY CATEGORY")
    cats: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r["verdict"])
    for cat, vs in sorted(cats.items()):
        ok = vs.count("OK")
        print(f"  {cat:16} {ok}/{len(vs)}")

    for label in ("FAIL", "THIN"):
        bad = [r for r in rows if r["verdict"] == label]
        if not bad:
            continue
        print("\n" + "=" * 66)
        print(f"{label} ({len(bad)})")
        for r in bad:
            print(f"\n  [{r['category']}] {r['question']}")
            tail = f" ({r['guardrail']})" if r["guardrail"] else ""
            print(f"    expected {r['expect']}, got {r['got']}{tail}"
                  f"{'  - ' + r['note'] if r['note'] else ''}")
            if r["contains"]:
                print(f"    looking for: {r['contains']}")
            print(f"    said: {r['answer'][:200]}")

    review = [r for r in rows if r["verdict"] == "REVIEW"]
    if review:
        print("\n" + "=" * 66)
        print("REVIEW - judge these by hand")
        for r in review:
            print(f"\n  [{r['got']}] {r['question']}")
            print(f"    said: {r['answer'][:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("set", choices=sorted(SETS), help="which question set")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run N times and report the spread (default 1)")
    parser.add_argument("--json", dest="out", help="write full results here")
    args = parser.parse_args()

    runs = []
    for n in range(args.repeat):
        if args.repeat > 1:
            print(f"\n######## run {n + 1} of {args.repeat} ########")
        rows = run(args.set)
        report(rows)
        runs.append(rows)

    if args.repeat > 1:
        print("\n" + "=" * 66)
        print("ACROSS RUNS")
        oks = [sum(1 for r in rows if r["verdict"] == "OK") for rows in runs]
        print(f"  OK per run: {oks}   spread {min(oks)}-{max(oks)}")
        flaky: dict[str, set[str]] = collections.defaultdict(set)
        for rows in runs:
            for r in rows:
                flaky[r["question"]].add(r["verdict"])
        unstable = {q: v for q, v in flaky.items() if len(v) > 1}
        if unstable:
            print(f"\n  unstable ({len(unstable)}) - these differ between runs:")
            for q, vs in unstable.items():
                print(f"    {sorted(vs)}  {q[:58]}")
        else:
            print("\n  every question gave the same verdict every run")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(runs if args.repeat > 1 else runs[0], fh, indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
