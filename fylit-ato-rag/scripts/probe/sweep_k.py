"""Sweep the evidence count K and measure what it costs and buys.

Raising K from 5 to 15 made rate questions answerable and made citations
noisier. This measures both sides at each K so the value can be chosen on
evidence rather than picked.

Requires `evidence_count` in config.Settings and EVIDENCE_COUNT honoured
from .env (docker-compose passes env_file: .env through to the api).

    python sweep_k.py

For each K it reports:
  answered     did it answer rather than refuse
  correct      did the answer contain an expected figure (where known)
  sources      average number of citations returned
  precision    share of citations whose title looks relevant to the question

Precision is approximate - it matches title keywords - but it is consistent
across K, which is what makes the comparison meaningful.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request

API = "http://localhost:8000"
ENV_FILE = ".env"
KS = [15, 20, 25, 30]

# (question, expected figures or None, keywords a relevant source title has)
QUESTIONS: list[tuple[str, list[str] | None, list[str]]] = [
    ("Can I claim working from home expenses?",
     None, ["working from home", "home", "work-related"]),
    ("What is the tax-free threshold in Australia?",
     ["18,200"], ["tax-free threshold", "threshold"]),
    ("What are the individual income tax rates for Australian residents?",
     ["18,200"], ["tax rates", "resident", "rates"]),
    ("What is the HELP repayment income threshold?",
     ["67,000", "54,435"], ["study", "training", "loan", "help", "repayment"]),
    ("How do I apply for a tax file number?",
     None, ["tax file number", "tfn"]),
    ("What is the Medicare levy and how much is it?",
     ["2%"], ["medicare"]),
    ("Can I claim car expenses for work travel?",
     None, ["car", "vehicle", "travel", "transport"]),
    ("When is the tax return deadline in Australia?",
     ["31 October"], ["lodg", "due", "return"]),
    ("Is income from driving for Uber taxable?",
     None, ["ride", "sharing", "sourcing", "gig", "business"]),
    ("Is cryptocurrency subject to capital gains tax?",
     None, ["crypto", "capital gains", "cgt"]),
    ("Can I claim self-education expenses?",
     None, ["self-education", "education", "study", "course"]),
    ("What is the superannuation guarantee rate?",
     None, ["super"]),
    ("Can I claim laundry costs for a work uniform?",
     None, ["clothing", "uniform", "laundry", "work-related"]),
    ("Do I need to lodge a tax return if I earned under the tax-free threshold?",
     None, ["lodg", "return", "threshold"]),
    ("I bought a laptop for $1,800 for work. Can I claim the full amount?",
     ["300"], ["tool", "equipment", "deprecia", "decline", "work-related"]),
    ("What records do I need to keep for my tax return?",
     None, ["record", "evidence", "receipt"]),
]


def read_env() -> list[str]:
    try:
        with open(ENV_FILE, encoding="utf-8") as fh:
            return fh.read().splitlines()
    except FileNotFoundError:
        return []


def set_k(k: int) -> None:
    lines = [ln for ln in read_env() if not ln.startswith("EVIDENCE_COUNT")]
    lines.append(f"EVIDENCE_COUNT={k}")
    with open(ENV_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def restart() -> None:
    subprocess.run(
        ["docker", "compose", "up", "-d", "--force-recreate", "api"],
        check=True, capture_output=True,
    )


def wait_ready(timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{API}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


def confirm_k(k: int) -> bool:
    """Make sure the container really picked the value up."""
    out = subprocess.run(
        ["docker", "exec", "fylit-ato-rag-api-1", "printenv", "EVIDENCE_COUNT"],
        capture_output=True, text=True,
    )
    return out.stdout.strip() == str(k)


def ask(question: str) -> dict | None:
    req = urllib.request.Request(
        f"{API}/ask",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception:  # noqa: BLE001
        return None


def run_one(k: int) -> dict:
    answered = correct = scored = 0
    source_counts: list[int] = []
    relevant = total_sources = 0

    for question, figures, keywords in QUESTIONS:
        body = ask(question)
        if body is None:
            continue
        diag = body.get("diagnostics") or {}
        if diag.get("refused"):
            continue
        answered += 1

        answer = body.get("answer") or ""
        if figures:
            scored += 1
            if any(f.lower() in answer.lower() for f in figures):
                correct += 1

        sources = body.get("useful_resources") or []
        source_counts.append(len(sources))
        for s in sources:
            total_sources += 1
            title = (s.get("title") or "").lower()
            if any(kw in title for kw in keywords):
                relevant += 1

    return {
        "k": k,
        "answered": answered,
        "of": len(QUESTIONS),
        "correct": correct,
        "scored": scored,
        "avg_sources": (sum(source_counts) / len(source_counts)) if source_counts else 0.0,
        "precision": (relevant / total_sources) if total_sources else 0.0,
    }


def main() -> None:
    original = next(
        (ln.split("=", 1)[1] for ln in read_env() if ln.startswith("EVIDENCE_COUNT")),
        None,
    )
    results = []
    try:
        for k in KS:
            print(f"\n--- K = {k} " + "-" * 40)
            set_k(k)
            restart()
            if not wait_ready():
                print("  api did not come up, skipping")
                continue
            if not confirm_k(k):
                print(f"  WARNING container does not report EVIDENCE_COUNT={k}")
            row = run_one(k)
            results.append(row)
            print(f"  answered {row['answered']}/{row['of']}"
                  f"  correct {row['correct']}/{row['scored']}"
                  f"  sources {row['avg_sources']:.1f}"
                  f"  precision {row['precision']:.0%}")
    finally:
        if original is not None:
            set_k(int(original))
            restart()
            print(f"\nrestored EVIDENCE_COUNT={original}")

    print("\n" + "=" * 58)
    print(f"{'K':>4}  {'answered':>9}  {'correct':>8}  {'sources':>8}  {'precision':>10}")
    for r in results:
        print(f"{r['k']:>4}  {r['answered']:>4}/{r['of']:<4}  "
              f"{r['correct']:>3}/{r['scored']:<4}  "
              f"{r['avg_sources']:>8.1f}  {r['precision']:>9.0%}")

    with open("sweep_k.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("\n-> sweep_k.json")


if __name__ == "__main__":
    main()
