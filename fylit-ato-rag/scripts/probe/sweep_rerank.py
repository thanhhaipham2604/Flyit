"""Compare the three reranking strategies on the same questions.

The brief asks for reranking approaches to be compared. `fusion` is the
current default and does no reranking at all - it keeps the order retrieval
produced. `mmr` re-sorts for relevance and diversity at no cost. `llm` scores
each candidate with a model call per query.

K and the citation limit are held fixed, so the only variable is the sort.

    python sweep_rerank.py

Reports, per strategy:
  answered    answered rather than refused
  correct     answer contained an expected figure, where one is known
  precision   share of shown citations whose title looks relevant
  retrieval   mean retrieval time, ms - llm reranking is not free
  total       mean end-to-end time, ms
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request

API = "http://localhost:8000"
ENV_FILE = ".env"
STRATEGIES = ["fusion", "mmr", "llm"]

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


def set_strategy(name: str) -> None:
    lines = [ln for ln in read_env() if not ln.startswith("RERANK_STRATEGY")]
    lines.append(f"RERANK_STRATEGY={name}")
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


def confirm(name: str) -> bool:
    out = subprocess.run(
        ["docker", "exec", "fylit-ato-rag-api-1", "printenv", "RERANK_STRATEGY"],
        capture_output=True, text=True,
    )
    return out.stdout.strip() == name


def ask(question: str) -> tuple[dict | None, float]:
    req = urllib.request.Request(
        f"{API}/ask",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read()), (time.time() - started) * 1000
    except Exception:  # noqa: BLE001
        return None, (time.time() - started) * 1000


def run_one(name: str) -> dict:
    answered = correct = scored = 0
    relevant = total_sources = 0
    retrieval_ms: list[float] = []
    total_ms: list[float] = []
    failures: list[str] = []

    for question, figures, keywords in QUESTIONS:
        body, elapsed = ask(question)
        total_ms.append(elapsed)
        if body is None:
            failures.append(f"no response: {question[:40]}")
            continue
        diag = body.get("diagnostics") or {}
        if diag.get("retrieval_ms"):
            retrieval_ms.append(diag["retrieval_ms"])
        if diag.get("refused"):
            failures.append(f"{diag.get('guardrail')}: {question[:40]}")
            continue
        answered += 1

        answer = body.get("answer") or ""
        if figures:
            scored += 1
            if any(f.lower() in answer.lower() for f in figures):
                correct += 1
            else:
                failures.append(f"wrong figure: {question[:40]}")

        for s in body.get("useful_resources") or []:
            total_sources += 1
            title = (s.get("title") or "").lower()
            if any(kw in title for kw in keywords):
                relevant += 1

    return {
        "strategy": name,
        "answered": answered,
        "of": len(QUESTIONS),
        "correct": correct,
        "scored": scored,
        "precision": (relevant / total_sources) if total_sources else 0.0,
        "retrieval_ms": (sum(retrieval_ms) / len(retrieval_ms)) if retrieval_ms else 0.0,
        "total_ms": (sum(total_ms) / len(total_ms)) if total_ms else 0.0,
        "failures": failures,
    }


def main() -> None:
    original = next(
        (ln.split("=", 1)[1] for ln in read_env() if ln.startswith("RERANK_STRATEGY")),
        "fusion",
    )
    results = []
    try:
        for name in STRATEGIES:
            print(f"\n--- {name} " + "-" * 44)
            set_strategy(name)
            restart()
            if not wait_ready():
                print("  api did not come up, skipping")
                continue
            if not confirm(name):
                print(f"  WARNING container does not report RERANK_STRATEGY={name}")
            row = run_one(name)
            results.append(row)
            print(f"  answered {row['answered']}/{row['of']}"
                  f"  correct {row['correct']}/{row['scored']}"
                  f"  precision {row['precision']:.0%}"
                  f"  retrieval {row['retrieval_ms']:.0f}ms"
                  f"  total {row['total_ms']:.0f}ms")
    finally:
        set_strategy(original)
        restart()
        print(f"\nrestored RERANK_STRATEGY={original}")

    print("\n" + "=" * 70)
    print(f"{'strategy':>9}  {'answered':>9}  {'correct':>8}  {'precision':>10}"
          f"  {'retrieval':>10}  {'total':>8}")
    for r in results:
        print(f"{r['strategy']:>9}  {r['answered']:>4}/{r['of']:<4}  "
              f"{r['correct']:>3}/{r['scored']:<4}  {r['precision']:>9.0%}"
              f"  {r['retrieval_ms']:>8.0f}ms  {r['total_ms']:>6.0f}ms")

    for r in results:
        if r["failures"]:
            print(f"\n{r['strategy']} - what went wrong:")
            for f in r["failures"]:
                print(f"  {f}")

    with open("sweep_rerank.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("\n-> sweep_rerank.json")


if __name__ == "__main__":
    main()
