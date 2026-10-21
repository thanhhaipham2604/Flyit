"""Second probe: does the answer contain the right figure?

The first probe scores answered-vs-refused. This one checks the answer
actually contains the value it should, so a confidently wrong answer no
longer counts as a pass.

Run from the repo root with the stack up:

    python probe_correctness.py

`expect` is a list of strings; ANY match counts as correct, because the
model may phrase a figure several ways ("$18,200", "18,200").

Verdicts:
  OK       answered and contained an expected value
  WRONG    answered but contained none of them  <- the interesting one
  REFUSED  declined to answer
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

API = "http://localhost:8000/ask"
TIMEOUT = 120

# (category, question, [accepted values], note)
QUESTIONS: list[tuple[str, str, list[str], str]] = [
    # --- stable figures, should be unambiguous -------------------------
    ("core", "What is the tax-free threshold for Australian residents?",
     ["18,200"], "nil band"),
    ("core", "What is the Medicare levy rate?",
     ["2%", "2 per cent"], "flat 2%"),
    ("core", "What is the top marginal tax rate for Australian residents?",
     ["45"], "top bracket"),
    ("core", "What is the CGT discount for individuals holding an asset over 12 months?",
     ["50%", "50 per cent"], "CGT discount"),
    ("core", "What is the rate of GST in Australia?",
     ["10%", "10 per cent"], "GST"),
    ("core", "When is the tax return deadline for self-lodgers?",
     ["31 October", "31st October"], "deadline"),

    # --- current-year figures ------------------------------------------
    ("current", "What is the income tax rate for someone earning $50,000 as a resident?",
     ["4,288", "30 cent", "30c"], "2024-25/2025-26 second bracket"),
    ("current", "What is the minimum repayment income for study loans this year?",
     ["67,000"], "2025-26 HELP threshold"),
    ("current", "What is the superannuation guarantee rate?",
     ["12%", "11.5%", "12 per cent"], "SG rate, year-dependent"),

    # --- explicitly year-scoped: the year bug should show up here ------
    ("year", "What were the resident income tax rates for the 2022-23 income year?",
     ["5,092", "32.5"], "must be 2022-23 figures"),
    ("year", "What were the resident income tax rates for the 2024-25 income year?",
     ["4,288", "16 cent", "16c"], "must be 2024-25 figures"),
    ("year", "What was the study loan repayment threshold for 2024-25?",
     ["54,435"], "must be 2024-25, not 67,000"),
    ("year", "What is the tax-free threshold for the 2023-24 income year?",
     ["18,200"], "unchanged across years, should be easy"),

    # --- figures that depend on the missing rates section --------------
    ("gap", "What tax rate applies to a working holiday maker earning $40,000?",
     ["15%", "15 per cent", "15c"], "WHM first bracket"),
    ("gap", "What is the tax rate for foreign residents on their first dollar of income?",
     ["30", "32.5"], "foreign resident rate"),

    # --- process answers, checkable by keyword -------------------------
    ("process", "How do I get a tax file number?",
     ["myGov", "Australia Post", "online", "apply"], "TFN application route"),
    ("process", "What records do I need to keep and for how long?",
     ["five years", "5 years"], "record retention"),
    ("process", "How long do I have to amend a tax return?",
     ["two years", "2 years", "four years", "amendment period"], "amendment window"),
    ("process", "What happens if I do not lodge my tax return on time?",
     ["penalt", "fail to lodge", "FTL"], "late lodgment"),

    # --- deduction rules with specific thresholds ----------------------
    ("deduction", "How much can I claim for work expenses without written evidence?",
     ["300"], "$300 substantiation threshold"),
    ("deduction", "What is the cents per kilometre rate for car expenses?",
     ["88", "85", "78"], "rate varies by year"),
    ("deduction", "What is the maximum number of kilometres under the cents per kilometre method?",
     ["5,000", "5000"], "km cap"),
    ("deduction", "What is the fixed rate for working from home expenses?",
     ["70 cent", "67 cent", "70c", "67c"], "fixed rate method"),
    ("deduction", "Can I claim a deduction for the cost of travelling to and from work?",
     ["cannot", "not deductible", "private", "generally"], "home-to-work travel"),
]


def ask(question: str) -> dict:
    payload = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        API, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__}


def snippet(answer: str, needles: list[str]) -> str:
    """A little context around the first match, or the answer's opening."""
    for needle in needles:
        i = answer.lower().find(needle.lower())
        if i >= 0:
            return re.sub(r"\s+", " ", answer[max(0, i - 45): i + 55]).strip()
    return re.sub(r"\s+", " ", answer[:95]).strip()


def main() -> None:
    rows = []
    for i, (cat, question, expected, note) in enumerate(QUESTIONS, 1):
        body = ask(question)
        if "error" in body:
            verdict, detail = "ERROR", body["error"]
        else:
            answer = body.get("answer") or ""
            refused = bool((body.get("diagnostics") or {}).get("refused"))
            if refused:
                verdict, detail = "REFUSED", (body.get("diagnostics") or {}).get("guardrail") or ""
            elif any(e.lower() in answer.lower() for e in expected):
                verdict, detail = "OK", snippet(answer, expected)
            else:
                verdict, detail = "WRONG", snippet(answer, expected)
        rows.append((cat, question, expected, note, verdict, detail))
        print(f"[{i:2}/{len(QUESTIONS)}] {verdict:7} {cat:9} {question[:52]}")
        if verdict in ("WRONG", "REFUSED"):
            print(f"         expected one of {expected}  ({note})")
            print(f"         got: {detail}")

    print("\n" + "=" * 64)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[4]] = counts.get(row[4], 0) + 1
    for verdict in ("OK", "WRONG", "REFUSED", "ERROR"):
        if counts.get(verdict):
            print(f"  {verdict:8} {counts[verdict]}")

    print("\nBY CATEGORY")
    cats: dict[str, list[str]] = {}
    for row in rows:
        cats.setdefault(row[0], []).append(row[4])
    for cat, verdicts in cats.items():
        ok = verdicts.count("OK")
        print(f"  {cat:9} {ok}/{len(verdicts)} correct")

    wrong = [r for r in rows if r[4] == "WRONG"]
    if wrong:
        print("\nWRONG ANSWERS - read these by hand:")
        for cat, question, expected, note, _, detail in wrong:
            print(f"\n  [{cat}] {question}")
            print(f"    expected: {expected}  ({note})")
            print(f"    got:      {detail}")

    with open("probe_correctness.json", "w", encoding="utf-8") as fh:
        json.dump(
            [
                {"category": c, "question": q, "expected": e,
                 "note": n, "verdict": v, "detail": d}
                for c, q, e, n, v, d in rows
            ],
            fh,
            indent=1,
        )
    print("\n-> probe_correctness.json")


if __name__ == "__main__":
    main()
