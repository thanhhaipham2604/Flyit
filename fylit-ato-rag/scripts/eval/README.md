# Evaluation probes

Measures what the API actually answers, against questions where we know what
a good answer looks like. Needs the stack running:

```bash
docker compose up -d
```

Everything is read-only — the probes ask questions over HTTP. The sweeps
additionally edit `EVIDENCE_COUNT` or `RERANK_STRATEGY` in `.env` and recreate
the api container, and restore your original value when they finish.

## Probes

```bash
python scripts/eval/probe.py realistic
python scripts/eval/probe.py realistic --repeat 3
python scripts/eval/probe.py behaviour --json results.json
```

| set | n | what it asks |
|---|---|---|
| `behaviour` | 48 | does it answer what it should, refuse what it should |
| `correctness` | 24 | does the answer contain the right figure |
| `realistic` | 70 | the same, phrased the way people actually type |

Verdicts: **OK**, **THIN** (answered but the expected figure was missing),
**FAIL**, **REVIEW** (a human decides), **ERROR**.

`THIN` is separate from `OK` on purpose. Scoring only "did it answer" once hid
an answer that returned rates from the wrong financial year — it answered, it
cited a source, and it was wrong.

## Sweeps

```bash
python scripts/eval/sweep.py k
python scripts/eval/sweep.py rerank
python scripts/eval/sweep.py k --values 5,10,15
```

Reports answered, correct, citation count, citation precision and timing at
each value.

## Reading the numbers honestly

**Generation is not deterministic.** The same set varies by about two
questions between runs. A single run of 57 and a single run of 59 are the same
result. Use `--repeat 3` for anything you intend to quote; it prints the spread
and lists which questions are unstable.

**Citation precision is approximate.** It matches keywords against source
titles, so a genuinely useful page with an unhelpful title scores as
irrelevant. It is measured identically at every setting, so the trend is sound
even though the absolute figure is not.

**The expectations in `questions.py` can be wrong.** Several `FAIL`s have
turned out to be a bad expectation rather than a bad answer. Read the failure
before believing it.

## Adding questions

One list, one shape, in `questions.py`:

```python
(category, question, expect, contains, source_keywords, note)
```

- `expect` — `"answer"`, `"refuse"`, or `"either"`
- `contains` — any one match counts, because the model phrases figures several
  ways; `None` skips the check
- `source_keywords` — used by the sweeps for citation precision; `None` skips
- `note` — why the row is here, for whoever reads the failure

Write questions the way a user would type them, typos included. The
`realistic` set exists because the first two were written like developer test
cases, and that blind spot hid an entire class of failure: every question
containing a dollar amount was being refused, including the one asked in the
client demo.
