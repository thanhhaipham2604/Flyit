# ADR-0004: The reciprocal-rank-fusion constant

- **Status:** accepted
- **Date:** 2026-09-24

## Context

Hybrid retrieval blends the vector and keyword lists with reciprocal rank
fusion:

```
score(chunk) = sum over halves of 1 / (k + rank_in_that_half)
```

`k` decides how much rank position matters. At `k = 1` the first result scores
0.5 and the second 0.33, so one confident half can outvote agreement between
both. At `k = 60` the ranks compress and agreement dominates. The project
shipped 60, the value from Cormack et al. (2009), chosen there because it was
robust across collections *without* per-collection tuning.

A stakeholder asked for a lower value, under 20, on the reasoning that a smaller
`k` sharpens the shortlist by trusting the top of each list more.

The question is worth answering with data rather than intuition, because the
change is a one-line edit and the effect is invisible without measurement.

## Evaluation

The 120 answerable questions in `data/eval/questions.jsonl`, against the full
index (5,588 documents, 55,062 chunks). Fusion depends only on the two ranked
lists, so each question was retrieved once (90 candidates per half, as the query
path uses) and then re-fused at every candidate `k` — the comparison is exact
rather than sampled, and costs one retrieval pass.

| k | chunk recall@5 | doc recall@5 | MRR |
|---|---|---|---|
| 1 | 0.558 | 0.625 | 0.368 |
| 5 | 0.550 | 0.625 | 0.365 |
| 10 | 0.542 | 0.617 | 0.362 |
| 20 | 0.525 | 0.600 | 0.358 |
| 30 | 0.533 | 0.608 | 0.359 |
| **60 (shipped)** | **0.533** | **0.608** | **0.359** |
| 100 | 0.533 | 0.608 | 0.359 |

Every configuration answers the same questions, so the comparison is paired and
only the questions where two configurations disagree carry information
(McNemar, exact binomial):

| vs k=60 | wins | losses | disagreements | p | identical top-5 |
|---|---|---|---|---|---|
| k=1 | 3 | 1 | 4 | 0.62 | 95/120 |
| k=5 | 3 | 1 | 4 | 0.62 | 99/120 |
| k=10 | 2 | 1 | 3 | 1.00 | 99/120 |
| k=20 | 0 | 1 | 1 | 1.00 | 108/120 |
| k=30 | 0 | 0 | 0 | 1.00 | 111/120 |
| k=100 | 0 | 0 | 0 | 1.00 | 118/120 |

Two things stand out. **Nothing is significant**: the largest effect rests on
four disagreeing questions out of 120. And the table is **not monotonic** —
`k = 20` scores below both `k = 10` and `k = 30`, which no mechanism explains.
A curve with a real shape would not do that; this is sampling noise.

The specific request, a value under 20, is the region where the measured numbers
are *worst*. The mild upward drift at `k = 1` to `5` is the only hint of a
direction, it is not significant, and it points the opposite way from a cautious
change.

The question set is synthetic (see `scripts/build_eval_set.py`): each question
was written *from* the chunk it is scored against, so it shares that chunk's
vocabulary and absolute recall is optimistic. The bias applies equally to every
row, which is what makes the comparison usable — and it also means a real
difference of two points could be hidden by it.

## Decision

**Keep 60 as the default, and expose the value as `settings.rrf_k`.**

- `RRF_K` in `retrieval/hybrid.py` remains the documented shipped value.
- `reciprocal_rank_fusion` reads `settings.rrf_k` at call time, so a different
  value can be trialled through the environment or the Kubernetes ConfigMap
  without a code change or a rebuild.
- The setting is constrained to `>= 1`: at 0 or below, `1 / (k + rank)` divides
  by zero for some rank.

Making it configuration rather than a constant is the point. It lets the
stakeholder's suggestion be trialled on request, and keeps the shipped default
at the only value with evidence behind it.

## Consequences

- The knob is available, reversible and needs no deployment to change.
- A configured value applies to the whole service, including the evaluation
  commands, so a trial must set it deliberately and record what it was.
- Anyone lowering `k` should know what they are choosing: more weight on a
  single half being confident, less on the two halves agreeing. On a query where
  the keyword half matches nothing — common here, since `plainto_tsquery`
  requires every word — a low `k` gives the vector half almost total control.
- The experiment does not generalise beyond this corpus and this question set.
  It is evidence that we cannot detect a difference, not proof that none exists.

## Alternatives considered

- **Change the default to a value under 20, as asked.** Rejected: that region
  measured worst, and adopting it would mean choosing a shortlist ordering on a
  p-value of 1.00.
- **Tune `k` per query** (for example, lower when both halves return results).
  Rejected as unjustifiable complexity while the effect is undetectable.
- **Leave it a hard-coded constant.** Rejected: the question will be asked
  again, and a setting makes the next trial cost minutes instead of a release.
- **Re-measure on hand-written questions first.** Not rejected — deferred. A set
  drawn from real taxpayer phrasing is the prerequisite for ever changing the
  default, and is the same gap ADR-0003 names for reranking.
