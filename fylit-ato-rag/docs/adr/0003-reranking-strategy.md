# ADR-0003: Reranking strategy

- **Status:** accepted
- **Date:** 2026-09-08

## Context

Hybrid retrieval returns a shortlist ordered by reciprocal rank fusion. The
brief requires us to compare at least two reranking approaches and record the
outcome, on the reasoning that a second pass can lift the strongest evidence to
the top before an answer is written.

Three strategies were implemented behind one interface in
`retrieval/rerank.py`, selected by `settings.rerank_strategy`:

- **`fusion`** — keep the RRF order. The baseline: free, deterministic, no
  query-path dependency.
- **`mmr`** — maximal marginal relevance (Carbonell & Goldstein, 1998), λ = 0.7,
  using embeddings already in the index. Free and local. Motivated by real
  redundancy in this corpus: the same guidance is restated across myTax
  2021/2022/2023 pages, so a shortlist can spend every slot on near-identical
  text.
- **`llm`** — `gpt-4o-mini` scores the top 20 candidates against the question in
  a single call. The only strategy that reads the question.

A cross-encoder (e.g. `ms-marco-MiniLM`) was considered and not implemented: it
would add torch and transformers, roughly 2 GB, to a service whose dependency
list is otherwise eleven light packages.

## Evaluation

`scripts/evaluate.py` over 120 questions from `data/eval/questions.jsonl`,
shortlist of 30, top-5 handed to the answer.

The question set is **synthetic** — `scripts/build_eval_set.py` samples a chunk
and asks an LLM to write a question it answers, so the correct chunk is known by
construction. Questions written from a chunk share its vocabulary, so absolute
recall is optimistic. The set is used to *compare* configurations, where that
bias applies to every row equally.

| strategy | chunk recall@5 | doc recall@5 | MRR | rerank latency |
|---|---|---|---|---|
| `llm` | 0.592 | 0.658 | 0.411 | 2180 ms |
| `fusion` | 0.550 | 0.625 | 0.376 | 0 ms |
| `mmr` | 0.483 | 0.617 | 0.362 | 40 ms |

Doc recall is reported alongside chunk recall because this corpus repeats
itself: an equally correct chunk from a sibling page counts as a miss under
chunk recall, so the strict number understates real quality.

**The ranking above is not statistically meaningful.** Every strategy answers
the same questions, so the comparison is paired; only questions where two
strategies disagree carry information (McNemar):

| vs `fusion` | wins | losses | disagreements | p | verdict |
|---|---|---|---|---|---|
| `llm` | 13 | 9 | 22 | 0.52 | not significant |
| `mmr` | 8 | 9 | 17 | 1.00 | not significant |

At 120 questions the LLM reranker's four-point edge in doc recall is noise. It
would take several hundred more questions to detect a difference this size, and
they would have to be better questions than synthetic ones to be worth acting on.

## Decision

**Default to `fusion`.** Ship reciprocal-rank-fusion order unreranked.

The evidence does not show that either reranker improves retrieval, and `llm`
costs about **2.2 seconds and an API call on every query** — against a brief that
lists responsiveness under concurrent load as a quality target. Paying that for
an effect we cannot measure is the wrong trade.

All three strategies stay in the codebase behind
`settings.rerank_strategy`, so the decision is configuration rather than code and
can be revisited the moment there is better evidence.

## Consequences

- The query path has no reranking step: no added latency, no per-query LLM cost,
  and one less external dependency between a question and an answer.
- `mmr` is available for the redundancy problem it was written for. It does not
  help on this question set, where each question has exactly one right answer and
  diversity can only cost relevance. It may still help a user-facing shortlist,
  where five near-identical passages read badly even when each is correct — that
  is a different objective from recall@5 and this evaluation does not measure it.
- `llm` is available and one setting away if a better question set later shows a
  real gain.
- **The honest limitation:** we have not shown reranking is useless, only that we
  cannot detect a benefit with 120 synthetic questions. The next step, when it
  matters, is a hand-written question set drawn from real taxpayer phrasing —
  which is also the only way to measure the case reranking should help most: a
  question whose wording shares little vocabulary with the answer.

## Alternatives considered

- **Cross-encoder reranker.** Usually the strongest option and no per-query API
  cost, but ~2 GB of dependencies for a benefit this evaluation could not
  establish for a much cheaper reranker. First thing to revisit if reranking
  quality becomes the bottleneck.
- **Defaulting to `llm` anyway**, on the grounds that it topped the table.
  Rejected: that is choosing a 2.2-second-per-query cost on a p-value of 0.52.
