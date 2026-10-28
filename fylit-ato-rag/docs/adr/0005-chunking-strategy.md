# ADR-0005: Chunking strategy

- **Status:** accepted
- **Date:** 2026-10-28

## Context

A chunk is the unit of **both** search and citation. Retrieval ranks chunks, the model
reads chunks, and "Useful Resources" names the page a chunk came from. A boundary in the
wrong place therefore damages two things at once: the evidence the answer is written
from, and the reader's ability to check it.

The ATO corpus is unusually structured for a web scrape. Measured over all 5,588
documents:

- **20.0%** of documents contain at least one Markdown table (3,882 table blocks in all),
  and tables are where the rates, thresholds and caps live - the facts most often asked
  for and the most damaging to get wrong.
- Pages are organised by headings, and the heading usually carries the qualifying
  context ("Expenses you can claim", "Records you need to keep"). Text lifted away from
  its heading loses the condition attached to it.
- Lists are everywhere, and ATO writes them with blank lines between items, which a
  naive paragraph splitter reads as separate paragraphs.

The question this ADR settles: where does a chunk end?

## Decision

**Structure-aware, heading-bounded chunking**, implemented in `ingestion/chunker.py`
with three rules:

1. **A chunk never spans a heading.** Every chunk belongs to exactly one heading path,
   so a citation names one section rather than straddling two. The section's heading is
   repeated at the top of every chunk in that section, because both halves of retrieval
   read the chunk text and a chunk that does not say where it came from is harder to
   rank and to cite.
2. **Structures stay whole.** A table or list is emitted intact even when it exceeds the
   size target. Half a rate table is worse than a large chunk: it is unusable as
   evidence and misleading as a citation.
3. **Within a section, blocks are packed** up to `TARGET_CHARS`, overflowing into
   further chunks that keep the same heading path. Only prose longer than `MAX_CHARS` is
   split, and only on sentence boundaries.

Thresholds, in characters: `TARGET_CHARS = 1200`, `MAX_CHARS = 2000`, `MIN_CHARS = 200`.

**Characters, not tokens.** The module is standard-library only, so the boundary rules
can be tested without a tokeniser, a model or a network. For English prose the
character-to-token ratio is stable enough that a character budget lands in the intended
token range. The cost is that the budget is approximate for non-English text, which this
corpus does not contain.

Two supporting decisions:

- **A section shorter than `MIN_CHARS` is folded into the next chunk**, but only when
  the two are siblings or parent-and-child. Merging across unrelated topics would be
  worse than a short chunk. The merged chunk is labelled with the deepest heading path
  common to both, so the citation names a section that genuinely contains all of it.
- **`chunk_id` is `doc_id#ordinal`** and each chunk carries a hash of its own text. That
  is what makes re-indexing incremental at chunk granularity: editing one paragraph of a
  long page re-embeds one chunk, not the page (ADR-0004).

## Measured outcome

Over the full corpus, chunked offline on 2026-10-28:

| Measure | Value |
|---|---|
| Documents / chunks | 5,588 → **53,334** |
| Chunks per document | mean 9.5, median 6, max 265 |
| Chunk size | p50 **807**, p90 **1,231**, p99 3,805 chars |
| Over `MAX_CHARS` | 1,767 (3.31%) - 680 tables, 1,078 lists |
| Under `MIN_CHARS` after merging | 1,073 (2.01%) |
| Chunks containing a table | 3,349 (6.3%) |
| Chunks with no heading path | 1 (0.002%) |
| Question-style headings | 4,081 (7.7%) |

The distribution sits where it was aimed: half the chunks are under 807 characters and
nine in ten under 1,231, while the 3.31% that exceed the ceiling are almost entirely the
tables and lists rule 2 protects deliberately.

The 7.7% with question-style headings ("What if I…?") motivated `_embedding_text`: for
an FAQ-shaped section the **answer body** is embedded rather than the question, because
embedding the question makes the chunk similar to every other question on the page.

## Consequences

**Gained.** Rate tables survive intact and remain citable. Citations name one real
section. Chunk-level hashing keeps re-indexing cheap. The rules are deterministic and
testable without infrastructure - `tests/test_chunker.py` pins each boundary case.

**Given up.** Chunk sizes are uneven, so a shortlist of five chunks is not a fixed
budget of context. Around 2% of chunks stay below the minimum because they have no
eligible neighbour to merge with.

**Known risk, measured here: the longest chunk cannot be embedded.** The largest chunk
in the corpus is **103,521 characters** - a single table from "PLS form errors" - and 57
chunks exceed 8,000 characters. `text-embedding-3-small` accepts 8,191 tokens, roughly
32,000 characters of English. One chunk is over that limit today.

`indexing/embeddings.py` neither truncates its input nor treats an over-long input as
recoverable: the API's `BadRequestError` is not in `TRANSIENT_ERRORS`, so it propagates
and stops the embedding run. The live index has not hit this because it was built on
15 September, before the fix that stopped tables being split (commit 670dfb1). **The
next full re-index will hit it.**

The fix belongs at embed time rather than in the chunker: cap the text sent to the
embedding API, while leaving `text` whole so the citation and the evidence are
unaffected. A chunk that large is a reference table whose first 32,000 characters are
ample for a similarity match. Splitting it instead would reopen rule 2 for the one case
rule 2 exists to protect.

## Alternatives considered

- **Fixed-size windows with overlap.** The usual default, and measured against this
  corpus: 1,200-character windows produce 41,254 chunks and **split 54.4% of all 3,882
  tables** (2,112 of them). More than half the rate tables would arrive at the model cut
  in half, with their headers in a different chunk from their numbers. Rejected on that
  number alone.
- **One chunk per page.** Simple and perfect for citation, hopeless for retrieval:
  pages run to 17,000+ characters covering several topics, so a question about one
  paragraph retrieves everything else with it.
- **Semantic (embedding-based) splitting.** Boundaries chosen by similarity between
  adjacent sentences. It costs an embedding pass over the whole corpus at ingest time,
  produces non-deterministic boundaries that move between runs, and has no obvious
  advantage on a corpus whose authors already marked the boundaries with headings.
- **Token-based sizing with `tiktoken`.** More precise against the model's real limit,
  at the cost of a dependency and a model-specific coupling in a module that currently
  needs neither. Worth revisiting if the corpus ever stops being English prose, or if
  the embedding cap above becomes a recurring problem rather than a single chunk.
- **Parent-child / hierarchical retrieval**, where small chunks are retrieved and their
  parent section is sent to the model. Genuinely attractive for this corpus, and more
  machinery than is justified while the measured ceiling is doc recall around 0.61
  (ADR-0003). Revisit when reranking is settled.

## Follow-ups

1. Cap the embedding input (see the risk above) before the next full re-index.
2. `TARGET_CHARS` has never been swept against retrieval quality - 1,200 is a reasoned
   starting value, not a measured optimum. `scripts/probe/sweep_k.py` is the pattern to
   copy, and the honest statement today is that chunk size is unmeasured.
3. The live index holds 55,062 rows against the 53,334 this measurement produces,
   because it predates both the table fix and `embedding_text`. A re-index will
   reconcile the two.
