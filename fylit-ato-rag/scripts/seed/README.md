# Seed pages

Two ATO pages the scraper does not collect, needed for the retrieval and
guardrail fixes to reproduce outside a developer's own database.

## Why

The scraped corpus contains 5,588 documents and 55,062 chunks, and no
resident income tax rate table among them:

| Probe | Documents |
| --- | --- |
| `18,201`, the first bracket boundary | 0 |
| `for each $1 over`, rate-table wording | 1, for deceased estates |
| `45c` / `37c` marginal rates | 8, none a rate table |
| Weekly, fortnightly, monthly tax tables | 0 |
| Any URL under `/tax-rates-and-codes/` | 0 |

Rate *discussion* is well covered - the tax-free threshold appears in 79
documents, superannuation caps in 78, study loan repayment in 33 - but the
thresholds-and-rates grid itself is absent, so no answer can state a
bracket or compute tax on an income.

## What is here

| Page | Chunks |
| --- | --- |
| [Tax rates - Australian resident](https://www.ato.gov.au/tax-rates-and-codes/tax-rates-australian-residents) | 9 |
| [Study and training loan repayment thresholds and rates](https://www.ato.gov.au/tax-rates-and-codes/study-and-training-support-loans-rates-and-repayment-thresholds) | 4 |

Content is reproduced from those pages. `rate_pages.csv` carries the
embeddings, so loading needs no API key and gives every developer a
byte-identical index.

The resident page covers 2020-21 to 2026-27. The live page goes back to
1983-84; earlier years were not transcribed.

## Use

From the repository root, after the corpus is ingested:

    ./scripts/seed/load.sh

Re-running replaces the rows. Remove this directory once the scraper
covers `/tax-rates-and-codes/`.
