#!/usr/bin/env bash
# Seeds two pages the scraper does not collect.
#
# The scraped corpus holds no resident income tax rate table: "18,201"
# appears in none of its 55,062 chunks, and the only "for each $1 over"
# rate table in it is for deceased estates. Without these rows a clean
# checkout cannot answer anything needing the bracket grid - what the
# brackets are, the marginal rate on an income, the tax on an amount -
# so the retrieval and guardrail fixes look broken when they are not.
#
# Both pages are on ato.gov.au and are reproduced verbatim. Delete this
# once the scraper covers /tax-rates-and-codes/.
#
# Run from the repository root, after the corpus is ingested:
#   ./scripts/seed/load.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

COLS="chunk_id, doc_id, chunk_ordinal, content_hash, text, heading_path, source_title, source_url, category, topic, last_updated, financial_year, version, status, superseded_by, indexed_at, embedding_text, embedding"

DOCS="'7fe454244d2739708b3484ddecf4937fe44afb38', 'af4ae731a8e5403baebd1c058b583f9d5ccd24fb'"

echo "Removing any existing copies of the seeded documents..."
docker compose exec -T postgres psql -U fylit -d fylit -v ON_ERROR_STOP=1 -c "DELETE FROM chunks WHERE doc_id IN ($DOCS);"

echo "Loading scripts/seed/rate_pages.csv..."
docker compose exec -T postgres psql -U fylit -d fylit -v ON_ERROR_STOP=1 -c "COPY chunks ($COLS) FROM STDIN WITH (FORMAT csv, HEADER true)" < scripts/seed/rate_pages.csv

echo "Verifying..."
docker compose exec -T postgres psql -U fylit -d fylit -c "SELECT source_title, count(*) AS chunks FROM chunks WHERE doc_id IN ($DOCS) GROUP BY 1 ORDER BY 1;"
