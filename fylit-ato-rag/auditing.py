"""Compatibility CLI for the Fylit ATO corpus audit.

The reusable audit implementation now lives under:

    src/fylit_rag/ingestion/audit.py

This root-level script is intentionally small, so existing development commands
such as:

    python auditing.py --input data/ato_corpus --output data/processed/audit

keep working, while the logic itself stays importable and testable inside the
package - matching datacleaning.py and extract_year.py.
"""

from fylit_rag.ingestion.audit import main

if __name__ == "__main__":
    raise SystemExit(main())
