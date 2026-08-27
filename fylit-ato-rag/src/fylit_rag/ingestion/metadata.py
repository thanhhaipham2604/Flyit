"""Financial-year metadata enrichment for ATO documents.

This module contains the reusable metadata logic for Zone A ingestion.

Design principles
-----------------
- Prefer precision over coverage.
- Distinguish Australian financial years from tax/income years.
- Preserve all explicit year mentions.
- Assign a primary financial year only when evidence is sufficiently strong.
- Leave evergreen or ambiguous documents untagged rather than guessing.

Primary financial-year precedence
---------------------------------
1. Explicit financial-year range in the URL.
2. Known ATO myTax year-specific URL.
3. Explicit applicability statement in page content.
4. Explicit July-to-June financial-year window.
5. Explicit scoped tax/income-year statement.
6. Explicit financial year in the title.
7. Explicit tax/income year in the title.
8. Explicit financial year in a Markdown heading.
9. Explicit tax/income year in a Markdown heading.
10. Otherwise leave primary_financial_year as None.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Core financial-year patterns
# --------------------------------------------------------------------------- #

# Supported forms:
#
#   2024-25
#   2024_25
#   2024–25
#   2024—25
#   2024-2025
#
FY_RANGE_RE = re.compile(
    r"\b"
    r"(?P<start>20\d{2})"
    r"\s*[-_\u2013\u2014]\s*"
    r"(?P<end>(?:20)?\d{2})"
    r"\b"
)


# Reusable candidate-range expression for larger patterns.
FY_TEXT = (
    r"20\d{2}"
    r"\s*[-_\u2013\u2014]\s*"
    r"(?:20)?\d{2}"
)


# Markdown headings such as:
#
#   # Heading
#   ## Heading
#   ### Heading
#
HEADING_RE = re.compile(
    r"^#{1,6}\s+(?P<heading>.+?)\s*$",
    re.MULTILINE,
)


# --------------------------------------------------------------------------- #
# Tax / income year patterns
# --------------------------------------------------------------------------- #

# Examples:
#
#   2025 income year
#   2025 tax year
#
TAX_YEAR_AFTER_RE = re.compile(
    r"\b"
    r"(?P<year>20\d{2})"
    r"\s+"
    r"(?:income|tax)\s+year"
    r"\b",
    re.IGNORECASE,
)


# Examples:
#
#   income year 2025
#   tax year 2025
#
TAX_YEAR_BEFORE_RE = re.compile(
    r"\b"
    r"(?:income|tax)\s+year"
    r"\s+"
    r"(?P<year>20\d{2})"
    r"\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Scoped financial-year applicability
# --------------------------------------------------------------------------- #

# High-confidence examples:
#
#   applies to the 2024-25 financial year
#   applicable for 2024-25
#   effective from 2025-26
#
# Medium-confidence examples:
#
#   for the 2024-25 income year
#   in the 2024-25 financial year
#   during financial year 2024-25
#
# Bare phrases such as:
#
#   in 2024-25
#   for 2024-25
#
# are deliberately NOT considered sufficient primary-year evidence.
#
APPLICABILITY_RANGE_PATTERNS = (
    (
        re.compile(
            rf"\b"
            rf"(?:applies?|applicable|effective)"
            rf"\s+"
            rf"(?:(?:from|to|for)\s+)?"
            rf"(?:the\s+)?"
            rf"(?P<fy>{FY_TEXT})"
            rf"(?:\s+(?:income|tax|financial)\s+year)?"
            rf"\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        re.compile(
            rf"\b"
            rf"(?:applies?|applicable|effective)"
            rf"\s+"
            rf"(?:(?:from|to|for)\s+)?"
            rf"(?:the\s+)?"
            rf"(?:income|tax|financial)\s+year"
            rf"\s+"
            rf"(?P<fy>{FY_TEXT})"
            rf"\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        re.compile(
            rf"\b"
            rf"(?:for|in|during)"
            rf"\s+"
            rf"(?:the\s+)?"
            rf"(?P<fy>{FY_TEXT})"
            rf"\s+"
            rf"(?:income|tax|financial)\s+year"
            rf"\b",
            re.IGNORECASE,
        ),
        "medium",
    ),
    (
        re.compile(
            rf"\b"
            rf"(?:for|in|during)"
            rf"\s+"
            rf"(?:the\s+)?"
            rf"(?:income|tax|financial)\s+year"
            rf"\s+"
            rf"(?P<fy>{FY_TEXT})"
            rf"\b",
            re.IGNORECASE,
        ),
        "medium",
    ),
)


# --------------------------------------------------------------------------- #
# Scoped single tax/income-year applicability
# --------------------------------------------------------------------------- #

# Examples:
#
#   applies to the 2025 income year
#   effective for tax year 2025
#   for the 2025 income year
#
# A random mention of:
#
#   2025 income year
#
# is not automatically sufficient to identify the entire page's primary year.
#
TAX_APPLICABILITY_PATTERNS = (
    (
        re.compile(
            r"\b"
            r"(?:applies?|applicable|effective)"
            r"\s+"
            r"(?:(?:from|to|for)\s+)?"
            r"(?:the\s+)?"
            r"(?P<year>20\d{2})"
            r"\s+"
            r"(?:income|tax)\s+year"
            r"\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        re.compile(
            r"\b"
            r"(?:applies?|applicable|effective)"
            r"\s+"
            r"(?:(?:from|to|for)\s+)?"
            r"(?:the\s+)?"
            r"(?:income|tax)\s+year"
            r"\s+"
            r"(?P<year>20\d{2})"
            r"\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        re.compile(
            r"\b"
            r"(?:for|in|during)"
            r"\s+"
            r"(?:the\s+)?"
            r"(?P<year>20\d{2})"
            r"\s+"
            r"(?:income|tax)\s+year"
            r"\b",
            re.IGNORECASE,
        ),
        "medium",
    ),
    (
        re.compile(
            r"\b"
            r"(?:for|in|during)"
            r"\s+"
            r"(?:the\s+)?"
            r"(?:income|tax)\s+year"
            r"\s+"
            r"(?P<year>20\d{2})"
            r"\b",
            re.IGNORECASE,
        ),
        "medium",
    ),
)


# --------------------------------------------------------------------------- #
# Explicit Australian financial-year window
# --------------------------------------------------------------------------- #

# Example:
#
#   From 1 July 2024 to 30 June 2025
#
JULY_TO_JUNE_RE = re.compile(
    r"\b"
    r"from\s+1\s+July\s+"
    r"(?P<start>20\d{2})"
    r"[^.\n]{0,50}?"
    r"(?:to|until|through)"
    r"\s+30\s+June\s+"
    r"(?P<end>20\d{2})"
    r"\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Known ATO URL families
# --------------------------------------------------------------------------- #

# In this URL family the year represents the tax/income year ending in YYYY.
#
# Example:
#
#   /mytax-instructions/2018/
#
# becomes:
#
#   tax_year = 2018
#   primary_financial_year = 2017-18
#
MYTAX_URL_YEAR_RE = re.compile(
    r"/mytax-instructions/"
    r"(?P<year>20\d{2})"
    r"(?:/|$)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def normalise_financial_year(
    start: str,
    end: str,
) -> str | None:
    """Return a canonical YYYY-YY financial year.

    Valid examples:
        2024 + 25   -> 2024-25
        2024 + 2025 -> 2024-25

    Invalid/non-consecutive ranges return None.
    """

    start_year = int(start)

    if len(end) == 2:
        end_two_digits = int(end)
        century = (start_year // 100) * 100
        end_year = century + end_two_digits

        if end_year <= start_year:
            end_year += 100

    else:
        end_year = int(end)

    if end_year != start_year + 1:
        return None

    return f"{start_year}-{end_year % 100:02d}"


def normalise_range_text(
    value: str,
) -> str | None:
    """Extract and normalise a financial-year candidate from text."""

    match = FY_RANGE_RE.search(value)

    if not match:
        return None

    return normalise_financial_year(
        match.group("start"),
        match.group("end"),
    )


def tax_year_to_financial_year(
    tax_year: int,
) -> str:
    """Convert an Australian tax/income year to its financial-year range.

    Example:
        tax year 2025 -> financial year 2024-25
    """

    start_year = tax_year - 1

    return f"{start_year}-{tax_year % 100:02d}"


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #


def extract_headings(
    content: str,
) -> list[str]:
    """Extract Markdown heading text from cleaned content."""

    return [
        match.group("heading").strip()
        for match in HEADING_RE.finditer(content)
    ]


# --------------------------------------------------------------------------- #
# Mention extraction
# --------------------------------------------------------------------------- #


def collect_financial_year_mentions(
    *texts: str,
) -> list[str]:
    """Return all valid explicit financial-year ranges found in the texts.

    Mentions are informational only. A mention does not automatically become
    the document's primary financial year.
    """

    years: set[str] = set()

    for text in texts:
        if not text:
            continue

        for match in FY_RANGE_RE.finditer(text):
            financial_year = normalise_financial_year(
                match.group("start"),
                match.group("end"),
            )

            if financial_year:
                years.add(financial_year)

    return sorted(years)


def collect_tax_year_mentions(
    *texts: str,
) -> list[str]:
    """Return explicit four-digit tax/income-year labels."""

    years: set[str] = set()

    for text in texts:
        if not text:
            continue

        for pattern in (
            TAX_YEAR_AFTER_RE,
            TAX_YEAR_BEFORE_RE,
        ):
            for match in pattern.finditer(text):
                years.add(
                    match.group("year")
                )

    return sorted(years)


# --------------------------------------------------------------------------- #
# Primary-year inference
# --------------------------------------------------------------------------- #


def infer_primary_year(
    doc: dict,
) -> dict:
    """Infer year metadata for one cleaned ATO document.

    Returned fields:
        financial_year_mentions
        tax_year_mentions
        primary_financial_year
        tax_year
        fy_source
        fy_confidence
        fy_evidence
    """

    url = doc.get("source_url") or ""
    title = doc.get("title") or ""
    content = doc.get("cleaned_content") or ""

    headings = extract_headings(content)
    heading_text = "\n".join(headings)

    financial_year_mentions = collect_financial_year_mentions(
        url,
        title,
        heading_text,
        content,
    )

    tax_year_mentions = collect_tax_year_mentions(
        title,
        heading_text,
        content,
    )

    # Annotated because the literal below mixes list[str] with None, which
    # narrows the inferred value type and rejects the str values the rules
    # assign further down.
    result: dict[str, Any] = {
        "financial_year_mentions":
            financial_year_mentions,
        "tax_year_mentions":
            tax_year_mentions,
        "primary_financial_year":
            None,
        "tax_year":
            None,
        "fy_source":
            None,
        "fy_confidence":
            None,
        "fy_evidence":
            None,
    }

    # ------------------------------------------------------------------ #
    # Rule 1 — explicit FY range in URL
    # ------------------------------------------------------------------ #

    url_range = FY_RANGE_RE.search(url)

    if url_range:
        financial_year = normalise_financial_year(
            url_range.group("start"),
            url_range.group("end"),
        )

        if financial_year:
            result.update(
                {
                    "primary_financial_year":
                        financial_year,
                    "fy_source":
                        "url_financial_year_range",
                    "fy_confidence":
                        "high",
                    "fy_evidence":
                        url_range.group(0),
                }
            )

            return result

    # ------------------------------------------------------------------ #
    # Rule 2 — known myTax URL tax year
    # ------------------------------------------------------------------ #

    mytax_match = MYTAX_URL_YEAR_RE.search(url)

    if mytax_match:
        tax_year = int(
            mytax_match.group("year")
        )

        financial_year = (
            tax_year_to_financial_year(
                tax_year
            )
        )

        result.update(
            {
                "primary_financial_year":
                    financial_year,
                "tax_year":
                    str(tax_year),
                "fy_source":
                    "mytax_url_tax_year",
                "fy_confidence":
                    "high",
                "fy_evidence":
                    mytax_match
                    .group(0)
                    .strip("/"),
            }
        )

        return result

    # ------------------------------------------------------------------ #
    # Rule 3A — scoped FY applicability statement
    # ------------------------------------------------------------------ #

    for pattern, confidence in APPLICABILITY_RANGE_PATTERNS:
        match = pattern.search(content)

        if not match:
            continue

        financial_year = normalise_range_text(
            match.group("fy")
        )

        if financial_year:
            result.update(
                {
                    "primary_financial_year":
                        financial_year,
                    "fy_source":
                        "explicit_applicability",
                    "fy_confidence":
                        confidence,
                    "fy_evidence":
                        match.group(0).strip(),
                }
            )

            return result

    # ------------------------------------------------------------------ #
    # Rule 3B — explicit July-to-June window
    # ------------------------------------------------------------------ #

    july_june_match = JULY_TO_JUNE_RE.search(
        content
    )

    if july_june_match:
        financial_year = normalise_financial_year(
            july_june_match.group("start"),
            july_june_match.group("end"),
        )

        if financial_year:
            result.update(
                {
                    "primary_financial_year":
                        financial_year,
                    "fy_source":
                        "explicit_july_to_june_window",
                    "fy_confidence":
                        "high",
                    "fy_evidence":
                        july_june_match
                        .group(0)
                        .strip(),
                }
            )

            return result

    # ------------------------------------------------------------------ #
    # Rule 3C — scoped tax/income-year statement
    # ------------------------------------------------------------------ #

    for pattern, confidence in TAX_APPLICABILITY_PATTERNS:
        match = pattern.search(content)

        if not match:
            continue

        tax_year = int(
            match.group("year")
        )

        financial_year = (
            tax_year_to_financial_year(
                tax_year
            )
        )

        result.update(
            {
                "primary_financial_year":
                    financial_year,
                "tax_year":
                    str(tax_year),
                "fy_source":
                    "explicit_tax_or_income_year",
                "fy_confidence":
                    confidence,
                "fy_evidence":
                    match.group(0).strip(),
            }
        )

        return result

    # ------------------------------------------------------------------ #
    # Rule 4A — title contains explicit FY range
    # ------------------------------------------------------------------ #

    title_range = FY_RANGE_RE.search(title)

    if title_range:
        financial_year = normalise_financial_year(
            title_range.group("start"),
            title_range.group("end"),
        )

        if financial_year:
            result.update(
                {
                    "primary_financial_year":
                        financial_year,
                    "fy_source":
                        "title_financial_year",
                    "fy_confidence":
                        "high",
                    "fy_evidence":
                        title_range.group(0),
                }
            )

            return result

    # ------------------------------------------------------------------ #
    # Rule 4B — title explicitly identifies tax/income year
    # ------------------------------------------------------------------ #

    for pattern in (
        TAX_YEAR_AFTER_RE,
        TAX_YEAR_BEFORE_RE,
    ):
        match = pattern.search(title)

        if not match:
            continue

        tax_year = int(
            match.group("year")
        )

        financial_year = (
            tax_year_to_financial_year(
                tax_year
            )
        )

        result.update(
            {
                "primary_financial_year":
                    financial_year,
                "tax_year":
                    str(tax_year),
                "fy_source":
                    "title_tax_or_income_year",
                "fy_confidence":
                    "high",
                "fy_evidence":
                    match.group(0),
            }
        )

        return result

    # ------------------------------------------------------------------ #
    # Rule 5A — Markdown heading contains explicit FY range
    # ------------------------------------------------------------------ #

    for heading in headings:
        heading_range = FY_RANGE_RE.search(
            heading
        )

        if not heading_range:
            continue

        financial_year = normalise_financial_year(
            heading_range.group("start"),
            heading_range.group("end"),
        )

        if financial_year:
            result.update(
                {
                    "primary_financial_year":
                        financial_year,
                    "fy_source":
                        "heading_financial_year",
                    "fy_confidence":
                        "medium",
                    "fy_evidence":
                        heading,
                }
            )

            return result

    # ------------------------------------------------------------------ #
    # Rule 5B — heading explicitly identifies tax/income year
    # ------------------------------------------------------------------ #

    for heading in headings:
        for pattern in (
            TAX_YEAR_AFTER_RE,
            TAX_YEAR_BEFORE_RE,
        ):
            match = pattern.search(heading)

            if not match:
                continue

            tax_year = int(
                match.group("year")
            )

            financial_year = (
                tax_year_to_financial_year(
                    tax_year
                )
            )

            result.update(
                {
                    "primary_financial_year":
                        financial_year,
                    "tax_year":
                        str(tax_year),
                    "fy_source":
                        "heading_tax_or_income_year",
                    "fy_confidence":
                        "medium",
                    "fy_evidence":
                        heading,
                }
            )

            return result

    # ------------------------------------------------------------------ #
    # Rule 6 — evergreen / ambiguous document
    # ------------------------------------------------------------------ #

    return result


# --------------------------------------------------------------------------- #
# Document enrichment
# --------------------------------------------------------------------------- #


def enrich_document_year_metadata(
    doc: dict,
) -> dict:
    """Return a copy of a document with conservative year metadata attached."""

    enriched = dict(doc)

    year_metadata = infer_primary_year(
        doc
    )

    enriched.update(
        year_metadata
    )

    # NOTE: deliberately no `financial_year` alias here. The indexing schema has
    # a column of that exact name which is `text[] NOT NULL`, so a nullable
    # scalar sharing the name is a trap: the obvious mapping
    # `doc["financial_year"] -> chunks.financial_year` fails on NOT NULL, and
    # coercing it silently drops every year but the primary one. Downstream code
    # wants `financial_years` (the list) for the column, and
    # `primary_financial_year` when it genuinely needs one representative year.

    return enriched


# --------------------------------------------------------------------------- #
# Corpus enrichment
# --------------------------------------------------------------------------- #


def process_corpus(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict:
    """Enrich a cleaned JSONL corpus with financial-year metadata.

    Existing document fields are preserved unchanged.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    tagged = 0
    null_count = 0
    mention_count = 0
    null_with_mentions = 0

    source_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    financial_year_counts: Counter[str] = Counter()
    tax_year_counts: Counter[str] = Counter()

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                doc = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON on line "
                    f"{line_number} of "
                    f"{input_path}: {exc}"
                ) from exc

            enriched = enrich_document_year_metadata(
                doc
            )

            output_file.write(
                json.dumps(
                    enriched,
                    ensure_ascii=False,
                )
                + "\n"
            )

            total += 1

            financial_year = enriched.get(
                "primary_financial_year"
            )

            tax_year = enriched.get(
                "tax_year"
            )

            source = enriched.get(
                "fy_source"
            )

            confidence = enriched.get(
                "fy_confidence"
            )

            mentions = enriched.get(
                "financial_year_mentions"
            ) or []

            if financial_year:
                tagged += 1

                financial_year_counts[
                    financial_year
                ] += 1

            else:
                null_count += 1

            if tax_year:
                tax_year_counts[
                    tax_year
                ] += 1

            if source:
                source_counts[
                    source
                ] += 1

            if confidence:
                confidence_counts[
                    confidence
                ] += 1

            if mentions:
                mention_count += 1

            if (
                financial_year is None
                and mentions
            ):
                null_with_mentions += 1

    summary = {
        "input_file":
            str(input_path),

        "output_file":
            str(output_path),

        "total_documents":
            total,

        "documents_with_primary_financial_year":
            tagged,

        "documents_with_null_primary_financial_year":
            null_count,

        "coverage_pct":
            (
                round(
                    tagged / total * 100,
                    2,
                )
                if total
                else 0.0
            ),

        "documents_with_financial_year_mentions":
            mention_count,

        "null_documents_with_financial_year_mentions":
            null_with_mentions,

        "source_counts":
            dict(
                source_counts.most_common()
            ),

        "confidence_counts":
            dict(
                confidence_counts.most_common()
            ),

        "top_financial_years":
            dict(
                financial_year_counts
                .most_common(30)
            ),

        "top_tax_years":
            dict(
                tax_year_counts
                .most_common(30)
            ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


__all__ = [
    "collect_financial_year_mentions",
    "collect_tax_year_mentions",
    "enrich_document_year_metadata",
    "extract_headings",
    "infer_primary_year",
    "normalise_financial_year",
    "normalise_range_text",
    "process_corpus",
    "tax_year_to_financial_year",
]