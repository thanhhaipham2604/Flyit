"""Temporal filtering helpers for ATO retrieval.

For undated questions, exclude old year-specific guidance while retaining
evergreen and reasonably recent annual material.

If the user explicitly asks about a year, historical evidence is retained.
"""

from __future__ import annotations

import re
from datetime import date


_YEAR_PATTERN = re.compile(
    r"\b(?:20\d{2}|20\d{2}\s*[-–]\s*\d{2,4})\b"
)


def has_explicit_year(query: str) -> bool:
    """Return True when the query explicitly refers to a year."""
    return bool(_YEAR_PATTERN.search(query or ""))


def latest_financial_year(candidate) -> int | None:
    """Return the ending year of the newest financial year."""
    result = getattr(candidate, "result", candidate)
    years = getattr(result, "financial_year", None) or []

    latest = None

    for value in years:
        text = str(value).replace("–", "-")

        if "-" not in text:
            continue

        start_text, end_text = text.split("-", 1)

        try:
            start = int(start_text)
            end = int(end_text)

            if end < 100:
                end = (start // 100) * 100 + end

                if end < start:
                    end += 100

        except ValueError:
            continue

        latest = end if latest is None else max(latest, end)

    return latest


def prefer_current(query: str, candidates, *, max_age_years: int = 3):
    """Prefer current general guidance for undated questions.

    Explicit-year questions keep their original candidate ordering.

    For undated questions:
    - remove clearly stale annual guidance
    - retain evergreen/general guidance
    - prefer general guidance over annual tax-return instructions
    - keep recent annual instructions as fallback evidence
    """
    candidates = list(candidates)

    if has_explicit_year(query):
        return candidates

    cutoff = date.today().year - max_age_years

    general = []
    annual = []

    for candidate in candidates:
        result = getattr(candidate, "result", candidate)

        years = getattr(
            result,
            "financial_year",
            None,
        ) or []

        url = (
            getattr(
                result,
                "source_url",
                "",
            )
            or ""
        ).lower()

        # Remove genuinely stale annual material.
        if years:
            latest = latest_financial_year(candidate)

            if (
                latest is not None
                and latest < cutoff
            ):
                continue

        # Annual tax-return instruction pages are useful fallback
        # evidence, but should not outrank general guidance for an
        # undated/current-law question.
        is_annual_tax_return = (
            "/mytax-instructions/" in url
            or "/paper-tax-return-instructions/" in url
        )

        if is_annual_tax_return:
            annual.append(candidate)
        else:
            general.append(candidate)

    return general + annual