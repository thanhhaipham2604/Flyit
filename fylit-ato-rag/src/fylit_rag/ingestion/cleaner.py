"""Clean scraped ATO Markdown while preserving useful document structure.

The cleaner removes scraper and website UI artefacts while keeping the
content needed for retrieval, chunking and citation.

Preserved content includes:

- Markdown headings
- paragraphs
- ordered and unordered lists
- tables
- links
- legitimate QC identifiers appearing inside URLs
- legitimate instructional uses of "Print or Download"

Removed content includes:

- known ATO navigation boilerplate
- ATO online-services login boilerplate
- standalone "Print or Download" UI text
- standalone "Last updated ..." metadata
- standalone "Published <date>" metadata
- standalone QC scraper footer codes
- excessive blank lines

Publication and last-updated dates are returned separately so they can be
stored as document metadata instead of being lost.
"""

from __future__ import annotations

import re
from typing import Optional


# --------------------------------------------------------------------------- #
# Metadata patterns
# --------------------------------------------------------------------------- #

# Examples:
#
#   Last updated 18 February 2026
#   Last updated 18 February 2026Print or Download
#
LAST_UPDATED_RE = re.compile(
    r"Last updated\s+"
    r"(?P<date>[A-Za-z0-9 ,]+?)"
    r"(?=Print or Download|\n|$)"
)


# Publication metadata appears as:
#
#   Published 13 February 2025
#
# or:
#
#   Published 13 February 2025Print or Download
#
# Requiring the entire line to match prevents legitimate prose containing
# the word "Published" from being removed.
#
PUBLISHED_RE = re.compile(
    r"^[ \t]*Published[ \t]+"
    r"(?P<date>\d{1,2}[ \t]+[A-Za-z]+[ \t]+\d{4})"
    r"(?:[ \t]*Print or Download)?[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# QC scraper footer
# --------------------------------------------------------------------------- #

# IMPORTANT:
#
# Only remove QC identifiers when the complete line is footer-like.
#
# Examples removed:
#
#   QC103635
#   QC103635Print or Download
#   QC103635 Print or Download
#
# A broad pattern such as `QC\d+` must NOT be used because legitimate ATO
# download URLs can contain QC identifiers.
#
QC_FOOTER_RE = re.compile(
    r"^[ \t]*QC\d+"
    r"(?:[ \t]*Print or Download)?"
    r"[ \t]*$",
    re.MULTILINE,
)


# --------------------------------------------------------------------------- #
# Known website boilerplate
# --------------------------------------------------------------------------- #

BOILERPLATE_LINES = {
    "- [ATO Community](https://community.ato.gov.au/s)",
    "- [Legal Database](https://www.ato.gov.au/single-page-applications/legaldatabase)",
    "## Log in to ATO online services",
    "Access secure services, view your details and lodge online.",
    "Search",
    "Print or Download",
}


BOILERPLATE_PREFIXES = (
    "- [What's New](",
)


# --------------------------------------------------------------------------- #
# Main cleaning functions
# --------------------------------------------------------------------------- #


def clean_body(
    body: str,
) -> tuple[
    str,
    Optional[str],
    Optional[str],
]:
    """Clean one scraped ATO Markdown body.

    Parameters
    ----------
    body:
        Page body after the scraper header has been removed.

    Returns
    -------
    tuple
        ``(
            cleaned_text,
            last_updated_display,
            published_display,
        )``

    The operation is intentionally conservative. Only known scraper/UI
    artefacts are removed. Markdown structure and legitimate page content
    are otherwise preserved.
    """

    # ------------------------------------------------------------------ #
    # Capture Last updated metadata
    # ------------------------------------------------------------------ #

    last_updated: Optional[str] = None

    last_updated_match = LAST_UPDATED_RE.search(
        body
    )

    if last_updated_match:
        last_updated = (
            last_updated_match
            .group("date")
            .strip()
        )

    # Remove Last updated UI metadata from the body after capturing it.
    body = LAST_UPDATED_RE.sub(
        "",
        body,
    )

    # ------------------------------------------------------------------ #
    # Capture Published metadata
    # ------------------------------------------------------------------ #

    published: Optional[str] = None

    published_match = PUBLISHED_RE.search(
        body
    )

    if published_match:
        published = (
            published_match
            .group("date")
            .strip()
        )

    # Remove standalone publication UI metadata while retaining the date
    # separately for document metadata.
    body = PUBLISHED_RE.sub(
        "",
        body,
    )

    # ------------------------------------------------------------------ #
    # Remove standalone QC scraper footer codes
    # ------------------------------------------------------------------ #

    body = QC_FOOTER_RE.sub(
        "",
        body,
    )

    # ------------------------------------------------------------------ #
    # Remove known navigation and UI boilerplate
    # ------------------------------------------------------------------ #

    cleaned_lines: list[str] = []

    for line in body.split("\n"):
        stripped = line.strip()

        if stripped in BOILERPLATE_LINES:
            continue

        if any(
            stripped.startswith(prefix)
            for prefix in BOILERPLATE_PREFIXES
        ):
            continue

        cleaned_lines.append(
            line
        )

    cleaned = "\n".join(
        cleaned_lines
    )

    # Collapse three or more consecutive line breaks into one normal
    # Markdown blank line.
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    ).strip()

    return (
        cleaned,
        last_updated,
        published,
    )


def clean(
    text: str,
) -> str:
    """Return only the cleaned Markdown text.

    This small compatibility function satisfies the original package
    interface while ``clean_body`` exposes the captured metadata required
    by the ingestion pipeline.
    """

    cleaned_text, _, _ = clean_body(
        text
    )

    return cleaned_text


__all__ = [
    "BOILERPLATE_LINES",
    "BOILERPLATE_PREFIXES",
    "LAST_UPDATED_RE",
    "PUBLISHED_RE",
    "QC_FOOTER_RE",
    "clean",
    "clean_body",
]
