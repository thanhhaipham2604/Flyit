"""Build focused ATO-friendly retrieval queries from taxpayer questions.

This module improves retrieval only. It must never answer the tax question,
determine entitlement, or add factual claims that are not present in the
retrieved ATO evidence.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Retrieval terminology normalisation
# ---------------------------------------------------------------------------

# Normalise genuinely ambiguous user terminology into clearer retrieval
# terminology.
#
# These replacements affect retrieval only. The original user question is
# still passed unchanged to answer generation.
#
# Example:
#   Uber Eats -> food delivery / providing services
#
# This prevents the word "Uber" inside "Uber Eats" from pulling retrieval
# toward ATO ride-sourcing guidance.
_NORMALISATIONS = (
    (
        re.compile(
            r"\buber eats\b",
            re.I,
        ),
        "food delivery providing services through a digital platform",
    ),
    (
        re.compile(
            r"\bdoordash\b",
            re.I,
        ),
        "food delivery providing services through a digital platform",
    ),
)


# ---------------------------------------------------------------------------
# Concept vocabulary bridges
# ---------------------------------------------------------------------------

# Vocabulary bridges only. Keep these deliberately short so that useful terms
# from the taxpayer's question are not overwhelmed by generic tax terminology.
_CONCEPTS = (
    (
        re.compile(
            r"\b(laptop|computer|pc|equipment|tool|tools)\b",
            re.I,
        ),
        "work expenses decline in value",
    ),
    (
        re.compile(
            r"\b(work from home|working from home|wfh|home office)\b",
            re.I,
        ),
        "working from home expenses",
    ),
    (
        re.compile(
            r"\b(receipt|receipts|written evidence|proof of purchase)\b",
            re.I,
        ),
        "record keeping written evidence exceptions",
    ),
    (
        re.compile(
            r"\b(share|shares|stock|stocks)\b",
            re.I,
        ),
        "capital gains tax CGT",
    ),
    (
        re.compile(
            r"\b(crypto|cryptocurrency|bitcoin|ethereum)\b",
            re.I,
        ),
        "crypto assets CGT",
    ),
    (
        re.compile(
            r"\b(uber eats|doordash|food delivery|delivery driver|deliveries)\b",
            re.I,
        ),
        "sharing economy providing services delivering food gig economy income",
    ),
    (
        re.compile(
            r"\b(rideshare|ride share|ride-sharing|ride-sourcing)\b"
            r"|\buber(?!\s+eats\b)\b",
            re.I,
        ),
        "sharing economy ride-sourcing income",
    ),
    (
        re.compile(
            r"\b(replace|replaced|replacing|replacement)\b",
            re.I,
        ),
        "replacement",
    ),
    (
        re.compile(
            r"\b(rental|tenant|tenants|investment property)\b",
            re.I,
        ),
        "rental property",
    ),
    (
        re.compile(
            r"\b(super|superannuation)\b",
            re.I,
        ),
        "superannuation contributions",
    ),
        (
        re.compile(
            r"\b(rental|tenant|tenants|investment property)\b",
            re.I,
        ),
        "rental property",
    ),
    (
        re.compile(
            r"\b(spouse|partner)\b",
            re.I,
        ),
        "spouse details spouse income income tests",
    ),
    (
        re.compile(
            r"\b(super|superannuation)\b",
            re.I,
        ),
        "superannuation contributions",
    ),

    (
        re.compile(
            r"\b(tax bracket|tax brackets|tax rate|tax rates)\b",
            re.I,
        ),
        "individual income tax rates",
    ),
)


# ---------------------------------------------------------------------------
# Conversational filler
# ---------------------------------------------------------------------------

# These phrases are useful to humans but usually add little retrieval value.
# They are removed only from the retrieval query.
_FILLER_PATTERNS = (
    re.compile(r"\bcan i\b", re.I),
    re.compile(r"\bdo i need to\b", re.I),
    re.compile(r"\bdo i have to\b", re.I),
    re.compile(r"\bam i able to\b", re.I),
    re.compile(r"\bmay i\b", re.I),
    re.compile(r"\bwhat should i do\b", re.I),
    re.compile(r"\bhow do i\b", re.I),
    re.compile(r"\bhow should i\b", re.I),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_question(question: str) -> str:
    """Normalise ambiguous terminology for retrieval only."""

    normalised = question

    for pattern, replacement in _NORMALISATIONS:
        normalised = pattern.sub(
            replacement,
            normalised,
        )

    return normalised


def has_normalisation(question: str) -> bool:
    """Return True when an ambiguous term needs retrieval normalisation.

    A normalised query is intentionally allowed to replace the original
    retrieval query because retaining the ambiguous wording can pull the
    retriever toward the wrong ATO activity.

    The original question is still preserved for answer generation.
    """

    question = (question or "").strip()

    if not question:
        return False

    return any(
        pattern.search(question)
        for pattern, _ in _NORMALISATIONS
    )


def _focus_question(question: str) -> str:
    """Remove light conversational noise without removing factual details."""

    focused = question

    for pattern in _FILLER_PATTERNS:
        focused = pattern.sub(
            " ",
            focused,
        )

    # Punctuation generally adds no retrieval value.
    focused = re.sub(
        r"[?!]",
        " ",
        focused,
    )

    # Collapse whitespace created by removals.
    focused = re.sub(
        r"\s+",
        " ",
        focused,
    ).strip()

    return focused


# ---------------------------------------------------------------------------
# Compact lexical query
# ---------------------------------------------------------------------------


def compact_query(question: str) -> str:
    """Return a short keyword-oriented query for lexical retrieval.

    This removes common conversational words while preserving domain nouns,
    amounts, dates and action words from the taxpayer's original question.

    It does not add tax rules or infer an answer.
    """

    question = (question or "").strip()

    if not question:
        return question

    # Apply the same terminology normalisation used by expand_query so
    # ambiguous platform names do not reintroduce the wrong activity.
    question = _normalise_question(question)

    stopwords = {
        "i",
        "my",
        "me",
        "we",
        "our",
        "you",
        "your",
        "a",
        "an",
        "the",
        "this",
        "that",
        "it",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "would",
        "should",
        "may",
        "might",
        "to",
        "of",
        "for",
        "and",
        "or",
        "in",
        "on",
        "at",
        "from",
        "with",
        "as",
        "if",
        "but",
        "what",
        "how",
        "claim",
        "full",
        "cost",
        "amount",
        "immediately",
        "need",
        "use",
    }

    tokens = re.findall(
        r"\$[\d,]+(?:\.\d+)?|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*",
        question,
    )

    kept = [
        token
        for token in tokens
        if token.lower() not in stopwords
    ]

    normalisations = {
        "replaced": "replacement",
        "replacing": "replacement",
        "replace": "replacement",
        "bought": "purchase",
        "buy": "purchase",
        "sold": "sale",
        "selling": "sale",
        "sell": "sale",
        "earned": "income",
        "earning": "income",
    }

    kept = [
        normalisations.get(
            token.lower(),
            token,
        )
        for token in kept
    ]

    return " ".join(kept)


# ---------------------------------------------------------------------------
# Main query expansion
# ---------------------------------------------------------------------------


def expand_query(question: str) -> str:
    """Return a focused ATO-friendly retrieval query.

    Concept detection uses the taxpayer's original wording.

    The retrieval query itself uses normalised wording where an ambiguous
    expression has a known ATO-aligned interpretation.

    The original question is never modified for answer generation.
    """

    question = (question or "").strip()

    if not question:
        return question

    # Build retrieval text from normalised wording.
    normalised = _normalise_question(question)

    focused = _focus_question(
        normalised,
    )

    additions: list[str] = []

    # Detect concepts against the ORIGINAL wording. This is important because
    # replacing "Uber Eats" above should not prevent the Uber Eats concept rule
    # from adding its sharing-economy vocabulary bridge.
    for pattern, expansion in _CONCEPTS:
        if pattern.search(question):
            if expansion.lower() not in focused.lower():
                additions.append(
                    expansion,
                )

    if additions:
        focused = (
            f"{focused} "
            f"{' '.join(additions)}"
        )

    return re.sub(
        r"\s+",
        " ",
        focused,
    ).strip()