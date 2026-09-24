"""Citation evaluation helpers.

Checks that citations returned with an answer can be traced back to the
retrieved ATO evidence.
"""

from fylit_rag.indexing.search import SearchResult


def citations_are_valid(
    sources: list[dict],
    evidence: list[SearchResult],
) -> bool:
    """Return True when every cited source exists in the retrieved evidence."""

    valid_sources = {
        (item.source_title, item.source_url)
        for item in evidence
    }

    for source in sources:
        citation = (
            source.get("title", ""),
            source.get("url", ""),
        )

        if citation not in valid_sources:
            return False

    return True