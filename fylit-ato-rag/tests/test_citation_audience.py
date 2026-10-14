"""Regression tests for citation filtering on first-person questions."""

from types import SimpleNamespace

import pytest

from fylit_rag.generation.llm import _sources


@pytest.mark.parametrize(
    "question,title,category",
    [
        (
            "What do I need to know about changes to interest rates "
            "for my loans or financial contracts because of the IBOR reform?",
            "About IBOR reform",
            "businesses-and-organisations",
        ),
        (
            "What do I need to do to get my community charity "
            "endorsed as a deductible gift recipient?",
            "Community charity",
            "businesses-and-organisations",
        ),
        (
            "What do I need to report if I have a loan or borrowing "
            "transaction with a non-resident?",
            "IRPD debt interests (including ordinary loans and borrowings)",
            "businesses-and-organisations",
        ),
        (
            "What happens if I make a contribution and then "
            "try to submit an election late?",
            "Capital gains tax cap election amounts",
            "tax-and-super-professionals",
        ),
    ],
)
def test_first_person_wording_preserves_retrieved_guidance(
    question, title, category
):
    evidence = SimpleNamespace(
        source_title=title,
        source_url=f"https://www.ato.gov.au/{category}/example",
        category=category,
        version=1,
        last_updated=None,
    )

    sources = _sources(question, [evidence])

    assert any(
        source["title"] == evidence.source_title
        and source["url"] == evidence.source_url
        for source in sources
    )