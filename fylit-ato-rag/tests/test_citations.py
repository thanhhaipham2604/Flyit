from fylit_rag.evaluation.citations import citations_are_valid
from fylit_rag.indexing.search import SearchResult


def make_result(title: str, url: str) -> SearchResult:
    return SearchResult(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="ATO evidence",
        score=0.9,
        source_title=title,
        source_url=url,
    )


def test_valid_citation_returns_true():
    evidence = [
        make_result(
            "ATO source",
            "https://www.ato.gov.au/example",
        )
    ]

    sources = [
        {
            "title": "ATO source",
            "url": "https://www.ato.gov.au/example",
        }
    ]

    assert citations_are_valid(sources, evidence) is True


def test_invalid_citation_returns_false():
    evidence = [
        make_result(
            "ATO source",
            "https://www.ato.gov.au/example",
        )
    ]

    sources = [
        {
            "title": "Unknown source",
            "url": "https://example.com",
        }
    ]

    assert citations_are_valid(sources, evidence) is False


def test_multiple_valid_citations_return_true():
    evidence = [
        make_result("ATO source 1", "https://www.ato.gov.au/one"),
        make_result("ATO source 2", "https://www.ato.gov.au/two"),
    ]

    sources = [
        {
            "title": "ATO source 1",
            "url": "https://www.ato.gov.au/one",
        },
        {
            "title": "ATO source 2",
            "url": "https://www.ato.gov.au/two",
        },
    ]

    assert citations_are_valid(sources, evidence) is True