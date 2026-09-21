from fylit_rag.evaluation.citations import citations_are_valid
from fylit_rag.retrieval.hybrid import retrieve


def test_citations_with_real_hybrid_retrieval():
    results = retrieve(
        "Can I claim work from home expenses?",
        top_k=5,
    )

    evidence = [item.result for item in results]

    valid_source = [
        {
            "title": evidence[0].source_title,
            "url": evidence[0].source_url,
        }
    ]

    fake_source = [
        {
            "title": "Fake ATO page",
            "url": "https://example.com/fake",
        }
    ]

    assert citations_are_valid(valid_source, evidence) is True
    assert citations_are_valid(fake_source, evidence) is False