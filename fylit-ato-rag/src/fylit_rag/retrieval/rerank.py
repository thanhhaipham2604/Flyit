"""Rerank the hybrid shortlist so the strongest evidence sits on top.

Requirement: compare at least two approaches (e.g. cross-encoder vs
LLM-as-reranker vs score fusion only) and record the outcome in an ADR
plus scripts/evaluate.py numbers.
"""


def rerank(query: str, candidates, top_n: int = 5):
    """TODO: implement 2+ strategies behind one interface; make choice configurable."""
    raise NotImplementedError
