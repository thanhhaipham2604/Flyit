"""Grounding checks: every factual claim must trace to a retrieved passage.

Also decides confidence: if top evidence is too weak/thin, trigger the
refusal path BEFORE generation (the red branch in the architecture diagram).
"""


def has_sufficient_evidence(query: str, evidence) -> bool:
    """TODO: score-threshold + coverage heuristic; tune against eval set."""
    raise NotImplementedError


def verify_grounding(answer: str, evidence) -> bool:
    """TODO: post-hoc check that claims map to passages (optional stretch)."""
    raise NotImplementedError
