"""Validate and screen incoming questions.

Block/deflect: off-topic requests, personalised-advice requests
("how much refund will *I* get"), unsafe content. On-topic general
questions pass through to retrieval.
"""


def check_input(question: str) -> tuple[bool, str | None]:
    """TODO: return (allowed, refusal_reason). Keep rules testable + documented."""
    raise NotImplementedError
