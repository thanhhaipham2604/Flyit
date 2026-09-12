"""Request/response models - the structured contract the guide specifies.

The API always returns: Answer (or controlled refusal), Useful Resources
(source titles + URLs when evidence was used), Diagnostics (timing +
retrieval details, for developers only - never shown to customers).

Diagnostics carries no passage text, no prompt and no chunk content - only
counts, timings and the name of whatever guardrail intervened. It is safe to log
and safe to return, but it is for developers, and a customer-facing UI should
not render it.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: str | None = None
    financial_year: str | None = Field(
        default=None,
        pattern=r"^20\d{2}-\d{2}$",
        description="Australian financial year, for example 2024-25",
    )


class Resource(BaseModel):
    title: str
    url: str


class Diagnostics(BaseModel):
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    chunks_considered: int | None = None
    refused: bool = False
    # Names what stopped or altered the answer - "insufficient_evidence",
    # "personalised_advice", "rate_limited". None when nothing intervened.
    guardrail: str | None = None
    # How many retrieved passages contained instruction-like text that had to be
    # neutralised. Non-zero is worth alerting on: it means someone is trying.
    injection_findings: int = 0


class AskResponse(BaseModel):
    answer: str
    useful_resources: list[Resource] = []
    disclaimer: str
    diagnostics: Diagnostics
