"""Request/response models - the structured contract the guide specifies.

The API always returns: Answer (or controlled refusal), Useful Resources
(source titles + URLs when evidence was used), Diagnostics (timing +
retrieval details, for developers only - never shown to customers).
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: str | None = None
    financial_year: str | None = None  # e.g. "2024-25"


class Resource(BaseModel):
    title: str
    url: str


class Diagnostics(BaseModel):
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    chunks_considered: int | None = None
    refused: bool = False


class AskResponse(BaseModel):
    answer: str
    useful_resources: list[Resource] = []
    disclaimer: str
    diagnostics: Diagnostics
