"""FastAPI query endpoints.

HTTP-specific concerns such as routing, request handling, response models, and
rate limiting live here. RAG pipeline orchestration is delegated to the
application service layer.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from fylit_rag.api.rate_limit import ask_limit, limiter
from fylit_rag.api.schemas import AskRequest, AskResponse
from fylit_rag.api.service import answer_question


router = APIRouter()


@router.post("/ask", response_model=AskResponse)
@limiter.limit(ask_limit)
def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer a question using the Fylit RAG application.

    The route handles the HTTP boundary while the service layer performs the
    application and RAG orchestration.
    """
    return answer_question(body)