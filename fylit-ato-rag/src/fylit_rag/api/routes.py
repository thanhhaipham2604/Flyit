"""Query endpoints.

POST /ask pipeline: validate -> input guards -> (memory contextualise) ->
hybrid retrieve -> rerank -> grounding confidence check -> generate or refuse
-> output guards -> respond with answer + Useful Resources + diagnostics.
"""

from fastapi import APIRouter

from fylit_rag.api.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """TODO: wire the full Zone B pipeline; apply rate limiting."""
    raise NotImplementedError
