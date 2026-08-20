"""LLM client wrapper (OpenAI chat completions).

Produces either a grounded answer or a controlled refusal - never
free-form knowledge. Timeouts and retries live here.
"""


def generate_answer(question: str, evidence, history=None) -> dict:
    """TODO: call OpenAI with prompts.SYSTEM_PROMPT + evidence; return
    {answer, refused: bool, sources: [{title, url}]}."""
    raise NotImplementedError
