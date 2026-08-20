"""Final checks before a response leaves the API.

Enforce: disclaimer present, sources attached when evidence used,
no personalised advice / refund guarantees / assumed deductions slipping
through, refusal format is the controlled one.
"""


def check_output(response: dict) -> dict:
    """TODO: validate/repair the outgoing response; fail closed to refusal."""
    raise NotImplementedError
