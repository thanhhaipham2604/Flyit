"""Retrieval + answer quality evaluation.

Tracks the guide's quality targets:
- retrieval: does the right chunk land near the top? (recall@k, MRR)
- grounding: do answer claims trace to retrieved passages?
- refusals: high refusal rate on genuinely unanswerable questions is a feature
- responsiveness: latency under concurrent load

TODO: build a small labelled question set (answerable + unanswerable),
run it after every significant change, print a comparison table.
"""

if __name__ == "__main__":
    raise NotImplementedError
