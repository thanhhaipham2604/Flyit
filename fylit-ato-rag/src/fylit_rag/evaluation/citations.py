"""Citation evaluation for the Fylit ATO RAG application.

Run from the project directory:
    python -m fylit_rag.evaluation.citations --limit 5

Checks citation presence, evidence provenance, duplicate URLs, ATO URL
structure, citation limits and matches to the labelled document.

These checks do not prove that citations support individual answer claims.
Answers and retrieved passages are saved for manual content review.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from fylit_rag.indexing.search import SearchResult


def citations_are_valid(
    sources: list[dict],
    evidence: list[SearchResult],
) -> bool:
    """Check that every citation title–URL pair exists in the evidence.

    An empty source list passes this provenance check.
    Citation presence must therefore be assessed separately.
    """
    valid_sources = {
        (item.source_title, item.source_url)
        for item in evidence
    }

    return all(
        (source.get("title", ""), source.get("url", "")) in valid_sources
        for source in sources
    )


def is_ato_url(url: str) -> bool:
    """Check URL structure and ATO hostname, not live link availability."""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()

        return (
            parsed.scheme in {"http", "https"}
            and (host == "ato.gov.au" or host.endswith(".ato.gov.au"))
            and parsed.username is None
            and parsed.password is None
        )
    except (TypeError, ValueError):
        return False


def ratio(numerator: int, denominator: int) -> float | None:
    """Return None when a rate cannot be measured."""
    return numerator / denominator if denominator else None


def evaluate_question(item: dict) -> dict:
    """Run the real application and capture its generation evidence.

    The wrapper records generation inputs and outputs while calling the
    original generation function. No synthetic answer is substituted.
    """
    from fylit_rag.api import service
    from fylit_rag.api.schemas import AskRequest

    captured = {}
    original_generate = service.generate_answer

    def capture_generation(question, evidence, *args, **kwargs):
        evidence = list(evidence)
        cap = kwargs.get("max_evidence") or service.EVIDENCE

        captured["evidence"] = evidence[:cap]
        result = original_generate(question, evidence, *args, **kwargs)
        captured["raw_result"] = result

        return result

    request = {"question": item["question"]}

    if item.get("financial_year"):
        request["financial_year"] = item["financial_year"]

    # No session ID: each question is assessed independently.
    # Sequential execution is required because this wraps a module function.
    with patch.object(service, "generate_answer", capture_generation):
        response = service.answer_question(AskRequest(**request))

    evidence_objects = [
        getattr(candidate, "result", candidate)
        for candidate in captured.get("evidence", [])
    ]

    evidence = [
        {
            "chunk_id": result.chunk_id,
            "doc_id": result.doc_id,
            "title": result.source_title,
            "url": result.source_url or "",
            "category": getattr(result, "category", None),
            "text": result.text,
        }
        for result in evidence_objects
    ]

    sources = [
        {"title": resource.title, "url": str(resource.url)}
        for resource in response.useful_resources
    ]

    urls = [source["url"] for source in sources]
    evidence_urls = {
        entry["url"] for entry in evidence if entry["url"]
    }

    cited_docs = {
        entry["doc_id"]
        for entry in evidence
        if entry["url"] and entry["url"] in urls
    }

    answerable = item.get("answerable", True)
    expected_doc = item.get("doc_id")
    raw = captured.get("raw_result")
    refused = response.diagnostics.refused
    reason = response.diagnostics.guardrail

    if reason in {"retrieval_failed", "generation_failed"}:
        status = "error"
    elif refused:
        status = "refused"
    else:
        status = "answered"

    labelled = answerable and bool(expected_doc)

    return {
        "question": item["question"],
        "answerable": answerable,
        "expected_doc_id": expected_doc,
        "expected_source_title": item.get("source_title"),
        "status": status,
        "guardrail": reason,
        "answer": response.answer,
        "sources": sources,
        "evidence": evidence,
        "source_count": len(urls),
        "sources_present": bool(sources),
        "citation_pairs_valid": citations_are_valid(
            sources, evidence_objects
        ),
        "unique_urls": len(urls) == len(set(urls)),
        "ato_url_count": sum(is_ato_url(url) for url in urls),
        "evidence_url_count": sum(
            url in evidence_urls for url in urls
        ),
        "expected_doc_in_evidence": (
            any(entry["doc_id"] == expected_doc for entry in evidence)
            if labelled else None
        ),
        "expected_doc_cited": (
            expected_doc in cited_docs if labelled else None
        ),
        "raw_answer_missing_sources": (
            not raw.get("refused", False) and not raw.get("sources")
            if raw is not None else None
        ),
        "claim_support_review": "pending",
    }


def summarise(rows: list[dict], citation_limit: int) -> dict:
    """Summarise citations on answered questions; report refusals separately."""
    answered = [
        row for row in rows if row["status"] == "answered"
    ]
    labelled = [
        row for row in answered
        if row["expected_doc_cited"] is not None
    ]
    available = [
        row for row in labelled if row["expected_doc_in_evidence"]
    ]

    citation_count = sum(row["source_count"] for row in answered)

    return {
        "questions": len(rows),
        "answered": len(answered),
        "refused": sum(row["status"] == "refused" for row in rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "guardrail_counts": dict(Counter(
            row["guardrail"]
            for row in rows
            if row.get("guardrail")
        )),
        "citation_count": citation_count,
        "citation_presence_rate": ratio(
            sum(row["sources_present"] for row in answered),
            len(answered),
        ),
        "citation_pair_validity_rate": ratio(
            sum(row["citation_pairs_valid"] for row in answered),
            len(answered),
        ),
        "evidence_url_membership_rate": ratio(
            sum(row["evidence_url_count"] for row in answered),
            citation_count,
        ),
        "ato_url_structure_rate": ratio(
            sum(row["ato_url_count"] for row in answered),
            citation_count,
        ),
        "unique_urls_answer_rate": ratio(
            sum(row["unique_urls"] for row in answered),
            len(answered),
        ),
        "citation_limit_compliance_rate": ratio(
            sum(
                row["source_count"] <= citation_limit
                for row in answered
            ),
            len(answered),
        ),
        "labelled_answer_count": len(labelled),
        "expected_document_citation_rate": ratio(
            sum(row["expected_doc_cited"] for row in labelled),
            len(labelled),
        ),
        "expected_doc_available_count": len(available),
        "expected_document_selection_rate_when_available": ratio(
            sum(row["expected_doc_cited"] for row in available),
            len(available),
        ),
        "raw_answers_missing_sources": sum(
            row.get("raw_answer_missing_sources") is True
            for row in rows
        ),
        "limitations": [
            "Synthetic questions may overestimate performance.",
            "The labelled document is not an exhaustive relevance judgement.",
            "Document matching uses cited URLs mapped to captured evidence.",
            "Evidence membership does not prove answer claim support.",
            "ATO URL structure checks do not verify live link availability.",
            "Final citation presence is enforced by the output guard.",
            "Manual claim-support review is pending.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate citations using the real Fylit service."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/eval/questions.jsonl"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of questions to run; 0 means all.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval/results/citations"),
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    if not args.questions.is_file():
        parser.error(f"Question file not found: {args.questions}")

    items = []

    with args.questions.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                parser.error(f"Invalid JSON on line {line_number}")

            if (
                not isinstance(item, dict)
                or not isinstance(item.get("question"), str)
                or not item["question"].strip()
            ):
                parser.error(f"Invalid question on line {line_number}")

            if (
                "answerable" in item
                and not isinstance(item["answerable"], bool)
            ):
                parser.error(
                    f"answerable must be boolean on line {line_number}"
                )

            items.append(item)

    if args.limit:
        items = items[:args.limit]

    if not items:
        parser.error("The evaluation set is empty")

    from fylit_rag.api import service
    from fylit_rag.config import settings

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    details_path = args.out / f"{stamp}-details.jsonl"
    summary_path = args.out / f"{stamp}-summary.json"
    rows = []

    print(
        f"Evaluating {len(items)} questions "
        f"with reranker={settings.rerank_strategy}",
        flush=True,
    )

    with details_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(items, start=1):
            print(
                f"[{index}/{len(items)}] {item['question']}",
                flush=True,
            )

            try:
                row = evaluate_question(item)
            except Exception as exc:
                row = {
                    "question": item["question"],
                    "status": "error",
                    "error_type": type(exc).__name__,
                }

            rows.append(row)
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
            handle.flush()
            print(f"  {row['status']}", flush=True)

    summary = summarise(rows, settings.citation_limit)
    summary.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_file": str(args.questions),
        "rerank_strategy": settings.rerank_strategy,
        "shortlist": service.SHORTLIST,
        "evidence_count": service.EVIDENCE,
        "citation_limit": settings.citation_limit,
        "generation_model": settings.generation_model,
    })

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nDetails: {details_path}")
    print(f"Summary: {summary_path}")

    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()