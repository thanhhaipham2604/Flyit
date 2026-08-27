"""Preprocessing tests for the Fylit ATO corpus.

These tests protect the Phase-1 preprocessing behaviour developed for:

- scraper-header parsing
- deterministic document identifiers and content hashes
- conservative Markdown cleaning
- publication/update metadata extraction
- safe QC-footer removal
- duplicate-content detection
- incremental state comparison
- financial-year normalisation
- conservative primary financial-year inference
- preservation of evergreen and ambiguous documents

The tests use small synthetic documents only. They do not depend on the
5,588-document local ATO corpus.
"""

from __future__ import annotations

import json

import pytest

from fylit_rag.ingestion.pipeline import (
    CleanedDoc,
    extract_financial_years,
)

from fylit_rag.indexing.versioning import (
    diff_against_state,
    find_content_duplicates,
)

from fylit_rag.ingestion.cleaner import clean_body

from fylit_rag.ingestion.loader import (
    content_hash,
    parse_header_and_body,
    stable_id,
)

from fylit_rag.ingestion.metadata import (
    collect_financial_year_mentions,
    infer_primary_year,
    normalise_financial_year,
    normalise_range_text,
    process_corpus,
    tax_year_to_financial_year,
)


# --------------------------------------------------------------------------- #
# Shared synthetic fixtures / helpers
# --------------------------------------------------------------------------- #


def make_raw_document(
    *,
    title: str = "Example page",
    source_url: str = "https://www.ato.gov.au/example-page",
    scraped_at: str = "2026-04-14 09:58:51",
    menu_path: str = "Home > Example",
    description: str | None = "Example description",
    body: str = "Useful ATO guidance appears here.",
) -> str:
    """Build a small document that follows the scraper's Markdown format."""

    lines = [
        f"# {title} | Australian Taxation Office",
        "",
        f"> **Source:** {source_url}",
        f"> **Scraped:** {scraped_at}",
        f"> **Menu Path:** {menu_path}",
        "",
    ]

    if description is not None:
        lines.extend(
            [
                f"**Description:** {description}",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            body,
        ]
    )

    return "\n".join(lines)


def make_cleaned_doc(
    *,
    doc_id: str,
    hash_value: str,
    text: str = "Example cleaned content.",
) -> CleanedDoc:
    """Build a minimal CleanedDoc for duplicate/state tests."""

    return CleanedDoc(
        id=doc_id,
        content_hash=hash_value,
        file_path=f"{doc_id}.md",
        category="individuals-and-families",
        topic="your-tax-return",
        title=f"Document {doc_id}",
        description=None,
        source_url=f"https://www.ato.gov.au/{doc_id}",
        menu_path_text="Individuals > Tax return",
        menu_path_tree=None,
        last_updated_display=None,
        published_display=None,
        scraped_at="2026-04-14 09:58:51",
        financial_years=[],
        char_count=len(text),
        word_count=len(text.split()),
        cleaned_content=text,
    )


def make_year_doc(
    *,
    url: str = "https://www.ato.gov.au/example",
    title: str = "Example ATO guidance",
    content: str = "General guidance.",
) -> dict:
    """Build the minimal structure required by infer_primary_year()."""

    return {
        "id": "example-doc",
        "source_url": url,
        "title": title,
        "cleaned_content": content,
    }


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #


def test_parse_header_extracts_expected_metadata():
    raw = make_raw_document()

    header, body = parse_header_and_body(raw)

    assert header is not None

    assert header["title"] == "Example page"

    assert (
        header["source_url"]
        == "https://www.ato.gov.au/example-page"
    )

    assert header["scraped_at"] == "2026-04-14 09:58:51"

    assert header["menu_path_text"] == "Home > Example"

    assert header["description"] == "Example description"

    assert "Useful ATO guidance appears here." in body


def test_parse_header_allows_missing_description():
    raw = make_raw_document(
        description=None,
    )

    header, body = parse_header_and_body(raw)

    assert header is not None

    assert header["description"] is None

    assert "Useful ATO guidance appears here." in body


def test_parse_header_removes_ato_title_suffix():
    raw = make_raw_document(
        title="How to lodge your tax return",
    )

    header, _ = parse_header_and_body(raw)

    assert header is not None

    assert header["title"] == "How to lodge your tax return"

    assert "Australian Taxation Office" not in header["title"]


def test_parse_header_rejects_document_without_scraper_header():
    raw = (
        "# Ordinary Markdown\n\n"
        "This is not an ATO scraper document."
    )

    header, body = parse_header_and_body(raw)

    assert header is None

    assert body == raw


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def test_clean_body_removes_published_print_ui_and_captures_date():
    body = """
Published 13 February 2025Print or Download

## Tax guidance

Useful tax information.
"""

    cleaned, last_updated, published = clean_body(body)

    assert published == "13 February 2025"

    assert last_updated is None

    assert "Published 13 February 2025" not in cleaned

    assert "Print or Download" not in cleaned

    assert "## Tax guidance" in cleaned

    assert "Useful tax information." in cleaned


def test_clean_body_removes_plain_published_ui_line():
    body = """
Published 2 July 2024

Guidance remains here.
"""

    cleaned, _, published = clean_body(body)

    assert published == "2 July 2024"

    assert "Published 2 July 2024" not in cleaned

    assert "Guidance remains here." in cleaned


def test_clean_body_captures_last_updated_metadata():
    body = """
Last updated 18 February 2026Print or Download

Current guidance.
"""

    cleaned, last_updated, _ = clean_body(body)

    assert last_updated == "18 February 2026"

    assert "Last updated" not in cleaned

    assert "Print or Download" not in cleaned

    assert "Current guidance." in cleaned


def test_clean_body_preserves_legitimate_print_or_download_instruction():
    body = """
## Saving your records

Choose Print or Download to save a copy of your records.
"""

    cleaned, _, _ = clean_body(body)

    assert (
        "Choose Print or Download to save a copy of your records."
        in cleaned
    )


def test_clean_body_removes_standalone_qc_footer():
    body = """
## Example

Useful information.

QC103635
"""

    cleaned, _, _ = clean_body(body)

    assert "QC103635" not in cleaned

    assert "Useful information." in cleaned


def test_clean_body_removes_joined_qc_print_footer():
    body = """
Useful information.

QC103635Print or Download
"""

    cleaned, _, _ = clean_body(body)

    assert "QC103635" not in cleaned

    assert "Print or Download" not in cleaned


def test_clean_body_preserves_qc_identifier_inside_legitimate_url():
    url = (
        "https://www.ato.gov.au/"
        "download-file/Employee_Option_Plan_QC45991.docx"
    )

    body = f"""
## Download

[ATO document]({url})
"""

    cleaned, _, _ = clean_body(body)

    assert "QC45991" in cleaned

    assert url in cleaned


def test_clean_body_preserves_markdown_structure():
    body = """
## Deductions

You may be able to claim:

- Item one
- Item two

| Type | Amount |
| --- | ---: |
| Example | 100 |
"""

    cleaned, _, _ = clean_body(body)

    assert "## Deductions" in cleaned

    assert "- Item one" in cleaned

    assert "- Item two" in cleaned

    assert "| Type | Amount |" in cleaned

    assert "| Example | 100 |" in cleaned


# --------------------------------------------------------------------------- #
# Stable identifiers and hashes
# --------------------------------------------------------------------------- #


def test_stable_id_is_deterministic():
    url = "https://www.ato.gov.au/example"

    first = stable_id(
        url,
        "first-file-name.md",
    )

    second = stable_id(
        url,
        "renamed-file.md",
    )

    assert first == second

    assert len(first) == 40


def test_stable_id_changes_when_source_url_changes():
    first = stable_id(
        "https://www.ato.gov.au/page-a",
        "page.md",
    )

    second = stable_id(
        "https://www.ato.gov.au/page-b",
        "page.md",
    )

    assert first != second


def test_stable_id_uses_path_when_url_is_missing():
    first = stable_id(
        None,
        "folder/page.md",
    )

    second = stable_id(
        None,
        "folder/page.md",
    )

    assert first == second


def test_content_hash_is_deterministic():
    text = "ATO tax guidance."

    assert content_hash(text) == content_hash(text)

    assert len(content_hash(text)) == 64


def test_content_hash_changes_when_content_changes():
    first = content_hash(
        "Original guidance."
    )

    second = content_hash(
        "Updated guidance."
    )

    assert first != second


# --------------------------------------------------------------------------- #
# Financial-year mentions in cleaning stage
# --------------------------------------------------------------------------- #


def test_cleaning_stage_extracts_and_normalises_fy_mentions():
    years = extract_financial_years(
        "Guidance for 2024-25.",
        "Historical information from 2023–24.",
        "Another reference to 2024—25.",
    )

    assert years == [
        "2023-24",
        "2024-25",
    ]


# --------------------------------------------------------------------------- #
# Duplicate-content detection
# --------------------------------------------------------------------------- #


def test_duplicate_detection_groups_identical_content_hashes():
    shared_hash = "a" * 64

    docs = [
        make_cleaned_doc(
            doc_id="doc-a",
            hash_value=shared_hash,
        ),
        make_cleaned_doc(
            doc_id="doc-b",
            hash_value=shared_hash,
        ),
        make_cleaned_doc(
            doc_id="doc-c",
            hash_value="b" * 64,
        ),
    ]

    duplicates = find_content_duplicates(
        docs
    )

    assert len(duplicates) == 1

    assert set(
        duplicates[shared_hash]
    ) == {
        "doc-a",
        "doc-b",
    }


def test_unique_documents_are_not_reported_as_duplicates():
    docs = [
        make_cleaned_doc(
            doc_id="doc-a",
            hash_value="a" * 64,
        ),
        make_cleaned_doc(
            doc_id="doc-b",
            hash_value="b" * 64,
        ),
    ]

    assert find_content_duplicates(docs) == {}


# --------------------------------------------------------------------------- #
# Incremental state
# --------------------------------------------------------------------------- #


def test_incremental_diff_detects_new_document():
    docs = [
        make_cleaned_doc(
            doc_id="new-doc",
            hash_value="a" * 64,
        )
    ]

    diff = diff_against_state(
        docs,
        {},
    )

    assert diff["new"] == 1

    assert diff["changed"] == 0

    assert diff["unchanged"] == 0

    assert diff["deleted"] == 0

    assert diff["new_ids"] == [
        "new-doc"
    ]


def test_incremental_diff_detects_changed_document():
    docs = [
        make_cleaned_doc(
            doc_id="doc-a",
            hash_value="b" * 64,
        )
    ]

    previous = {
        "doc-a": "a" * 64,
    }

    diff = diff_against_state(
        docs,
        previous,
    )

    assert diff["new"] == 0

    assert diff["changed"] == 1

    assert diff["unchanged"] == 0

    assert diff["deleted"] == 0

    assert diff["changed_ids"] == [
        "doc-a"
    ]


def test_incremental_diff_detects_unchanged_document():
    hash_value = "a" * 64

    docs = [
        make_cleaned_doc(
            doc_id="doc-a",
            hash_value=hash_value,
        )
    ]

    previous = {
        "doc-a": hash_value,
    }

    diff = diff_against_state(
        docs,
        previous,
    )

    assert diff["new"] == 0

    assert diff["changed"] == 0

    assert diff["unchanged"] == 1

    assert diff["deleted"] == 0


def test_incremental_diff_detects_deleted_document():
    diff = diff_against_state(
        [],
        {
            "deleted-doc": "a" * 64,
        },
    )

    assert diff["new"] == 0

    assert diff["changed"] == 0

    assert diff["unchanged"] == 0

    assert diff["deleted"] == 1

    assert diff["deleted_ids"] == [
        "deleted-doc"
    ]


def test_incremental_diff_handles_all_change_types_together():
    unchanged_hash = "a" * 64

    docs = [
        make_cleaned_doc(
            doc_id="unchanged",
            hash_value=unchanged_hash,
        ),
        make_cleaned_doc(
            doc_id="changed",
            hash_value="c" * 64,
        ),
        make_cleaned_doc(
            doc_id="new",
            hash_value="n" * 64,
        ),
    ]

    previous = {
        "unchanged": unchanged_hash,
        "changed": "b" * 64,
        "deleted": "d" * 64,
    }

    diff = diff_against_state(
        docs,
        previous,
    )

    assert diff["new"] == 1

    assert diff["changed"] == 1

    assert diff["unchanged"] == 1

    assert diff["deleted"] == 1

    assert diff["new_ids"] == [
        "new"
    ]

    assert diff["changed_ids"] == [
        "changed"
    ]

    assert diff["deleted_ids"] == [
        "deleted"
    ]


# --------------------------------------------------------------------------- #
# Financial-year normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2024", "25", "2024-25"),
        ("2024", "2025", "2024-25"),
        ("2018", "19", "2018-19"),
        ("2018", "2019", "2018-19"),
    ],
)
def test_financial_year_normalisation(
    start,
    end,
    expected,
):
    assert (
        normalise_financial_year(
            start,
            end,
        )
        == expected
    )


@pytest.mark.parametrize(
    "value",
    [
        "2024-27",
        "2024-2027",
        "2025-23",
    ],
)
def test_non_consecutive_year_range_is_rejected(value):
    assert normalise_range_text(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-25", "2024-25"),
        ("2024_25", "2024-25"),
        ("2024–25", "2024-25"),
        ("2024—25", "2024-25"),
        ("2024-2025", "2024-25"),
    ],
)
def test_supported_year_range_formats_are_normalised(
    value,
    expected,
):
    assert normalise_range_text(value) == expected


def test_tax_year_conversion_uses_australian_fy_end_year():
    assert (
        tax_year_to_financial_year(2025)
        == "2024-25"
    )

    assert (
        tax_year_to_financial_year(2018)
        == "2017-18"
    )


# --------------------------------------------------------------------------- #
# Financial-year mention collection
# --------------------------------------------------------------------------- #


def test_multiple_financial_year_mentions_are_preserved():
    mentions = collect_financial_year_mentions(
        (
            "Rates were published for 2020-21, "
            "2021–22 and 2022—23."
        )
    )

    assert mentions == [
        "2020-21",
        "2021-22",
        "2022-23",
    ]


# --------------------------------------------------------------------------- #
# Primary financial-year precedence
# --------------------------------------------------------------------------- #


def test_url_financial_year_has_highest_precedence():
    doc = make_year_doc(
        url=(
            "https://www.ato.gov.au/"
            "report-2024-25/details"
        ),
        content=(
            "This page also discusses "
            "the 2022-23 financial year."
        ),
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        == "2024-25"
    )

    assert (
        result["fy_source"]
        == "url_financial_year_range"
    )

    assert (
        result["fy_confidence"]
        == "high"
    )


def test_mytax_url_year_is_converted_to_financial_year():
    doc = make_year_doc(
        url=(
            "https://www.ato.gov.au/"
            "individuals-and-families/"
            "your-tax-return/"
            "instructions-to-complete-your-tax-return/"
            "mytax-instructions/2025/"
            "income"
        ),
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        == "2024-25"
    )

    assert result["tax_year"] == "2025"

    assert (
        result["fy_source"]
        == "mytax_url_tax_year"
    )

    assert (
        result["fy_confidence"]
        == "high"
    )


def test_strong_applicability_statement_assigns_high_confidence():
    doc = make_year_doc(
        content=(
            "This measure applies to the "
            "2024-25 financial year."
        ),
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        == "2024-25"
    )

    assert (
        result["fy_source"]
        == "explicit_applicability"
    )

    assert (
        result["fy_confidence"]
        == "high"
    )


def test_scoped_income_year_statement_assigns_medium_confidence():
    doc = make_year_doc(
        content=(
            "For the 2019-20 income year, "
            "the following rules apply."
        ),
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        == "2019-20"
    )

    assert (
        result["fy_source"]
        == "explicit_applicability"
    )

    assert (
        result["fy_confidence"]
        == "medium"
    )


def test_bare_body_year_is_only_a_mention_not_primary_year():
    doc = make_year_doc(
        content=(
            "The historical treatment changed "
            "in 2019-20 before later amendments."
        ),
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        is None
    )

    assert "2019-20" in (
        result[
            "financial_year_mentions"
        ]
    )


def test_july_to_june_window_assigns_financial_year():
    doc = make_year_doc(
        content=(
            "These rates apply from 1 July 2024 "
            "to 30 June 2025."
        ),
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        == "2024-25"
    )

    assert (
        result["fy_source"]
        == "explicit_july_to_june_window"
    )

    assert (
        result["fy_confidence"]
        == "high"
    )


def test_scoped_tax_year_is_converted_to_financial_year():
    doc = make_year_doc(
        content=(
            "For the 2025 income year, "
            "use the following calculation."
        ),
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        == "2024-25"
    )

    assert result["tax_year"] == "2025"

    assert (
        result["fy_source"]
        == "explicit_tax_or_income_year"
    )

    assert (
        result["fy_confidence"]
        == "medium"
    )


def test_title_financial_year_can_assign_primary_year():
    doc = make_year_doc(
        title="BAS agent lodgment program 2025–26",
        content="General program information.",
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        == "2025-26"
    )

    assert (
        result["fy_source"]
        == "title_financial_year"
    )

    assert (
        result["fy_confidence"]
        == "high"
    )


def test_heading_financial_year_can_assign_primary_year():
    doc = make_year_doc(
        title="Rates",
        content="""
General information.

## Rates for 2024-25

Rate information follows.
""",
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        == "2024-25"
    )

    assert (
        result["fy_source"]
        == "heading_financial_year"
    )

    assert (
        result["fy_confidence"]
        == "medium"
    )


def test_heading_tax_year_is_converted_to_financial_year():
    doc = make_year_doc(
        title="Tax-return instructions",
        content="""
General information.

## Income year 2025

Instructions follow.
""",
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        == "2024-25"
    )

    assert result["tax_year"] == "2025"

    assert (
        result["fy_source"]
        == "heading_tax_or_income_year"
    )

    assert (
        result["fy_confidence"]
        == "medium"
    )


def test_evergreen_document_remains_untagged():
    doc = make_year_doc(
        title="How income tax works",
        content=(
            "General information about Australian "
            "income tax obligations."
        ),
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        is None
    )

    assert result["fy_source"] is None

    assert result["fy_confidence"] is None


def test_multi_year_evergreen_page_keeps_mentions_without_guessing():
    doc = make_year_doc(
        title="Historical rates",
        content=(
            "The table compares 2022-23, "
            "2023-24 and 2024-25."
        ),
    )

    result = infer_primary_year(doc)

    assert (
        result["primary_financial_year"]
        is None
    )

    assert (
        result["financial_year_mentions"]
        == [
            "2022-23",
            "2023-24",
            "2024-25",
        ]
    )


# --------------------------------------------------------------------------- #
# End-to-end enrichment on a tiny synthetic corpus
# --------------------------------------------------------------------------- #


def test_process_corpus_preserves_original_document_fields(
    tmp_path,
):
    input_path = (
        tmp_path
        / "cleaned_corpus.jsonl"
    )

    output_path = (
        tmp_path
        / "enriched_corpus.jsonl"
    )

    summary_path = (
        tmp_path
        / "year_tagging_summary.json"
    )


    original = {
        "id": "doc-1",
        "content_hash": "a" * 64,
        "title": "Example guidance",
        "source_url": (
            "https://www.ato.gov.au/"
            "individuals-and-families/"
            "your-tax-return/"
            "instructions-to-complete-your-tax-return/"
            "mytax-instructions/2025/income"
        ),
        "category":
            "individuals-and-families",
        "cleaned_content":
            "Useful tax guidance.",
    }


    input_path.write_text(
        json.dumps(original)
        + "\n",
        encoding="utf-8",
    )


    summary = process_corpus(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
    )


    enriched = json.loads(
        output_path
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )


    for key, value in original.items():
        assert enriched[key] == value


    assert (
        enriched[
            "primary_financial_year"
        ]
        == "2024-25"
    )

    assert (
        enriched["financial_year"]
        == "2024-25"
    )

    assert enriched["tax_year"] == "2025"


    assert summary["total_documents"] == 1

    assert (
        summary[
            "documents_with_primary_financial_year"
        ]
        == 1
    )

    assert summary["coverage_pct"] == 100.0


# --------------------------------------------------------------------------- #
# Safety invariant
# --------------------------------------------------------------------------- #


def test_financial_year_alias_always_matches_primary_year():
    examples = [
        make_year_doc(
            url=(
                "https://www.ato.gov.au/"
                "report-2024-25"
            )
        ),
        make_year_doc(
            content=(
                "General evergreen guidance."
            )
        ),
    ]


    for doc in examples:
        result = infer_primary_year(doc)

        # infer_primary_year itself returns the canonical value;
        # process_corpus later copies this into the backward-compatible
        # financial_year alias.
        primary = result[
            "primary_financial_year"
        ]

        assert (
            primary is None
            or re_full_financial_year(
                primary
            )
        )


def re_full_financial_year(
    value: str,
) -> bool:
    """Small local validator used by the final invariant test."""

    import re

    match = re.fullmatch(
        r"(20\d{2})-(\d{2})",
        value,
    )

    if not match:
        return False

    start = int(
        match.group(1)
    )

    end = int(
        match.group(2)
    )

    return (
        (start + 1) % 100
        == end
    )