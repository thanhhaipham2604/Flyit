"""
Corpus audit
============

Fylit ATO RAG System — Zone A (offline ingestion) — Step 1: Audit.

Run it via the root-level ``auditing.py`` wrapper, or
``python -m fylit_rag.ingestion.audit``.

This is a READ-ONLY diagnostic pass over the RAW .md corpus. It does not
clean, modify, or write back to any source file. Its only job is to produce
evidence — counts, percentages, and concrete file examples — so that
cleaning rules (Step 2) and metadata/FY-tagging rules (Step 3) are based on
what the whole corpus actually looks like, not on a handful of manually
inspected samples.

Checks performed
-----------------
1. Header structure compliance  — does every file match the expected
   "# Title" / Source / Scraped / Menu Path / (Description) / --- header?
   If not, which field is missing?
2. Encoding                     — any files that aren't clean UTF-8?
3. Boilerplate consistency      — how many files actually contain the nav
   block, the "Log in" block, "Print or Download", "Last updated ..."?
   (We assumed "always", based on 3 samples — this checks that at scale.)
4. QC footer code variants      — how many QC codes per file, in what shape
   (lone line / glued to "Print or Download" / neither / more than 2)?
5. Possible duplicate list items — heuristic flag for the "repeated bullet"
   scraping artifact seen in one of the samples.
6. Markdown tables               — how many files contain pipe tables, since
   that affects chunking strategy later.
7. Financial-year mention shape — where FY mentions show up (URL slug vs.
   title/H1 vs. body) so Step 3's precedence chain can be built on real
   numbers instead of assumptions.
8. Coverage vs. menu_tree.json  — % of files whose source_url is a known key.
9. Coverage vs. visited_urls.json — URLs that were visited but never saved
   as a file (and vice versa).
10. Per-category file counts and size distribution.

Usage
-----
    python auditing.py --input data/ato_corpus --output data/processed/audit

Either data/ato_corpus or data/ato_corpus/atoData is accepted; the layout is
resolved the same way the preprocessing pipeline resolves it.

Outputs (all under --output, nothing written anywhere else):
    audit_report.json   -> full structured findings + example file paths
    audit_summary.md     -> human-readable summary for the team
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fylit_rag.ingestion.pipeline import resolve_corpus_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("auditing")

MAX_EXAMPLES = 15  # cap how many example file paths we keep per finding

# --------------------------------------------------------------------------- #
# Patterns (kept independent from datacleaning.py on purpose — an audit
# should not silently inherit assumptions baked into the cleaner).
# --------------------------------------------------------------------------- #

HEADER_TITLE_RE = re.compile(r"^#\s*(?P<title>.+?)\s*\n", re.MULTILINE)
SOURCE_RE = re.compile(r"^>\s*\*\*Source:\*\*\s*(?P<url>\S+)\s*$", re.MULTILINE)
SCRAPED_RE = re.compile(r"^>\s*\*\*Scraped:\*\*\s*(?P<val>.+?)\s*$", re.MULTILINE)
MENU_PATH_RE = re.compile(r"^>\s*\*\*Menu Path:\*\*\s*(?P<val>.+?)\s*$", re.MULTILINE)
DESCRIPTION_RE = re.compile(r"^\*\*Description:\*\*\s*(?P<val>.+?)\s*$", re.MULTILINE)
DIVIDER_RE = re.compile(r"^---\s*$", re.MULTILINE)

NAV_BLOCK_MARKERS = [
    "[ATO Community]",
    "[Legal Database]",
    "[What's New]",
    "## Log in to ATO online services",
    "Access secure services, view your details and lodge online.",
]
PRINT_OR_DOWNLOAD = "Print or Download"
LAST_UPDATED_RE = re.compile(r"Last updated\s+[A-Za-z0-9 ,]+")
QC_CODE_RE = re.compile(r"QC(\d+)")
QC_GLUED_RE = re.compile(r"QC\d+Print or Download")
QC_LONE_LINE_RE = re.compile(r"^\s*QC\d+\s*$", re.MULTILINE)

TABLE_ROW_RE = re.compile(r"^\|.+\|\s*$", re.MULTILINE)
TABLE_SEP_RE = re.compile(r"^\|[\s:\-|]+\|\s*$", re.MULTILINE)

FINANCIAL_YEAR_RE = re.compile(r"\b(20\d{2})\s*[-\u2013\u2014]\s*(\d{2})\b")
URL_FY_RE = re.compile(r"(20\d{2})[-\u2013\u2014](\d{2})")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def discover_md_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def read_raw(path: Path) -> tuple[str, bool]:
    """Returns (text, had_encoding_issue).

    Always a str: undecodable bytes fall back to errors="replace" rather than
    returning None, and a genuine I/O failure raises out of read_bytes().
    """
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
        return text, False
    except UnicodeDecodeError:
        text = raw_bytes.decode("utf-8", errors="replace")
        return text, True


def find_duplicate_list_items(body: str) -> int:
    """Heuristic for the scraping artifact where a bullet's sub-items get
    re-emitted as their own top-level bullets later in the same section
    (seen verbatim in the BTR sample). Flags exact-duplicate non-trivial
    bullet lines within the same file.
    """
    bullets = re.findall(r"^-\s+(.{25,})$", body, re.MULTILINE)
    counts = Counter(bullets)
    return sum(1 for _, n in counts.items() if n > 1)


def classify_qc_footer(text: str) -> str:
    glued = bool(QC_GLUED_RE.search(text))
    lone = bool(QC_LONE_LINE_RE.search(text))
    total_codes = len(QC_CODE_RE.findall(text))
    if total_codes == 0:
        return "missing"
    if glued and lone and total_codes == 2:
        return "expected_pair"  # the pattern our cleaning currently assumes
    if total_codes > 2:
        return "more_than_two"
    return "other_shape"


def classify_fy_location(source_url: str | None, title: str, body: str) -> str:
    """Cheap preview of where an FY mention (if any) shows up, to inform
    Step 3's precedence chain (URL > explicit statement > heading > null).
    This does NOT implement the real precedence logic — just measures shape.
    """
    if source_url and URL_FY_RE.search(source_url):
        return "in_url"
    if title and FINANCIAL_YEAR_RE.search(title):
        return "in_title"
    headings = "\n".join(re.findall(r"^#{1,6}\s.+$", body, re.MULTILINE))
    if FINANCIAL_YEAR_RE.search(headings):
        return "in_heading"
    if FINANCIAL_YEAR_RE.search(body):
        return "in_body_only"
    return "none_found"


def load_json_safely(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s (%s)", path, exc)
        return None


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #

def audit_file(path: Path, root: Path) -> dict:
    rel_path = str(path.relative_to(root))
    text, encoding_issue = read_raw(path)

    findings: dict = {
        "file_path": rel_path,
        "size_bytes": path.stat().st_size,
        "encoding_issue": encoding_issue,
    }

    title_m = HEADER_TITLE_RE.search(text)
    source_m = SOURCE_RE.search(text)
    scraped_m = SCRAPED_RE.search(text)
    menu_m = MENU_PATH_RE.search(text)
    desc_m = DESCRIPTION_RE.search(text)
    divider_m = DIVIDER_RE.search(text)

    missing_fields = []
    if not title_m:
        missing_fields.append("title")
    if not source_m:
        missing_fields.append("source_url")
    if not scraped_m:
        missing_fields.append("scraped_at")
    if not menu_m:
        missing_fields.append("menu_path")
    if not desc_m:
        missing_fields.append("description")
    if not divider_m:
        missing_fields.append("header_divider")

    findings["missing_header_fields"] = missing_fields
    findings["header_ok"] = len(missing_fields) == 0 or missing_fields == ["description"]

    source_url = source_m.group("url") if source_m else None
    title = title_m.group("title") if title_m else ""
    findings["source_url"] = source_url

    # Body = everything after the header divider, if we found one.
    body = text[divider_m.end():] if divider_m else text

    findings["nav_markers_present"] = [m for m in NAV_BLOCK_MARKERS if m in body]
    findings["print_or_download_present"] = PRINT_OR_DOWNLOAD in body
    findings["last_updated_present"] = bool(LAST_UPDATED_RE.search(body))
    findings["qc_footer_shape"] = classify_qc_footer(text)
    findings["duplicate_list_item_groups"] = find_duplicate_list_items(body)
    findings["has_table"] = bool(TABLE_ROW_RE.search(body) and TABLE_SEP_RE.search(body))
    findings["fy_location"] = classify_fy_location(source_url, title, body)

    category = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) else "unknown"
    findings["category"] = category

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/ato_corpus"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/audit"))
    args = parser.parse_args()

    # Accept either data/ato_corpus or data/ato_corpus/atoData, exactly as the
    # preprocessing pipeline does. The audit and the pipeline must never
    # disagree about where the corpus is.
    input_root: Path = resolve_corpus_root(args.input)
    output_root: Path = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        log.error("Input folder does not exist: %s", input_root)
        return 1

    # menu_tree.json is a URL -> menu-path mapping. Anything else (a list, a
    # truncated file) is unusable here, so fall back to empty rather than
    # failing on .keys() much further down.
    loaded_menu_tree = load_json_safely(input_root / "menu_tree.json")
    menu_tree: dict = loaded_menu_tree if isinstance(loaded_menu_tree, dict) else {}
    visited_raw = load_json_safely(input_root / "visited_urls.json")
    if isinstance(visited_raw, dict):
        visited_urls = set(visited_raw.get("visited", []))
    elif isinstance(visited_raw, list):
        visited_urls = set(visited_raw)
    else:
        visited_urls = set()

    md_files = discover_md_files(input_root)
    if not md_files:
        log.error("No .md files found under %s", input_root)
        return 1
    log.info("Auditing %d .md files under %s", len(md_files), input_root)

    all_findings: list[dict] = []
    for i, path in enumerate(md_files, start=1):
        all_findings.append(audit_file(path, input_root))
        if i % 1000 == 0 or i == len(md_files):
            log.info("Audited %d / %d files", i, len(md_files))

    n = len(all_findings)

    # --- Aggregate ---
    header_ok = sum(1 for f in all_findings if f["header_ok"])
    missing_field_counter: Counter = Counter()
    for f in all_findings:
        missing_field_counter.update(f["missing_header_fields"])

    encoding_issues = [f for f in all_findings if f["encoding_issue"]]
    header_bad = [f for f in all_findings if not f["header_ok"]]

    nav_full_block = sum(1 for f in all_findings if len(f["nav_markers_present"]) == len(NAV_BLOCK_MARKERS))
    nav_partial = [
        f for f in all_findings
        if 0 < len(f["nav_markers_present"]) < len(NAV_BLOCK_MARKERS)
    ]
    nav_absent = [f for f in all_findings if len(f["nav_markers_present"]) == 0]

    qc_shape_counter = Counter(f["qc_footer_shape"] for f in all_findings)
    qc_not_expected = [f for f in all_findings if f["qc_footer_shape"] != "expected_pair"]

    dup_list_files = [f for f in all_findings if f["duplicate_list_item_groups"] > 0]

    table_files = [f for f in all_findings if f["has_table"]]

    fy_location_counter = Counter(f["fy_location"] for f in all_findings)

    by_category_count: Counter = Counter(f["category"] for f in all_findings)
    by_category_sizes: dict = defaultdict(list)
    for f in all_findings:
        by_category_sizes[f["category"]].append(f["size_bytes"])
    size_stats_by_category = {
        cat: {
            "count": len(sizes),
            "min_bytes": min(sizes),
            "max_bytes": max(sizes),
            "mean_bytes": round(statistics.mean(sizes), 1),
            "median_bytes": statistics.median(sizes),
        }
        for cat, sizes in by_category_sizes.items()
    }

    # --- Coverage vs menu_tree.json ---
    source_urls = {f["source_url"] for f in all_findings if f["source_url"]}
    menu_tree_keys = set(menu_tree.keys())
    files_not_in_menu_tree = sorted(source_urls - menu_tree_keys)
    menu_tree_coverage_pct = (
        round(100 * len(source_urls & menu_tree_keys) / len(source_urls), 2)
        if source_urls else 0.0
    )

    # --- Coverage vs visited_urls.json ---
    visited_not_scraped = sorted(visited_urls - source_urls) if visited_urls else []
    scraped_not_visited = sorted(source_urls - visited_urls) if visited_urls else []

    report: dict[str, Any] = {
        "total_files": n,
        "header_compliance": {
            "header_ok_count": header_ok,
            "header_ok_pct": round(100 * header_ok / n, 2),
            "missing_field_counts": dict(missing_field_counter),
            "example_bad_header_files": [f["file_path"] for f in header_bad[:MAX_EXAMPLES]],
        },
        "encoding": {
            "files_with_issues": len(encoding_issues),
            "examples": [f["file_path"] for f in encoding_issues[:MAX_EXAMPLES]],
        },
        "nav_boilerplate": {
            "full_nav_block_count": nav_full_block,
            "full_nav_block_pct": round(100 * nav_full_block / n, 2),
            "partial_nav_block_count": len(nav_partial),
            "partial_nav_block_examples": [f["file_path"] for f in nav_partial[:MAX_EXAMPLES]],
            "no_nav_block_count": len(nav_absent),
            "no_nav_block_examples": [f["file_path"] for f in nav_absent[:MAX_EXAMPLES]],
        },
        "qc_footer": {
            "shape_counts": dict(qc_shape_counter),
            "note": "'expected_pair' is the QC123Print-or-Download + lone-QC123 "
                    "pattern the cleaning script currently assumes.",
            "examples_not_expected_pair": [f["file_path"] for f in qc_not_expected[:MAX_EXAMPLES]],
        },
        "possible_duplicate_list_items": {
            "files_affected": len(dup_list_files),
            "files_affected_pct": round(100 * len(dup_list_files) / n, 2),
            "examples": [f["file_path"] for f in dup_list_files[:MAX_EXAMPLES]],
        },
        "markdown_tables": {
            "files_with_tables": len(table_files),
            "files_with_tables_pct": round(100 * len(table_files) / n, 2),
            "examples": [f["file_path"] for f in table_files[:MAX_EXAMPLES]],
        },
        "financial_year_location_preview": dict(fy_location_counter),
        "documents_by_category": dict(by_category_count),
        "size_stats_by_category_bytes": size_stats_by_category,
        "menu_tree_coverage": {
            "menu_tree_urls_loaded": len(menu_tree_keys),
            "files_with_source_url": len(source_urls),
            "coverage_pct": menu_tree_coverage_pct,
            "files_missing_from_menu_tree_examples": files_not_in_menu_tree[:MAX_EXAMPLES],
            "files_missing_from_menu_tree_total": len(files_not_in_menu_tree),
        },
        "visited_urls_coverage": {
            "visited_urls_loaded": len(visited_urls),
            "visited_but_not_scraped_total": len(visited_not_scraped),
            "visited_but_not_scraped_examples": visited_not_scraped[:MAX_EXAMPLES],
            "scraped_but_not_in_visited_total": len(scraped_not_visited),
            "scraped_but_not_in_visited_examples": scraped_not_visited[:MAX_EXAMPLES],
        },
    }

    report_path = output_root / "audit_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)

    # --- Human-readable summary ---
    summary_lines = [
        "# ATO corpus audit summary",
        "",
        f"Total files audited: **{n}**",
        "",
        "## Header compliance",
        f"- OK (all required fields present): {header_ok} ({report['header_compliance']['header_ok_pct']}%)",
        f"- Missing-field counts: {dict(missing_field_counter)}",
        "",
        "## Encoding",
        f"- Files with non-clean UTF-8: {len(encoding_issues)}",
        "",
        "## Nav boilerplate consistency",
        f"- Full nav block present: {nav_full_block} ({round(100 * nav_full_block / n, 2)}%)",
        f"- Partial nav block: {len(nav_partial)}",
        f"- No nav block at all: {len(nav_absent)}",
        "",
        "## QC footer code shape",
        f"- {dict(qc_shape_counter)}",
        "",
        "## Possible duplicate list-item artifact",
        f"- Files affected: {len(dup_list_files)} ({round(100 * len(dup_list_files) / n, 2)}%)",
        "",
        "## Markdown tables",
        f"- Files containing at least one table: {len(table_files)} ({round(100 * len(table_files) / n, 2)}%)",
        "",
        "## Financial-year mention location (preview for Step 3)",
        f"- {dict(fy_location_counter)}",
        "",
        "## Coverage vs menu_tree.json",
        f"- {menu_tree_coverage_pct}% of scraped source_urls are known keys in menu_tree.json",
        f"- Files missing from menu_tree.json: {len(files_not_in_menu_tree)}",
        "",
        "## Coverage vs visited_urls.json",
        f"- Visited but never scraped into a file: {len(visited_not_scraped)}",
        f"- Scraped but not present in visited_urls.json: {len(scraped_not_visited)}",
        "",
        "## Documents by category",
        f"- {dict(by_category_count)}",
        "",
        "Full details (including example file paths per finding) are in audit_report.json.",
    ]
    summary_path = output_root / "audit_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    log.info("=" * 60)
    log.info("AUDIT DONE — %d files", n)
    log.info("Header OK: %d (%.2f%%)", header_ok, report["header_compliance"]["header_ok_pct"])
    log.info("QC footer shapes: %s", dict(qc_shape_counter))
    log.info("Nav block: full=%d partial=%d none=%d", nav_full_block, len(nav_partial), len(nav_absent))
    log.info("Possible duplicate list items: %d files", len(dup_list_files))
    log.info("menu_tree.json coverage: %.2f%%", menu_tree_coverage_pct)
    log.info("Reports written to %s", output_root)
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())