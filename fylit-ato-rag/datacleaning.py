#!/usr/bin/env python3
"""
datacleaning.py
================

Fylit ATO RAG System — Zone A (offline ingestion) — Data preparation / cleaning stage.

What this script does (and only this — chunking/embedding/indexing are separate,
later stages of the pipeline):

  1. Discover every .md file under the supplied ATO corpus root.
  2. Parse the standard scraper header (Source / Scraped / Menu Path / Description)
     that every file in this corpus carries, and separate it from the page body.
  3. Clean the body: strip repeated site-navigation boilerplate, "Log in to ATO
     online services" blocks, "Print or Download" / "Last updated ..." UI noise,
     and trailing "QC######" footer codes — while KEEPING headings, lists, tables
     and links, per the project brief.
  4. Assign a stable id (sha1 of the source URL) and a content hash (sha256 of the
     cleaned body) to every document, so the same file is never processed twice.
  5. Enrich metadata by cross-referencing menu_tree.json (official breadcrumb) and
     extracting any financial-year mentions (e.g. "2025-26") from the text.
  6. Support incremental runs: compares against a saved state.json and reports
     new / unchanged / changed / deleted documents.
  7. Write:
       - cleaned_corpus.jsonl   -> one cleaned document (+ metadata) per line
       - manifest.json          -> run statistics
       - invalid_files.json     -> files that failed validation, with reasons
       - duplicate_content.json -> documents whose cleaned content hash collides
       - state.json             -> id -> content_hash, for the next incremental run

Usage
-----
    python datacleaning.py \
        --input data/ato_corpus/atoData \
        --output data/processed

No third-party dependencies — standard library only, so anyone on the team can
run it without a fresh pip install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("datacleaning")


# --------------------------------------------------------------------------- #
# Regex patterns
# --------------------------------------------------------------------------- #

# Header block: everything up to (and including) the first standalone "---"
# line. Example:
#
#   # Title | Australian Taxation Office
#
#   > **Source:** https://...
#   > **Scraped:** 2026-04-14 09:58:51
#   > **Menu Path:** Foo > Bar > Baz
#
#   **Description:** blah
#
#   ---
#
HEADER_RE = re.compile(
    r"^#\s*(?P<title>.+?)\s*\n"
    r".*?"
    r"^>\s*\*\*Source:\*\*\s*(?P<source>\S+)\s*$"
    r".*?"
    r"^>\s*\*\*Scraped:\*\*\s*(?P<scraped>.+?)\s*$"
    r".*?"
    r"^>\s*\*\*Menu Path:\*\*\s*(?P<menu_path>.+?)\s*$"
    r"(?:.*?^\*\*Description:\*\*\s*(?P<description>.+?)\s*$)?"
    r".*?"
    r"^---\s*$",
    re.MULTILINE | re.DOTALL,
)

# The " | Australian Taxation Office" suffix on the H1 title line is a site
# template artifact, not part of the actual document title.
TITLE_SUFFIX_RE = re.compile(r"\s*\|\s*Australian Taxation Office\s*$")

# Financial-year mentions, e.g. "2025-26" or "2025–26".
FINANCIAL_YEAR_RE = re.compile(r"\b(20\d{2})\s*[-\u2013\u2014]\s*(\d{2})\b")

# "Last updated <date>" (optionally glued directly to "Print or Download" with
# no whitespace, as the scraper sometimes captures it that way).
LAST_UPDATED_RE = re.compile(
    r"Last updated\s+(?P<date>[A-Za-z0-9 ,]+?)(?=Print or Download|\n|$)"
)

# Trailing QC footer code, e.g. "QC103635", usually alone on its own line,
# sometimes doubled up.
QC_FOOTER_RE = re.compile(r"^\s*QC\d+\s*$", re.MULTILINE)

# Exact boilerplate lines to drop outright (site chrome, not page content).
BOILERPLATE_LINES = {
    "- [ATO Community](https://community.ato.gov.au/s)",
    "- [Legal Database](https://www.ato.gov.au/single-page-applications/legaldatabase)",
    "## Log in to ATO online services",
    "Access secure services, view your details and lodge online.",
    "Search",
    "Print or Download",
}

# Line *prefixes* to drop (covers minor variations in the exact boilerplate
# text across scrape batches).
BOILERPLATE_PREFIXES = (
    "- [What's New](",
)

MIN_BODY_CHARS = 40  # below this, treat the document as empty/invalid


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class CleanedDoc:
    id: str
    content_hash: str
    file_path: str
    category: str
    topic: Optional[str]
    title: str
    description: Optional[str]
    source_url: Optional[str]
    menu_path_text: Optional[str]
    menu_path_tree: Optional[list]
    last_updated_display: Optional[str]
    scraped_at: Optional[str]
    financial_years: list = field(default_factory=list)
    char_count: int = 0
    word_count: int = 0
    cleaned_content: str = ""


# --------------------------------------------------------------------------- #
# Menu tree lookup
# --------------------------------------------------------------------------- #

def load_menu_tree(path: Optional[Path]) -> dict:
    """menu_tree.json maps a full ATO URL -> breadcrumb list, e.g.
    "https://www.ato.gov.au/.../foo": ["Businesses and organisations", "Foo", "Foo"]
    Returns {} if the file is missing or unreadable, rather than failing the run.
    """
    if not path or not path.exists():
        log.warning("menu_tree.json not found at %s — breadcrumb enrichment disabled", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        log.info("Loaded menu_tree.json with %d URL entries", len(data))
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not parse menu_tree.json (%s) — continuing without it", exc)
        return {}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def discover_md_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.md"))
    log.info("Discovered %d .md files under %s", len(files), root)
    return files


def parse_header_and_body(raw_text: str) -> tuple[Optional[dict], str]:
    """Split a raw scraped .md file into (header_fields, body_text).
    Returns (None, raw_text) if the expected header pattern isn't found —
    the caller decides whether that makes the file invalid.
    """
    match = HEADER_RE.search(raw_text)
    if not match:
        return None, raw_text

    header = {
        "title": TITLE_SUFFIX_RE.sub("", match.group("title")).strip(),
        "source_url": match.group("source").strip(),
        "scraped_at": match.group("scraped").strip(),
        "menu_path_text": match.group("menu_path").strip(),
        "description": (match.group("description") or "").strip() or None,
    }
    body = raw_text[match.end():]
    return header, body


def clean_body(body: str) -> tuple[str, Optional[str]]:
    """Strip navigation/UI boilerplate from the page body while preserving
    headings, lists, tables and links. Returns (cleaned_text, last_updated_display).
    """
    last_updated = None
    lu_match = LAST_UPDATED_RE.search(body)
    if lu_match:
        last_updated = lu_match.group("date").strip()

    # Drop "Last updated ..." occurrences (with or without a glued-on
    # "Print or Download") — this is redisplayed UI chrome, not content.
    body = LAST_UPDATED_RE.sub("", body)

    # Drop trailing QC footer codes.
    body = QC_FOOTER_RE.sub("", body)

    cleaned_lines = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped in BOILERPLATE_LINES:
            continue
        if any(stripped.startswith(p) for p in BOILERPLATE_PREFIXES):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Collapse 3+ consecutive blank lines down to a single blank line, and
    # trim leading/trailing whitespace.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned, last_updated


def extract_financial_years(*texts: str) -> list[str]:
    years = set()
    for text in texts:
        if not text:
            continue
        for m in FINANCIAL_YEAR_RE.finditer(text):
            years.add(f"{m.group(1)}-{m.group(2)}")
    return sorted(years)


def stable_id(source_url: Optional[str], fallback_path: str) -> str:
    key = source_url if source_url else f"path:{fallback_path}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def category_and_topic(file_path: Path, root: Path) -> tuple[str, Optional[str]]:
    rel_parts = file_path.relative_to(root).parts
    category = rel_parts[0] if rel_parts else "unknown"
    topic = rel_parts[1] if len(rel_parts) > 2 else None
    return category, topic


# --------------------------------------------------------------------------- #
# Main processing
# --------------------------------------------------------------------------- #

def process_file(
    file_path: Path,
    root: Path,
    menu_tree: dict,
    invalid: list,
) -> Optional[CleanedDoc]:
    rel_path = str(file_path.relative_to(root))

    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        invalid.append({"file_path": rel_path, "reason": f"read error: {exc}"})
        return None

    header, body = parse_header_and_body(raw_text)
    if header is None:
        invalid.append({"file_path": rel_path, "reason": "header pattern not found"})
        return None

    cleaned, last_updated = clean_body(body)

    if len(cleaned) < MIN_BODY_CHARS:
        invalid.append(
            {"file_path": rel_path, "reason": f"cleaned body too short ({len(cleaned)} chars)"}
        )
        return None

    category, topic = category_and_topic(file_path, root)
    menu_path_tree = menu_tree.get(header["source_url"]) if header["source_url"] else None
    financial_years = extract_financial_years(
        header["title"], header.get("description") or "", cleaned
    )

    doc_id = stable_id(header["source_url"], rel_path)
    doc_hash = content_hash(cleaned)

    return CleanedDoc(
        id=doc_id,
        content_hash=doc_hash,
        file_path=rel_path,
        category=category,
        topic=topic,
        title=header["title"],
        description=header.get("description"),
        source_url=header["source_url"],
        menu_path_text=header["menu_path_text"],
        menu_path_tree=menu_path_tree,
        last_updated_display=last_updated,
        scraped_at=header["scraped_at"],
        financial_years=financial_years,
        char_count=len(cleaned),
        word_count=len(cleaned.split()),
        cleaned_content=cleaned,
    )


def find_content_duplicates(docs: list[CleanedDoc]) -> dict[str, list[str]]:
    """Group document ids by content_hash; return only groups with >1 member."""
    by_hash: dict[str, list[str]] = {}
    for doc in docs:
        by_hash.setdefault(doc.content_hash, []).append(doc.id)
    return {h: ids for h, ids in by_hash.items() if len(ids) > 1}


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning("Could not read previous state.json — treating this as a first run")
        return {}


def diff_against_state(docs: list[CleanedDoc], prev_state: dict) -> dict:
    current_ids = {doc.id: doc.content_hash for doc in docs}
    new_ids = [i for i in current_ids if i not in prev_state]
    changed_ids = [
        i for i in current_ids
        if i in prev_state and prev_state[i] != current_ids[i]
    ]
    unchanged_ids = [
        i for i in current_ids
        if i in prev_state and prev_state[i] == current_ids[i]
    ]
    deleted_ids = [i for i in prev_state if i not in current_ids]
    return {
        "new": len(new_ids),
        "changed": len(changed_ids),
        "unchanged": len(unchanged_ids),
        "deleted": len(deleted_ids),
        "deleted_ids": deleted_ids,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/ato_corpus/atoData"),
        help="Root folder of the ATO markdown corpus (contains menu_tree.json, "
             "visited_urls.json and the category subfolders).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed"),
        help="Folder to write cleaned_corpus.jsonl, manifest.json, etc.",
    )
    args = parser.parse_args()

    input_root: Path = args.input
    output_root: Path = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        log.error("Input folder does not exist: %s", input_root)
        return 1

    menu_tree = load_menu_tree(input_root / "menu_tree.json")

    md_files = discover_md_files(input_root)
    if not md_files:
        log.error("No .md files found under %s — nothing to do", input_root)
        return 1

    invalid: list = []
    docs: list[CleanedDoc] = []

    for i, file_path in enumerate(md_files, start=1):
        doc = process_file(file_path, input_root, menu_tree, invalid)
        if doc is not None:
            docs.append(doc)
        if i % 500 == 0 or i == len(md_files):
            log.info("Processed %d / %d files (%d valid so far)", i, len(md_files), len(docs))

    duplicates = find_content_duplicates(docs)
    duplicate_ids = {doc_id for ids in duplicates.values() for doc_id in ids}

    state_path = output_root / "state.json"
    prev_state = load_state(state_path)
    diff = diff_against_state(docs, prev_state)

    # --- Write cleaned_corpus.jsonl ---
    corpus_path = output_root / "cleaned_corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            record = asdict(doc)
            record["is_duplicate_content"] = doc.id in duplicate_ids
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("Wrote %d cleaned documents to %s", len(docs), corpus_path)

    # --- Write state.json for the next incremental run ---
    new_state = {doc.id: doc.content_hash for doc in docs}
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(new_state, f, indent=2)

    # --- Write invalid_files.json ---
    invalid_path = output_root / "invalid_files.json"
    with invalid_path.open("w", encoding="utf-8") as f:
        json.dump(invalid, f, indent=2)

    # --- Write duplicate_content.json ---
    dup_path = output_root / "duplicate_content.json"
    with dup_path.open("w", encoding="utf-8") as f:
        json.dump(duplicates, f, indent=2)

    # --- Write manifest.json ---
    by_category: dict[str, int] = {}
    for doc in docs:
        by_category[doc.category] = by_category.get(doc.category, 0) + 1

    manifest = {
        "input_root": str(input_root),
        "total_files_discovered": len(md_files),
        "valid_documents": len(docs),
        "invalid_files": len(invalid),
        "duplicate_content_groups": len(duplicates),
        "documents_by_category": by_category,
        "incremental_diff_vs_previous_run": diff,
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("=" * 60)
    log.info("DONE")
    log.info("  Valid documents      : %d", len(docs))
    log.info("  Invalid files        : %d", len(invalid))
    log.info("  Duplicate-content grp: %d", len(duplicates))
    log.info("  vs previous run      : new=%d changed=%d unchanged=%d deleted=%d",
              diff["new"], diff["changed"], diff["unchanged"], diff["deleted"])
    log.info("Outputs written to %s", output_root)
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())