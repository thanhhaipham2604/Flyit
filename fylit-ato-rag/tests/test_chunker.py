"""Chunker tests: the boundary rules retrieval and citation depend on.

Stdlib-only and no database, like the schema tests - these are invariants of the
text, not of the store.
"""

import hashlib

from fylit_rag.indexing.schema import Status
from fylit_rag.ingestion.chunker import (
    MAX_CHARS,
    TARGET_CHARS,
    _merge_short,
    chunk_document,
    chunk_text,
)


def paths_and_texts(markdown):
    return _merge_short(chunk_text(markdown))


def body(label, times=8):
    """A section body comfortably over MIN_CHARS, so the merge pass leaves it be."""
    return f"Body text for {label} written at length so it stands alone. " * times


def test_heading_path_tracks_nesting():
    md = (
        f"# Title\n\n{body('the intro')}\n\n"
        f"## Section A\n\n{body('section A')}\n\n"
        f"### Sub A1\n\n{body('sub A1')}\n"
    )
    paths = [p for p, _ in paths_and_texts(md)]
    assert paths[0] == ["Title"]
    assert ["Title", "Section A"] in paths
    assert ["Title", "Section A", "Sub A1"] in paths


def test_a_short_section_is_folded_into_its_neighbour():
    """Sections under MIN_CHARS are poor things to embed alone, so they merge -
    and the merged chunk is labelled with the heading path common to both."""
    md = "# Title\n\nShort intro.\n\n## Section A\n\nAlso short.\n\n### Sub A1\n\nShort too.\n"
    pairs = paths_and_texts(md)
    assert len(pairs) < 3, "short sections should have merged"
    assert pairs[0][0] == ["Title"], "merged chunk must carry the common ancestor path"


def test_a_deeper_heading_pops_back_to_its_own_level():
    """H3 then H2 must not leave the H3 dangling in the path."""
    md = """# T

### Deep one, written long enough to survive the short-section merge pass intact.

## Shallow again, also written at length so that it becomes a chunk of its own.
"""
    paths = [p for p, _ in paths_and_texts(md)]
    assert not any(len(p) > 1 and p[1].startswith("Deep") and len(p) > 2 for p in paths)


def test_a_table_is_never_split():
    row = "| a | b |\n"
    table = "| h | h |\n|---|---|\n" + row * 400          # far beyond MAX_CHARS
    md = f"# T\n\n## Rates\n\n{table}\n"
    texts = [t for _, t in paths_and_texts(md)]
    holding = [t for t in texts if "| a | b |" in t]
    assert len(holding) == 1, "the table was split across chunks"
    assert holding[0].count("| a | b |") == 400, "rows were lost"


def test_a_list_is_never_split():
    items = "".join(f"- item number {i} with some explanatory text after it\n" for i in range(200))
    md = f"# T\n\n## Things\n\n{items}\n"
    texts = [t for _, t in paths_and_texts(md)]
    holding = [t for t in texts if "item number 0 " in t]
    assert len(holding) == 1
    assert holding[0].count("- item number") == 200


def test_a_list_survives_blank_lines_between_items():
    """ATO lists are routinely written with blank lines between items."""
    md = "# T\n\n## L\n\n- first item\n\n- second item\n\n- third item\n"
    texts = [t for _, t in paths_and_texts(md)]
    holding = [t for t in texts if "first item" in t]
    assert "third item" in holding[0]


def test_long_prose_is_split_on_sentence_boundaries():
    sentence = "This is a sentence about deductions that carries a reasonable amount of text. "
    md = f"# T\n\n## Long\n\n{sentence * 80}\n"
    texts = [t for _, t in paths_and_texts(md)]
    bodies = [t for t in texts if "deductions" in t]
    assert len(bodies) > 1, "an over-long paragraph should be split"
    for t in bodies:
        stripped = t.split("\n\n", 1)[-1].strip()
        assert stripped.endswith("."), "a chunk ended mid-sentence"


def test_every_chunk_carries_its_section_heading():
    md = "# T\n\n## Overview\n\n" + ("Body text about the overview section. " * 60)
    for path, text in paths_and_texts(md):
        if len(path) > 1:
            assert path[-1] in text, "the section heading is missing from the chunk text"


def test_packing_many_blocks_stays_near_the_target():
    """A section of many paragraphs is packed to TARGET_CHARS, not to MAX."""
    para = "Sentence of body text for this section. " * 6      # ~240 chars each
    md = "# T\n\n## Section\n\n" + "\n\n".join(para for _ in range(30))
    sizes = [len(t) for _, t in paths_and_texts(md)]
    assert max(sizes) <= TARGET_CHARS + 300, "packing overshot the target badly"
    assert len(sizes) > 3, "a long section should produce several chunks"


def test_a_single_paragraph_under_max_is_kept_whole():
    """Deliberate: prose is only broken up past MAX_CHARS. Splitting every
    paragraph that merely exceeds TARGET would fragment arguments mid-thought."""
    para = "Sentence of body text for this section. " * 40      # ~1600 chars, one block
    md = f"# T\n\n## Section\n\n{para}\n"
    texts = [t for _, t in paths_and_texts(md)]
    assert len(texts) == 1
    assert TARGET_CHARS < len(texts[0]) <= MAX_CHARS


def test_content_before_the_first_heading_is_kept():
    md = "Loose opening prose with no heading above it at all, which must not vanish.\n\n# T\n\nBody.\n"
    assert any("Loose opening prose" in t for _, t in paths_and_texts(md))


def test_empty_document_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  \n") == []


# ---------------------------------------------------------------- ChunkRecord


SAMPLE_META = {
    "id": "doc-1",
    "title": "Build to rent",
    "source_url": "https://ato.gov.au/btr",
    "category": "businesses-and-organisations",
    "topic": "assets-and-property",
    "last_updated_display": "18 February 2026",
    "financial_years": ["2023-24", "2024-25"],
    "primary_financial_year": "2023-24",
}


def sample_markdown():
    return "# Build to rent\n\n## Overview\n\n" + ("Body text about the incentive. " * 50)


def test_chunk_ids_are_deterministic():
    """Re-ingesting an unchanged document must overwrite, not duplicate."""
    first = chunk_document("doc-1", sample_markdown(), SAMPLE_META)
    second = chunk_document("doc-1", sample_markdown(), SAMPLE_META)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id == "doc-1#0"
    assert [c.chunk_ordinal for c in first] == list(range(len(first)))


def test_content_hash_is_of_the_chunk_not_the_document():
    """The upsert uses this to decide whether an embedding is stale, so it has
    to move when the chunk's own text moves."""
    chunks = chunk_document("doc-1", sample_markdown(), SAMPLE_META)
    for c in chunks:
        assert c.content_hash == hashlib.sha256(c.text.encode("utf-8")).hexdigest()
    changed = chunk_document("doc-1", sample_markdown() + "\n\nA new closing sentence.", SAMPLE_META)
    assert changed[-1].content_hash != chunks[-1].content_hash


def test_metadata_lands_on_every_chunk():
    for c in chunk_document("doc-1", sample_markdown(), SAMPLE_META):
        assert c.source_title == "Build to rent"
        assert c.source_url == "https://ato.gov.au/btr"
        assert c.category == "businesses-and-organisations"
        assert c.topic == "assets-and-property"
        assert c.last_updated.isoformat() == "2026-02-18"
        assert c.version == 1
        assert c.status is Status.ACTIVE


def test_financial_year_preserves_existing_year_list():
    """Existing multi-year applicability must be preserved."""
    chunks = chunk_document(
        "doc-1",
        sample_markdown(),
        SAMPLE_META,
    )

    assert chunks[0].financial_year == [
        "2023-24",
        "2024-25",
    ]


def test_primary_financial_year_is_added_when_list_is_empty():
    """A strongly inferred primary FY must reach the chunk index."""
    meta = SAMPLE_META | {
        "financial_years": [],
        "primary_financial_year": "2023-24",
    }

    chunks = chunk_document(
        "doc-1",
        sample_markdown(),
        meta,
    )

    assert chunks[0].financial_year == [
        "2023-24"
    ]


def test_evergreen_documents_get_an_empty_year_list():
    """A document with no year evidence remains evergreen."""
    meta = SAMPLE_META | {
        "financial_years": [],
        "primary_financial_year": None,
    }

    chunks = chunk_document(
        "doc-1",
        sample_markdown(),
        meta,
    )

    assert chunks[0].financial_year == []


def test_version_and_status_are_passed_through():
    chunks = chunk_document(
        "doc-1", sample_markdown(), SAMPLE_META, version=3, status=Status.SUPERSEDED
    )
    assert all(c.version == 3 for c in chunks)
    assert all(c.status is Status.SUPERSEDED for c in chunks)


def test_rows_exclude_generated_columns():
    """as_row() feeds the INSERT; including a generated column is an error."""
    row = chunk_document("doc-1", sample_markdown(), SAMPLE_META)[0].as_row()
    assert "active" not in row
    assert "search_vector" not in row
