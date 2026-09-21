"""Cut cleaned documents into roughly paragraph-sized chunks.

Chunks are the unit of search AND citation, so each chunk carries:
source title, URL, financial year, heading path, and document version.
Never split a table or list mid-structure.

Output is `schema.ChunkRecord`, not a chunker-local type. The schema module is
the single definition of what a chunk is (see its docstring), and a second
loosely-typed representation here would be one more thing that can drift from
the table.

Three rules decide where a chunk ends:

1. **A chunk never spans a heading**, with one exception. Each chunk belongs to
   exactly one heading path, so a citation names one section rather than
   straddling two - worth more than perfectly even chunk sizes, because a chunk
   is what the answer cites. The exception is a section too short to embed on
   its own (see MIN_CHARS): it is folded into the following chunk when the two
   are siblings or parent-and-child, and the merged chunk is then labelled with
   the heading path common to both.
2. **Structures stay whole.** A table or list is emitted intact even when it
   busts the size target - a half table is worse than a large chunk, and 20% of
   this corpus contains tables.
3. **Within a section, blocks are packed** up to TARGET_CHARS, overflowing into
   further chunks that keep the same heading path.

Sizes are in characters rather than tokens on purpose: the chunker is
stdlib-only so it can be tested without a tokeniser or a model, and for English
prose the ratio is stable enough that a character budget lands in the intended
token range.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence

from fylit_rag.indexing.schema import ChunkRecord, Status, parse_last_updated

# Aim for chunks around this size; retrieval degrades when chunks carry several
# unrelated ideas, and citation gets vague when they are much larger.
TARGET_CHARS = 1200

# A paragraph run longer than this is split on sentence boundaries. Tables and
# lists are exempt - rule 2 above.
MAX_CHARS = 2000

# Below this a chunk is mostly heading and carries little to embed. Such a
# section is merged forward into the next chunk *only* when that next chunk sits
# under the same parent heading, so the merge never crosses a topic boundary.
MIN_CHARS = 200

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TABLE_ROW = re.compile(r"^\s*\|")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_QUESTION_HEADING = re.compile(
    r"^(?:question\s*:|what|when|where|which|who|why|how|can|could|do|does|is|are|will|should)\b.*(?:\?|$)",
    re.IGNORECASE,
)


def _block_kind(line: str) -> str:
    if _HEADING.match(line):
        return "heading"
    if _TABLE_ROW.match(line):
        return "table"
    if _LIST_ITEM.match(line):
        return "list"
    return "para"


def _blocks(text: str) -> Iterator[tuple[str, str]]:
    """Split markdown into (kind, text) blocks: heading, table, list, para.

    Consecutive table rows form one block, as do consecutive list items and
    their indented continuation lines, so neither can be split later.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        kind = _block_kind(line)
        if kind == "heading":
            yield "heading", line.rstrip()
            i += 1
            continue

        buf = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                # A blank line ends a paragraph, but not a list: ATO lists are
                # routinely written with blank lines between items.
                if kind != "list":
                    break
                after = i + 1
                while after < len(lines) and not lines[after].strip():
                    after += 1
                if after >= len(lines) or _block_kind(lines[after]) != "list":
                    break
                buf.append("")
                i = after
                continue
            nxt_kind = _block_kind(nxt)
            if nxt_kind == "heading" or (nxt_kind != kind and not nxt.startswith(("  ", "\t"))):
                break
            buf.append(nxt.rstrip())
            i += 1
        yield kind, "\n".join(buf).strip()


def _split_paragraph(text: str, limit: int) -> list[str]:
    """Break an over-long paragraph on sentence boundaries, never mid-sentence."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + len(sentence) + 1 > limit:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        parts.append(current.strip())
    return parts or [text]


def _sections(text: str) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (heading_path, body_blocks) for each heading-delimited section.

    Content before the first heading is yielded with an empty path, so a
    document that opens with prose loses nothing.
    """
    path: list[tuple[int, str]] = []
    body: list[str] = []
    started = False

    for kind, block in _blocks(text):
        if kind == "heading":
            if started or body:
                yield [h for _, h in path], body
            match = _HEADING.match(block)
            level, title = len(match.group(1)), match.group(2)
            path = [(lvl, h) for lvl, h in path if lvl < level]
            path.append((level, title))
            body = []
            started = True
        else:
            body.append(block)

    if started or body:
        yield [h for _, h in path], body


def _pack(blocks: Sequence[str]) -> list[str]:
    """Pack blocks into TARGET_CHARS-sized texts, never splitting a structure."""
    out: list[str] = []
    current: list[str] = []
    size = 0

    for block in blocks:
        pieces = [block] if len(block) <= MAX_CHARS else _split_paragraph(block, TARGET_CHARS)
        for piece in pieces:
            if current and size + len(piece) + 2 > TARGET_CHARS:
                out.append("\n\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 2

    if current:
        out.append("\n\n".join(current))
    return out


def chunk_text(cleaned_text: str) -> list[tuple[list[str], str]]:
    """Cut markdown into (heading_path, text) pairs. No metadata, no I/O.

    Split out from `chunk_document` so the boundary rules can be tested against
    plain strings, with no document dict and no schema involved.
    """
    out: list[tuple[list[str], str]] = []
    carried: list[str] = []          # heading-only sections awaiting a body
    last_path: list[str] = []

    for path, body in _sections(cleaned_text):
        last_path = path
        texts = _pack(body)
        heading_line = f"{'#' * len(path)} {path[-1]}" if path else ""

        if not texts:
            # A heading with no body of its own: hold it so it leads the next
            # chunk, rather than emitting a chunk that is only a title.
            if heading_line:
                carried.append(heading_line)
            continue

        for n, text in enumerate(texts):
            # Repeat the section heading in every chunk of the section: both
            # halves of retrieval read `text`, and a chunk that does not say
            # which section it came from is harder to rank and to cite.
            piece = f"{heading_line}\n\n{text}" if heading_line else text
            if n == 0 and carried:
                piece = "\n\n".join([*carried, piece])
                carried = []
            out.append((path, piece))

    if carried:
        # Trailing headings with no body anywhere below them - keep them rather
        # than dropping content that a reader would still see on the page.
        out.append((last_path, "\n\n".join(carried)))

    return out


def _merge_short(pairs: list[tuple[list[str], str]]) -> list[tuple[list[str], str]]:
    """Fold a too-short chunk into the next one when they share a parent heading.

    A section of one sentence is a poor thing to embed on its own, but merging
    across unrelated topics would be worse - so the merge only happens when the
    shorter path is a prefix of, or a sibling under, the next chunk's path.
    """
    merged: list[tuple[list[str], str]] = []
    i = 0
    while i < len(pairs):
        path, text = pairs[i]
        if len(text) < MIN_CHARS and i + 1 < len(pairs):
            nxt_path, nxt_text = pairs[i + 1]
            siblings = path[:-1] == nxt_path[:-1]
            ancestor = path == nxt_path[: len(path)]
            if (siblings or ancestor) and len(text) + len(nxt_text) <= MAX_CHARS:
                # Label the merged chunk with the deepest heading path common to
                # both, so the citation names a section that genuinely contains
                # all of the text rather than only its first half.
                common = [a for a, b in zip(path, nxt_path, strict=False) if a == b]
                merged.append((common, f"{text}\n\n{nxt_text}"))
                i += 2
                continue
        merged.append((path, text))
        i += 1
    return merged


def _embedding_text(path: list[str], text: str) -> str:
    """Use the answer body as the semantic representation of an FAQ chunk."""
    if not path or not _QUESTION_HEADING.match(path[-1].strip()):
        return text
    heading = path[-1].strip()
    body = "\n".join(
        line for line in text.splitlines()
        if line.lstrip("#").strip() != heading
    ).strip()
    return body or text


def chunk_document(
    doc_id: str,
    cleaned_text: str,
    metadata: dict,
    *,
    version: int = 1,
    status: Status = Status.ACTIVE,
) -> list[ChunkRecord]:
    """Cut one cleaned document into indexable chunks.

    `metadata` is an enriched document as `ingestion.pipeline` emits it. The
    financial-year column preserves all values from `financial_years` and also
    includes `primary_financial_year` when that stronger inferred value is not
    already present. This preserves multi-year applicability without dropping a
    known primary year.

    `chunk_id` is `doc_id#ordinal`, so re-ingesting an unchanged document
    overwrites its rows instead of duplicating them. `content_hash` is the hash
    of the chunk's own text, which is what lets an upsert tell that a chunk
    actually changed and its embedding must be recomputed.
    """
    pairs = _merge_short(chunk_text(cleaned_text))

    last_updated = parse_last_updated(metadata.get("last_updated_display"))
    financial_year = list(metadata.get("financial_years") or [])

    primary_financial_year = metadata.get(
        "primary_financial_year"
    )

    if (
        primary_financial_year
        and primary_financial_year not in financial_year
    ):
        financial_year.append(
            primary_financial_year
        )

    return [
        ChunkRecord(
            chunk_id=f"{doc_id}#{ordinal}",
            doc_id=doc_id,
            chunk_ordinal=ordinal,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            text=text,
            embedding_text=_embedding_text(path, text),
            heading_path=list(path),
            source_title=metadata.get("title") or "",
            source_url=metadata.get("source_url") or "",
            category=metadata.get("category"),
            topic=metadata.get("topic"),
            last_updated=last_updated,
            financial_year=financial_year,
            version=version,
            status=status,
        )
        for ordinal, (path, text) in enumerate(pairs)
    ]
