"""Document versioning and incremental-ingestion helpers.

This module provides the state-management layer used by Zone A ingestion.

Responsibilities
----------------
- detect duplicate cleaned content without deleting documents
- load the previous document-id -> content-hash state
- compare a current corpus with the previous state
- identify new, changed, unchanged and deleted documents
- maintain an in-memory manifest representation
- distinguish active, superseded and deleted document versions

The PostgreSQL chunk schema remains the authoritative indexed representation.
This module prepares document-level change information for that indexing layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# These values intentionally match indexing.schema.Status.
STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
STATUS_DELETED = "deleted"

VALID_STATUSES = {
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    STATUS_DELETED,
}


# --------------------------------------------------------------------------- #
# Generic document access
# --------------------------------------------------------------------------- #


def _document_value(
    document: Any,
    *names: str,
) -> Any:
    """Read a field from either a mapping or document object.

    Preprocessing currently works with both:

    - ``CleanedDoc.id``
    - ``RawDocument.doc_id``

    Supporting both here prevents the versioning layer from being coupled to
    one particular ingestion-stage dataclass.
    """

    if isinstance(document, Mapping):
        for name in names:
            if name in document:
                return document[name]

    else:
        for name in names:
            if hasattr(document, name):
                return getattr(
                    document,
                    name,
                )

    expected = " or ".join(
        repr(name)
        for name in names
    )

    raise AttributeError(
        "Document is missing required field "
        f"{expected}"
    )


def document_id(
    document: Any,
) -> str:
    """Return the stable identifier from a document."""

    value = _document_value(
        document,
        "id",
        "doc_id",
    )

    if not isinstance(
        value,
        str,
    ) or not value:
        raise ValueError(
            "Document identifier must be "
            "a non-empty string"
        )

    return value


def document_content_hash(
    document: Any,
) -> str:
    """Return the content hash from a document."""

    value = _document_value(
        document,
        "content_hash",
    )

    if not isinstance(
        value,
        str,
    ) or not value:
        raise ValueError(
            "Document content_hash must be "
            "a non-empty string"
        )

    return value


# --------------------------------------------------------------------------- #
# Duplicate-content detection
# --------------------------------------------------------------------------- #


def find_content_duplicates(
    docs: Iterable[Any],
) -> dict[str, list[str]]:
    """Group documents that have identical content hashes.

    Only groups containing more than one document are returned.

    Documents are deliberately NOT removed. Different source URLs,
    historical versions, or related ATO pages may legitimately contain
    identical content.
    """

    by_hash: dict[
        str,
        list[str],
    ] = {}

    for doc in docs:
        hash_value = document_content_hash(
            doc
        )

        doc_id = document_id(
            doc
        )

        by_hash.setdefault(
            hash_value,
            [],
        ).append(
            doc_id
        )

    return {
        hash_value: ids
        for hash_value, ids
        in by_hash.items()
        if len(ids) > 1
    }


# --------------------------------------------------------------------------- #
# Legacy preprocessing state
# --------------------------------------------------------------------------- #


def load_state(
    state_path: Path,
) -> dict[str, str]:
    """Load the previous document-id -> content-hash state.

    A missing or malformed state file is treated as a first ingestion run.

    This preserves the behaviour of the validated Phase-1 preprocessing
    script while allowing the richer ``Manifest`` abstraction to be used by
    later indexing stages.
    """

    state_path = Path(
        state_path
    )

    if not state_path.exists():
        return {}

    try:
        with state_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

    except (
        json.JSONDecodeError,
        OSError,
    ):
        log.warning(
            "Could not read previous state.json - "
            "treating this as a first run"
        )

        return {}

    if not isinstance(
        data,
        dict,
    ):
        log.warning(
            "Previous state.json is not a JSON object - "
            "treating this as a first run"
        )

        return {}

    return {
        str(doc_id): str(hash_value)
        for doc_id, hash_value
        in data.items()
    }


def diff_against_state(
    docs: Iterable[Any],
    prev_state: Mapping[str, str],
) -> dict:
    """Compare the current corpus against a previous content-hash state.

    Returns the same validated Phase-1 structure used by ``datacleaning.py``:

    ``new``
        Number of newly discovered document IDs.

    ``changed``
        Number of existing document IDs whose content hash changed.

    ``unchanged``
        Number of existing document IDs with the same content hash.

    ``deleted``
        Number of IDs that existed previously but are absent now.

    The new, changed and deleted IDs are also returned so downstream indexing
    can process only the affected documents.
    """

    current_ids = {
        document_id(doc):
            document_content_hash(doc)
        for doc in docs
    }

    new_ids = [
        doc_id
        for doc_id in current_ids
        if doc_id not in prev_state
    ]

    changed_ids = [
        doc_id
        for doc_id in current_ids
        if (
            doc_id in prev_state
            and prev_state[doc_id]
            != current_ids[doc_id]
        )
    ]

    unchanged_ids = [
        doc_id
        for doc_id in current_ids
        if (
            doc_id in prev_state
            and prev_state[doc_id]
            == current_ids[doc_id]
        )
    ]

    deleted_ids = [
        doc_id
        for doc_id in prev_state
        if doc_id not in current_ids
    ]

    return {
        "new":
            len(new_ids),

        "changed":
            len(changed_ids),

        "unchanged":
            len(unchanged_ids),

        "deleted":
            len(deleted_ids),

        "new_ids":
            new_ids,

        "changed_ids":
            changed_ids,

        "deleted_ids":
            deleted_ids,
    }


# --------------------------------------------------------------------------- #
# Manifest representation
# --------------------------------------------------------------------------- #


def _now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(
        UTC
    ).isoformat()


@dataclass
class ManifestEntry:
    """One document-level versioning entry."""

    doc_id: str
    content_hash: str
    version: int = 1
    status: str = STATUS_ACTIVE
    superseded_by: str | None = None
    indexed_at: str | None = None

    def __post_init__(
        self,
    ) -> None:
        if self.version < 1:
            raise ValueError(
                "Manifest version must be >= 1"
            )

        if self.status not in VALID_STATUSES:
            raise ValueError(
                "Invalid manifest status: "
                f"{self.status}"
            )

        if self.indexed_at is None:
            self.indexed_at = _now()

    def as_dict(
        self,
    ) -> dict:
        """Return a serialisable manifest record."""

        return {
            "doc_id":
                self.doc_id,

            "content_hash":
                self.content_hash,

            "version":
                self.version,

            "status":
                self.status,

            "superseded_by":
                self.superseded_by,

            "indexed_at":
                self.indexed_at,
        }


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


class Manifest:
    """In-memory document-version manifest.

    ``Manifest`` complements the PostgreSQL chunk schema rather than replacing
    it. It provides document-level change planning before changed documents are
    chunked and written to the index.

    Parameters
    ----------
    entries:
        Optional existing manifest data.

        Supported forms include:

        ``{"doc-id": "sha256..."}``

        for the Phase-1 legacy state format, or:

        ``{"doc-id": {"content_hash": "...", "version": 2, ...}}``

        for richer manifest records.
    """

    def __init__(
        self,
        entries: Mapping[
            str,
            Any,
        ]
        | None = None,
    ) -> None:
        self.entries: dict[
            str,
            ManifestEntry,
        ] = {}

        if entries:
            self._load_entries(
                entries
            )

    def _load_entries(
        self,
        entries: Mapping[
            str,
            Any,
        ],
    ) -> None:
        """Normalise legacy or rich manifest entries."""

        for doc_id, value in entries.items():

            # Legacy Phase-1 state:
            #
            #     doc_id -> content_hash
            #
            if isinstance(
                value,
                str,
            ):
                self.entries[
                    str(doc_id)
                ] = ManifestEntry(
                    doc_id=str(
                        doc_id
                    ),
                    content_hash=value,
                )

                continue

            if not isinstance(
                value,
                Mapping,
            ):
                raise TypeError(
                    "Manifest entry for "
                    f"{doc_id!r} must be a hash "
                    "string or mapping"
                )

            content_hash = value.get(
                "content_hash"
            )

            if not isinstance(
                content_hash,
                str,
            ) or not content_hash:
                raise ValueError(
                    "Manifest entry for "
                    f"{doc_id!r} has no valid "
                    "content_hash"
                )

            self.entries[
                str(doc_id)
            ] = ManifestEntry(
                doc_id=str(
                    doc_id
                ),
                content_hash=content_hash,
                version=int(
                    value.get(
                        "version",
                        1,
                    )
                ),
                status=str(
                    value.get(
                        "status",
                        STATUS_ACTIVE,
                    )
                ),
                superseded_by=value.get(
                    "superseded_by"
                ),
                indexed_at=value.get(
                    "indexed_at"
                ),
            )

    def active_state(
        self,
    ) -> dict[str, str]:
        """Return the current non-deleted document hash state."""

        return {
            doc_id:
                entry.content_hash

            for doc_id, entry
            in self.entries.items()

            if entry.status
            != STATUS_DELETED
        }

    def diff(
        self,
        current_docs: Iterable[Any],
    ) -> dict:
        """Return document IDs requiring incremental action.

        The original stub promised:

            {
                "new": [...],
                "changed": [...],
                "deleted": [...]
            }

        We retain that API and additionally expose ``unchanged``.
        """

        docs = list(
            current_docs
        )

        previous = self.active_state()

        result = diff_against_state(
            docs,
            previous,
        )

        current_ids = [
            document_id(doc)
            for doc in docs
        ]

        new_set = set(
            result[
                "new_ids"
            ]
        )

        changed_set = set(
            result[
                "changed_ids"
            ]
        )

        unchanged_ids = [
            doc_id
            for doc_id in current_ids
            if (
                doc_id not in new_set
                and doc_id
                not in changed_set
            )
        ]

        return {
            "new":
                result[
                    "new_ids"
                ],

            "changed":
                result[
                    "changed_ids"
                ],

            "unchanged":
                unchanged_ids,

            "deleted":
                result[
                    "deleted_ids"
                ],
        }

    def record_active(
        self,
        doc_id: str,
        content_hash: str,
        version: int = 1,
    ) -> ManifestEntry:
        """Insert or replace an active manifest entry."""

        entry = ManifestEntry(
            doc_id=doc_id,
            content_hash=content_hash,
            version=version,
            status=STATUS_ACTIVE,
        )

        self.entries[
            doc_id
        ] = entry

        return entry

    def mark_superseded(
        self,
        doc_id: str,
        by_doc_id: str,
    ) -> None:
        """Mark a document version as superseded.

        Superseded content is intentionally retained rather than deleted so it
        can remain available for historically appropriate retrieval.
        """

        if doc_id not in self.entries:
            raise KeyError(
                "Cannot supersede unknown "
                f"document: {doc_id}"
            )

        entry = self.entries[
            doc_id
        ]

        entry.status = (
            STATUS_SUPERSEDED
        )

        entry.superseded_by = (
            by_doc_id
        )

    def mark_deleted(
        self,
        doc_id: str,
    ) -> None:
        """Mark a document as deleted from the source corpus."""

        if doc_id not in self.entries:
            raise KeyError(
                "Cannot delete unknown "
                f"document: {doc_id}"
            )

        entry = self.entries[
            doc_id
        ]

        entry.status = (
            STATUS_DELETED
        )

        entry.superseded_by = None

    def as_dict(
        self,
    ) -> dict[str, dict]:
        """Return the complete serialisable manifest."""

        return {
            doc_id:
                entry.as_dict()

            for doc_id, entry
            in self.entries.items()
        }


__all__ = [
    "STATUS_ACTIVE",
    "STATUS_DELETED",
    "STATUS_SUPERSEDED",
    "VALID_STATUSES",
    "Manifest",
    "ManifestEntry",
    "diff_against_state",
    "document_content_hash",
    "document_id",
    "find_content_duplicates",
    "load_state",
]