"""Ingestion tests: validation, stable IDs/hashes, chunk integrity
(tables and lists never split), incremental diff (new/changed/deleted/superseded)."""

import pytest


@pytest.mark.skip(reason="stub - implement with ingestion.loader")
def test_same_file_never_processed_twice():
    ...
