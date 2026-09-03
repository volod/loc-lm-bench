"""Build a complete store metadata record for a test that fakes one on disk.

`store_meta.json` is a registered artifact contract, so a test that writes a stub with only the
one key it cares about is writing something that is not a store meta at all. This helper states
the whole record and lets the test override just the part it is exercising.
"""

import json
from pathlib import Path
from typing import Any

STORE_META_RECORD: dict[str, Any] = {
    "schema_id": "llb.rag-store-meta",
    "schema_version": "1.0.0",
    "mode": "flat",
    "strategy": "recursive",
    "size": 800,
    "overlap": 120,
    "child_size": 400,
    "embedding_model": "fake-embedder",
    "n_indexed": 0,
    "n_parents": 0,
    "dim": 4,
    "backend": "faiss",
    "collapse_duplicates": True,
    "duplicate_tier": "exact",
}


def store_meta(**overrides: Any) -> dict[str, Any]:
    """A complete `store_meta.json` record with `overrides` applied."""
    return {**STORE_META_RECORD, **overrides}


def write_store_meta(directory: Path, **overrides: Any) -> Path:
    """Write a complete store meta into `directory` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "store_meta.json"
    path.write_text(json.dumps(store_meta(**overrides)), encoding="utf-8")
    return path
