"""Tests for adapter registry gc helpers."""

import json
from pathlib import Path
from llb.finetune.registry.io import registry_path
from llb.finetune.registry.model import (
    AdapterEntry,
)
from adapter_registry_helpers import _entry, _register_event, _trained_adapter


def _store_meta(tmp_path: Path, *, name: str = "rag", **overrides) -> Path:
    """A minimal RAG store dir holding only the store_meta.json the fingerprint reads."""
    meta = {
        "mode": "flat",
        "strategy": "markdown",
        "size": 800,
        "overlap": 120,
        "embedding_model": "intfloat/multilingual-e5-base",
    }
    meta.update(overrides)
    index_dir = tmp_path / name
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "store_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return index_dir


def _superseded_pair(tmp_path: Path) -> tuple[Path, Path, AdapterEntry, AdapterEntry]:
    """Two registered adapters for the same base model; the first is superseded."""
    registry = registry_path(tmp_path)
    old = _trained_adapter(tmp_path, seed=1)
    new = _trained_adapter(tmp_path, seed=2)
    old_entry = _entry(old, created_at="2026-01-01T00:00:00Z")
    new_entry = _entry(new, created_at="2026-02-01T00:00:00Z")
    _register_event(registry, old_entry)
    _register_event(registry, new_entry)
    return old, new, old_entry, new_entry
