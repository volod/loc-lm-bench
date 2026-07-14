"""Tests for adapter registry roundtrip."""

from pathlib import Path
from llb.finetune.registry.io import load_registry, registry_path
from llb.finetune.registry.register import register_adapter
from llb.finetune.registry.resolve import resolve_adapter
from adapter_registry_helpers import _goldset, _trained_adapter


def test_registry_round_trip_is_idempotent(tmp_path: Path):
    goldset = _goldset(tmp_path)
    registry = registry_path(tmp_path)
    adapter_dir = _trained_adapter(tmp_path)

    entry = register_adapter(
        registry=registry,
        adapter_dir=adapter_dir,
        goldset_path=goldset,
        source_run=tmp_path / "run-tuning",
        eval_summary={"objective_score": 0.75, "delta": 0.1},
    )
    again = register_adapter(
        registry=registry,
        adapter_dir=adapter_dir,
        goldset_path=goldset,
        source_run=tmp_path / "run-tuning",
        eval_summary={"objective_score": 0.75, "delta": 0.1},
    )

    assert again.adapter_id == entry.adapter_id
    assert registry.read_text(encoding="utf-8").count('"event"') == 1
    entries = load_registry(registry)
    loaded = entries[entry.adapter_id]
    assert loaded.base_model == "base-model"
    assert loaded.dataset_split_counts == {"tuning": 1}
    assert loaded.eval_summary["objective_score"] == 0.75
    assert loaded.goldset_digest is not None
    assert resolve_adapter(entries, entry.adapter_id[:8]).adapter_id == entry.adapter_id
    assert resolve_adapter(entries, entry.adapter_label).adapter_id == entry.adapter_id
