"""Tests for miss probe helpers."""

import json
from pathlib import Path
from llb.board.miss_analysis.model import (
    MissRecord,
)
from llb.goldset.schema import GoldItem
from miss_analysis_helpers import _all_class_rows, _goldset, _score_row, _write_bundle


def _probe_manifest(tmp_path: Path) -> dict:
    run_dir = _write_bundle(tmp_path, *_all_class_rows())
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _probe_subset(analysis_misses: list[MissRecord]) -> list[GoldItem]:
    items_by_id = {item.id: item for item in _goldset()}
    return sorted((items_by_id[m.item_id] for m in analysis_misses), key=lambda item: item.id)


def _write_probe_bundle(tmp_path: Path, name: str, run_name: str, subset: list[GoldItem]) -> Path:
    """A finalized probe bundle where every case hit retrieval and scored well."""
    run_dir = tmp_path / "run-eval" / name
    run_dir.mkdir(parents=True)
    rows = [_score_row(item.id, "ok", 0.8, 1.0) for item in subset]
    manifest = {
        "run_id": "probe" + name[-4:],
        "run_name": run_name,
        "split": "final",
        "n_cases": len(rows),
        "config": {},
        "metrics": {"objective_score": 0.8},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "scores.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return run_dir
