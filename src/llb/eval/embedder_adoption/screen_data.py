"""Recover item-level embedder deltas from persisted run-eval bundles."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from llb.eval.embedder_adoption.compare import with_reciprocal_rank
from llb.eval.embedder_adoption.models import (
    METRIC_OBJECTIVE,
    METRIC_RECIPROCAL_RANK,
    AdoptionBarReport,
    ItemDeltas,
)


def cell_item_deltas(report: AdoptionBarReport, cell_label: str) -> ItemDeltas:
    """Re-derive one cell's per-item deltas from its run bundles."""
    cell = next((entry for entry in report["cells"] if entry["label"] == cell_label), None)
    if cell is None:
        raise ValueError(
            f"cell {cell_label!r} is not in this sweep "
            f"({', '.join(entry['label'] for entry in report['cells'])})"
        )
    baseline, candidate = report["baseline"], report["candidate"]
    rows = {
        model: _bundle_rows(cell["lanes"][model]["run_dirs"]) for model in (baseline, candidate)
    }
    if set(rows[baseline]) != set(rows[candidate]):
        missing = sorted(set(rows[baseline]) - set(rows[candidate]))
        extra = sorted(set(rows[candidate]) - set(rows[baseline]))
        raise ValueError(
            f"cell {cell_label!r}: the encoders scored different item sets -- "
            f"{candidate} did not score {missing[:3]}, {baseline} did not score {extra[:3]}"
        )
    ids = sorted(rows[baseline])
    if not ids:
        raise ValueError(f"cell {cell_label!r}: the bundles carry no scored item")
    return ItemDeltas(
        item_ids=ids,
        objective=[
            _value(rows[candidate][item_id], METRIC_OBJECTIVE)
            - _value(rows[baseline][item_id], METRIC_OBJECTIVE)
            for item_id in ids
        ],
        reciprocal_rank=[
            _value(rows[candidate][item_id], METRIC_RECIPROCAL_RANK)
            - _value(rows[baseline][item_id], METRIC_RECIPROCAL_RANK)
            for item_id in ids
        ],
    )


def _value(row: Mapping[str, object], metric: str) -> float:
    return float(row.get(metric) or 0.0)  # type: ignore[arg-type]


def _bundle_rows(run_dirs: Sequence[str]) -> dict[str, Mapping[str, object]]:
    from llb.board.io import read_case_rows

    rows: dict[str, Mapping[str, object]] = {}
    for run_dir in run_dirs:
        scores = Path(run_dir) / "scores.jsonl"
        if not scores.is_file():
            raise ValueError(f"missing run bundle scores: {scores}")
        for row in with_reciprocal_rank(list(read_case_rows(scores))):
            rows[str(row["item_id"])] = row
    return rows
