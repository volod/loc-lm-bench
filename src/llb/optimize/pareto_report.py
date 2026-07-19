"""Pareto front report writers for multi-objective RAG tuning."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from llb.optimize.objectives import GoalPick, ParetoPoint

TUNE_METHOD = "tune"


def tune_run_dir(data_dir: Path, run_id: str | None = None) -> Path:
    """Artifact root ``$DATA_DIR/tune/<run_id>/`` for Pareto JSON + Markdown."""
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = data_dir / TUNE_METHOD / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_pareto_report(
    out_dir: Path,
    *,
    study_name: str,
    objectives: Sequence[str],
    front: Sequence[ParetoPoint],
    picks: Sequence[GoalPick],
    n_trials: int,
    n_complete: int,
    n_pruned: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write ``pareto.json`` + ``pareto.md``; return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "study_name": study_name,
        "objectives": list(objectives),
        "n_trials": n_trials,
        "n_complete": n_complete,
        "n_pruned": n_pruned,
        "pareto_front": [p.to_dict() for p in front],
        "picks": [p.to_dict() for p in picks],
    }
    if extra:
        payload.update(extra)
    json_path = out_dir / "pareto.json"
    md_path = out_dir / "pareto.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _render_markdown(payload: dict[str, Any]) -> str:
    objectives = ", ".join(payload["objectives"])
    lines = [
        f"# Multi-objective tune: {payload['study_name']}",
        "",
        f"Objectives: `{objectives}`",
        (
            f"Trials: {payload['n_complete']} complete, {payload['n_pruned']} pruned "
            f"of {payload['n_trials']}"
        ),
        "",
        "## Pareto front",
        "",
        "| trial | quality | latency_s | cost_usd | throughput | overrides |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for point in payload["pareto_front"]:
        overrides = ", ".join(f"{k}={v}" for k, v in sorted(point["overrides"].items()))
        lines.append(
            f"| {point['number']} | {point['quality']:.4f} | {point['latency_s']:.3f} | "
            f"{point['cost_usd']:.4f} | {point['throughput']:.1f} | `{overrides}` |"
        )
    lines.extend(["", "## Per-goal picks", ""])
    for pick in payload["picks"]:
        point = pick["point"]
        lines.append(
            f"- {pick['goal']}: trial {point['number']} "
            f"quality={point['quality']:.4f} latency_s={point['latency_s']:.3f} "
            f"cost_usd={point['cost_usd']:.4f}"
        )
        lines.append(f"  - overrides: `{point['overrides']}`")
    lines.append("")
    return "\n".join(lines)
