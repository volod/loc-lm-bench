"""Before/after retrieval and answer-quality report for a conflict overlay."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llb.core.fsutil import atomic_write_text
from llb.rag.refresh.drift import RetrievalDrift


def objective_from_manifest(path: Path | str | None) -> float | None:
    if path is None:
        return None
    value = Path(path)
    manifest_path = value / "manifest.json" if value.is_dir() else value
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    objective = metrics.get("objective_score") if isinstance(metrics, dict) else None
    if not isinstance(objective, int | float):
        raise ValueError(f"{manifest_path}: metrics.objective_score is missing")
    return float(objective)


def render_effect(
    drift: RetrievalDrift | None,
    *,
    before_objective: float | None = None,
    after_objective: float | None = None,
    overlay_path: Path | str | None = None,
    unresolved_reviews: int = 0,
) -> str:
    lines = [
        "# Corpus conflict resolution effect",
        "",
        f"- applied overlay: `{overlay_path}`" if overlay_path else "- applied overlay: none",
        "",
        "| metric | before | after | delta |",
        "| --- | --- | --- | --- |",
    ]
    if drift is not None:
        lines += [
            f"| recall@{drift.k} | {drift.old_recall:.4f} | {drift.new_recall:.4f} "
            f"| {drift.delta_recall:+.4f} |",
            f"| MRR | {drift.old_mrr:.4f} | {drift.new_mrr:.4f} | {drift.delta_mrr:+.4f} |",
        ]
    else:
        lines += ["| recall@10 | n/a | n/a | n/a |", "| MRR | n/a | n/a | n/a |"]
    if before_objective is not None and after_objective is not None:
        delta = after_objective - before_objective
        lines.append(
            f"| objective | {before_objective:.4f} | {after_objective:.4f} | {delta:+.4f} |"
        )
    else:
        lines.append("| objective | pending run-eval | pending run-eval | n/a |")
    lines += [
        "",
        _verdict(drift, before_objective, after_objective, unresolved_reviews),
        "",
    ]
    return "\n".join(lines)


def _verdict(
    drift: RetrievalDrift | None,
    before_objective: float | None,
    after_objective: float | None,
    unresolved_reviews: int,
) -> str:
    if drift is None:
        return "Verdict: MEASUREMENT REQUIRED before adoption."
    if before_objective is None or after_objective is None:
        return "Verdict: MEASUREMENT REQUIRED before adoption (objective delta is pending)."
    if unresolved_reviews:
        return f"Verdict: REVERT; {unresolved_reviews} review decisions remain unresolved."
    retrieval_regressed = drift.delta_recall < 0 or drift.delta_mrr < 0
    objective_regressed = (
        before_objective is not None
        and after_objective is not None
        and after_objective < before_objective
    )
    return (
        "Verdict: REVERT; at least one measured quality axis regressed."
        if retrieval_regressed or objective_regressed
        else "Verdict: ADOPT; no measured quality axis regressed."
    )


def write_effect(
    path: Path | str, drift: Any, *, effect_key: str | None = None, **kwargs: Any
) -> Path:
    value = Path(path)
    data_path = value.with_suffix(".json")
    if drift is None and data_path.is_file():
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        cached = payload.get("drift") if isinstance(payload, dict) else None
        if (
            isinstance(payload, dict)
            and payload.get("effect_key") == effect_key
            and isinstance(cached, dict)
        ):
            drift = RetrievalDrift(**cached)
    elif drift is not None:
        payload = {"schema_version": 1, "effect_key": effect_key, "drift": asdict(drift)}
        atomic_write_text(data_path, json.dumps(payload, indent=2) + "\n")
    atomic_write_text(value, render_effect(drift, **kwargs))
    return value
