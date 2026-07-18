"""Artifact writers for joint-search runs (ledger + scoreboard)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from llb.optimize.joint_search.constants import (
    JOINT_SEARCH_METHOD,
    LEDGER_FILE,
    MANIFEST_FILE,
    SCOREBOARD_JSON,
    SCOREBOARD_MD,
)
from llb.optimize.joint_search.halving import HalvingLedger
from llb.optimize.tuning_space import FINAL_SPLIT


def joint_run_dir(data_dir: Path, run_id: str | None = None) -> Path:
    """``$DATA_DIR/joint-search/<run_id>/`` artifact root."""
    stamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = data_dir / JOINT_SEARCH_METHOD / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(run_dir: Path, payload: dict[str, Any]) -> Path:
    """Write the run manifest JSON."""
    path = run_dir / MANIFEST_FILE
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_ledger(run_dir: Path, ledger: HalvingLedger) -> Path:
    """Persist the successive-halving ledger (tuning-split eliminations only)."""
    path = run_dir / LEDGER_FILE
    path.write_text(json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def assert_final_split(entry: dict[str, Any]) -> None:
    """Refuse scoreboard rows that claim a non-final split (tuning/final leak fence)."""
    split = entry.get("split")
    if split != FINAL_SPLIT:
        raise ValueError(
            f"scoreboard entry for {entry.get('model')!r} must use split={FINAL_SPLIT!r}; "
            f"got {split!r}"
        )


def write_scoreboard(
    run_dir: Path,
    *,
    run_id: str,
    entries: Sequence[dict[str, Any]],
    recommended: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write scoreboard JSON + Markdown; every entry must be final-split only."""
    for entry in entries:
        assert_final_split(entry)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "split": FINAL_SPLIT,
        "entries": list(entries),
        "recommended": recommended,
    }
    json_path = run_dir / SCOREBOARD_JSON
    md_path = run_dir / SCOREBOARD_MD
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_scoreboard_md(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _render_scoreboard_md(payload: dict[str, Any]) -> str:
    lines = [
        f"# Joint search scoreboard: {payload['run_id']}",
        "",
        f"Split: `{payload['split']}` (final-split runs only; no tuning leakage)",
        "",
        "| model | backend | pick | quality | overrides |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in payload["entries"]:
        overrides = entry.get("overrides") or {}
        override_text = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        quality = entry.get("quality")
        quality_text = f"{quality:.4f}" if isinstance(quality, (int, float)) else "-"
        lines.append(
            f"| {entry.get('model', '-')} | {entry.get('backend', '-')} | "
            f"{entry.get('pick', '-')} | {quality_text} | `{override_text}` |"
        )
    recommended = payload.get("recommended")
    if recommended:
        lines.extend(
            [
                "",
                "## Recommended",
                "",
                (f"- model: `{recommended.get('model')}` ({recommended.get('backend')})"),
                f"- pick: `{recommended.get('pick')}`",
                f"- quality: {recommended.get('quality')}",
                f"- overrides: `{recommended.get('overrides')}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)
