"""Resumable campaign journal: append entries, rewrite state, and read completed entries back."""

import json
from pathlib import Path

from llb.core.contracts import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.campaign.coerce import (
    _ci_from_value,
    _float_or_none,
    _path_from,
    _str_or_none,
)
from llb.finetune.campaign.model import (
    PROGRESS_FILENAME,
    SHARED_DATASET_DIRNAME,
    CampaignEntry,
)
from llb.finetune.dataset import DATASET_MANIFEST


def _append_entry(root: Path, entry: CampaignEntry) -> None:
    existing = list(_read_completed_entries(root).values())
    path = root / PROGRESS_FILENAME
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"event": "entry", "entry": entry.as_dict()}, ensure_ascii=False) + "\n"
        )
    _write_state(root, [*existing, entry])


def _write_state(root: Path, entries: list[CampaignEntry]) -> None:
    atomic_write_text(
        root / "campaign_state.json",
        json.dumps(
            {"entries": [entry.as_dict() for entry in entries]}, ensure_ascii=False, indent=2
        )
        + "\n",
    )


def _read_completed_entries(root: Path) -> dict[str, CampaignEntry]:
    path = root / PROGRESS_FILENAME
    if not path.is_file():
        return {}
    entries: dict[str, CampaignEntry] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = row.get("entry") if isinstance(row, dict) else None
        if isinstance(payload, dict) and payload.get("model"):
            entries[str(payload["model"])] = _entry_from_dict(payload)
    return entries


def _entry_from_dict(row: JsonObject) -> CampaignEntry:
    return CampaignEntry(
        model=str(row["model"]),
        status=str(row["status"]),
        reason=str(row["reason"]) if row.get("reason") is not None else None,
        base_final_run_dir=_path_from(row.get("base_final_run_dir")),
        tuning_run_dir=_path_from(row.get("tuning_run_dir")),
        final_run_dir=_path_from(row.get("final_run_dir")),
        adapter_dir=_path_from(row.get("adapter_dir")),
        preference_dataset_dir=_path_from(row.get("preference_dataset_dir")),
        shared_dataset_digest=_str_or_none(row.get("shared_dataset_digest")),
        base_objective=_float_or_none(row.get("base_objective")),
        tuned_objective=_float_or_none(row.get("tuned_objective")),
        delta=_float_or_none(row.get("delta")),
        base_ci=_ci_from_value(row.get("base_ci")),
        tuned_ci=_ci_from_value(row.get("tuned_ci")),
        train_wall_clock_s=_float_or_none(row.get("train_wall_clock_s")),
        peak_vram_mb=_float_or_none(row.get("peak_vram_mb")),
        planner=dict(row.get("planner") or {}),
        reclaim=dict(row.get("reclaim") or {}),
    )


def _existing_shared_dataset(root: Path) -> Path | None:
    path = root / SHARED_DATASET_DIRNAME
    return path if (path / DATASET_MANIFEST).is_file() else None
