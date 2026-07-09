"""Contamination guards for adapter-backed evaluation."""

from pathlib import Path

from llb.core.contracts import JsonObject
from llb.finetune.dataset import TUNING_SPLIT
from llb.finetune.trainer import load_adapter_manifest
from llb.goldset.schema import GoldItem

PROTECTED_SPLITS = frozenset({"calibration", "final"})


def validate_adapter_for_eval(
    *,
    adapter_path: Path | str | None,
    items: list[GoldItem],
    model: str,
    judge_model: str | None = None,
) -> JsonObject | None:
    """Refuse adapter evaluations that leak protected item ids or self-judge."""
    if adapter_path is None:
        return None
    manifest = load_adapter_manifest(adapter_path)
    dataset_ids = {str(item_id) for item_id in manifest.get("dataset_item_ids") or []}
    eval_protected = {item.id for item in items if item.split in PROTECTED_SPLITS}
    overlap = sorted(dataset_ids & eval_protected)
    split_counts = manifest.get("dataset_split_counts") or {}
    poisoned_splits = sorted(split for split in split_counts if split != TUNING_SPLIT)
    if overlap or poisoned_splits:
        details = []
        if overlap:
            details.append(f"offending ids: {', '.join(overlap)}")
        if poisoned_splits:
            details.append(f"dataset splits: {', '.join(poisoned_splits)}")
        raise SystemExit(
            "[run-eval] adapter contamination guard refused this run; " + "; ".join(details)
        )
    if judge_model is not None and judge_model == model:
        raise SystemExit("[run-eval] tuned model is barred from judging its own answers")
    return manifest
