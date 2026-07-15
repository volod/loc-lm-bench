"""Contamination guards for adapter-backed evaluation."""

from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.finetune.dataset import TUNING_SPLIT
from llb.finetune.registry.resolve import find_by_digest
from llb.finetune.adapter_manifest import load_adapter_manifest
from llb.goldset.schema import GoldItem

PROTECTED_SPLITS = frozenset({"calibration", "final"})
PROVENANCE_REGISTRY = "registry"
PROVENANCE_MANIFEST = "manifest"


def validate_adapter_for_eval(
    *,
    adapter_path: Path | str | None,
    items: list[GoldItem],
    model: str,
    judge_model: str | None = None,
    registry: Path | str | None = None,
) -> JsonObject | None:
    """Refuse adapter evaluations that leak protected item ids or self-judge.

    When the adapter is registered, the training provenance is read from the append-only registry
    rather than the adapter directory: an `adapter_manifest.json` sitting beside the weights is
    operator-writable, so a hand-edited manifest could otherwise launder a calibration/final-split
    adapter past this gate. The on-disk manifest is only trusted for an unregistered adapter (a
    freshly trained one, which registers after its first eval).
    """
    if adapter_path is None:
        return None
    manifest = load_adapter_manifest(adapter_path)
    recorded = _recorded_provenance(registry, manifest)
    provenance = PROVENANCE_MANIFEST if recorded is None else PROVENANCE_REGISTRY
    source = recorded or manifest

    _refuse_contaminated_adapter(source, items, provenance)
    if judge_model is not None and judge_model == model:
        raise SystemExit("[run-eval] tuned model is barred from judging its own answers")
    return manifest


def _refuse_contaminated_adapter(
    source: JsonObject, items: list[GoldItem], provenance: str
) -> None:
    """SystemExit when training touched protected item ids or non-tuning splits."""
    dataset_ids = {str(item_id) for item_id in source.get("dataset_item_ids") or []}
    eval_protected = {item.id for item in items if item.split in PROTECTED_SPLITS}
    overlap = sorted(dataset_ids & eval_protected)
    split_counts = source.get("dataset_split_counts") or {}
    poisoned_splits = sorted(split for split in split_counts if split != TUNING_SPLIT)
    if not overlap and not poisoned_splits:
        return
    details = []
    if overlap:
        details.append(f"offending ids: {', '.join(overlap)}")
    if poisoned_splits:
        details.append(f"dataset splits: {', '.join(poisoned_splits)}")
    details.append(f"provenance: {provenance}")
    raise SystemExit(
        "[run-eval] adapter contamination guard refused this run; " + "; ".join(details)
    )


def _recorded_provenance(registry: Path | str | None, manifest: JsonObject) -> JsonObject | None:
    """The registry's record of this adapter digest, or None when it was never registered."""
    digest = manifest.get("adapter_digest")
    if registry is None or not digest:
        return None
    entry = find_by_digest(registry, str(digest))
    if entry is None:
        return None
    return {
        "dataset_item_ids": list(entry.dataset_item_ids),
        "dataset_split_counts": dict(entry.dataset_split_counts),
    }
