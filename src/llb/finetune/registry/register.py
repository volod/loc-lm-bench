"""Adapter registration and lifecycle-event commands."""

import logging
from pathlib import Path

from llb.core.contracts import JsonObject
from llb.finetune.registry.io import append_event, load_registry, utc_now
from llb.finetune.registry.model import EVENT_DELETE, EVENT_MERGE, EVENT_REGISTER, AdapterEntry
from llb.finetune.registry.staleness import (
    corpus_digest_for,
    goldset_digest_for,
    retrieval_fingerprint_for,
)
from llb.finetune.trainer import adapter_label, load_adapter_manifest

_LOG = logging.getLogger(__name__)


def register_adapter(
    *,
    registry: Path | str,
    adapter_dir: Path | str,
    goldset_path: Path | str | None = None,
    corpus_root: Path | str | None = None,
    index_dir: Path | str | None = None,
    source_run: Path | str | None = None,
    eval_summary: JsonObject | None = None,
) -> AdapterEntry:
    """Register a trained adapter and its training-time benchmark provenance."""
    manifest = load_adapter_manifest(adapter_dir)
    entry = _entry_from_manifest(
        manifest,
        adapter_dir=Path(adapter_dir),
        goldset_path=goldset_path,
        corpus_root=corpus_root,
        index_dir=index_dir,
        source_run=source_run,
        eval_summary=eval_summary or {},
    )
    existing = load_registry(registry).get(entry.adapter_id)
    if existing is not None and _identity(existing) == _identity(entry):
        return existing
    append_event(registry, {"event": EVENT_REGISTER, **entry.as_dict()})
    return load_registry(registry)[entry.adapter_id]


def try_register_adapter(
    *,
    registry: Path | str,
    adapter_dir: Path | str,
    goldset_path: Path | str | None = None,
    corpus_root: Path | str | None = None,
    index_dir: Path | str | None = None,
    source_run: Path | str | None = None,
    eval_summary: JsonObject | None = None,
) -> AdapterEntry | None:
    """Register when possible without invalidating an orchestrator's other artifacts."""
    try:
        return register_adapter(
            registry=registry,
            adapter_dir=adapter_dir,
            goldset_path=goldset_path,
            corpus_root=corpus_root,
            index_dir=index_dir,
            source_run=source_run,
            eval_summary=eval_summary,
        )
    except (ValueError, OSError) as exc:
        _LOG.warning("[adapters] not registering %s: %s", adapter_dir, exc)
        return None


def record_merge(
    *, registry: Path | str, adapter_id: str, backend: str, artifacts: JsonObject
) -> JsonObject:
    """Record that an adapter was merged into a servable backend artifact."""
    payload = {
        "event": EVENT_MERGE,
        "adapter_id": adapter_id,
        "backend": backend,
        "created_at": utc_now(),
        **artifacts,
    }
    append_event(registry, payload)
    return payload


def record_delete(*, registry: Path | str, adapter_id: str, reason: str) -> None:
    """Tombstone an adapter while retaining its registration history."""
    append_event(
        registry,
        {
            "event": EVENT_DELETE,
            "adapter_id": adapter_id,
            "reason": reason,
            "created_at": utc_now(),
        },
    )


def _entry_from_manifest(
    manifest: JsonObject,
    *,
    adapter_dir: Path,
    goldset_path: Path | str | None,
    corpus_root: Path | str | None,
    index_dir: Path | str | None,
    source_run: Path | str | None,
    eval_summary: JsonObject,
) -> AdapterEntry:
    adapter_id = str(manifest.get("adapter_digest") or "")
    if not adapter_id:
        raise ValueError(f"adapter manifest has no adapter_digest: {adapter_dir}")
    base_model = str(manifest.get("base_model") or "")
    return AdapterEntry(
        adapter_id=adapter_id,
        base_model=base_model,
        adapter_label=str(manifest.get("adapter_label") or adapter_label(base_model, adapter_id)),
        adapter_dir=adapter_dir.resolve(),
        dataset_digest=str(manifest.get("dataset_digest") or ""),
        dataset_item_ids=tuple(str(item) for item in manifest.get("dataset_item_ids") or []),
        dataset_split_counts=_split_counts(manifest.get("dataset_split_counts")),
        goldset_digest=goldset_digest_for(goldset_path),
        corpus_digest=corpus_digest_for(corpus_root),
        goldset_path=str(goldset_path) if goldset_path is not None else None,
        corpus_root=str(corpus_root) if corpus_root is not None else None,
        retrieval_fingerprint=retrieval_fingerprint_for(index_dir),
        index_dir=str(index_dir) if index_dir is not None else None,
        source_run=str(source_run) if source_run is not None else None,
        eval_summary=dict(eval_summary),
        created_at=utc_now(),
    )


def _identity(entry: AdapterEntry) -> JsonObject:
    payload = entry.as_dict()
    payload.pop("created_at", None)
    return payload


def _split_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count in value.items():
        try:
            counts[str(key)] = int(count)
        except (TypeError, ValueError):
            continue
    return counts
