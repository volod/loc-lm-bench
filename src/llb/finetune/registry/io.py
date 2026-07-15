"""Append-only adapter event-log paths, writes, and folding."""

import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.finetune.registry.model import (
    ADAPTERS_METHOD,
    EVENT_DELETE,
    EVENT_MERGE,
    EVENT_REGISTER,
    MERGED_DIRNAME,
    REGISTRY_FILENAME,
    TIMESTAMP_FORMAT,
    AdapterEntry,
)

_LOG = logging.getLogger(__name__)


def registry_path(data_dir: Path | str) -> Path:
    """Return the append-only adapter event-log path."""
    return Path(data_dir) / ADAPTERS_METHOD / REGISTRY_FILENAME


def merged_root(data_dir: Path | str) -> Path:
    """Return the merge-output root for non-LoRA serving backends."""
    return Path(data_dir) / ADAPTERS_METHOD / MERGED_DIRNAME


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def append_event(registry: Path | str, payload: JsonObject) -> None:
    """Append one event without rewriting any earlier registry history."""
    path = Path(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_events(registry: Path | str) -> list[JsonObject]:
    """Read events in order, skipping malformed lines so one bad append is isolated."""
    path = Path(registry)
    if not path.is_file():
        return []
    events: list[JsonObject] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            _LOG.warning("[adapters] skipping malformed registry event at %s:%d", path, line_no)
            continue
        if isinstance(row, dict):
            events.append(row)
    return events


def load_registry(registry: Path | str) -> dict[str, AdapterEntry]:
    """Fold the event log into the current entry set, keyed by adapter id."""
    entries: dict[str, AdapterEntry] = {}
    registrations = 0
    for event in read_events(registry):
        kind = str(event.get("event") or "")
        adapter_id = str(event.get("adapter_id") or "")
        if not adapter_id:
            continue
        if kind == EVENT_REGISTER:
            registrations += 1
            entries[adapter_id] = _entry_from_event(event, sequence=registrations)
        elif kind == EVENT_MERGE:
            current = entries.get(adapter_id)
            if current is not None:
                entries[adapter_id] = replace(current, merges=(*current.merges, _merge_of(event)))
        elif kind == EVENT_DELETE:
            entries.pop(adapter_id, None)
    return entries


def _entry_from_event(event: JsonObject, *, sequence: int = 0) -> AdapterEntry:
    return AdapterEntry(
        sequence=sequence,
        adapter_id=str(event["adapter_id"]),
        base_model=str(event.get("base_model") or ""),
        adapter_label=str(event.get("adapter_label") or ""),
        adapter_dir=Path(str(event.get("adapter_dir") or "")),
        dataset_digest=str(event.get("dataset_digest") or ""),
        dataset_item_ids=tuple(str(item) for item in event.get("dataset_item_ids") or []),
        dataset_split_counts=_split_counts(event.get("dataset_split_counts")),
        goldset_digest=_str_or_none(event.get("goldset_digest")),
        corpus_digest=_str_or_none(event.get("corpus_digest")),
        goldset_path=_str_or_none(event.get("goldset_path")),
        corpus_root=_str_or_none(event.get("corpus_root")),
        retrieval_fingerprint=(
            dict(event["retrieval_fingerprint"])
            if isinstance(event.get("retrieval_fingerprint"), dict)
            else None
        ),
        index_dir=_str_or_none(event.get("index_dir")),
        source_run=_str_or_none(event.get("source_run")),
        eval_summary=dict(event.get("eval") or {}),
        created_at=str(event.get("created_at") or ""),
    )


def _merge_of(event: JsonObject) -> JsonObject:
    return {key: value for key, value in event.items() if key not in {"event", "adapter_id"}}


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


def _str_or_none(value: object) -> str | None:
    return None if value is None else str(value)
