"""Append-only registry for locally trained LoRA adapters.

An adapter is content-addressed by its training digest (`adapter_digest`), so the registry id is
derived from the base model, dataset digest, seed, and hyperparameters -- never assigned. The log
is an append-only JSONL event stream (`register` / `merge` / `delete`) folded into the current
entry set on read, so a crash mid-write can never corrupt earlier history and every adapter number
stays traceable back to the run that produced it.

Entries record the goldset and corpus digests observed AT TRAINING TIME. Comparing them against
the present benchmark is how staleness is detected: when the goldset or corpus changed, the
recorded eval evidence no longer describes the current benchmark, so the adapter is stamped stale
rather than silently trusted.
"""

import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from llb.core.contracts import JsonObject
from llb.core.paths import resolve_project_path
from llb.finetune.trainer import (
    ADAPTER_DIGEST_SHORT_CHARS,
    adapter_label,
    load_adapter_manifest,
)

_LOG = logging.getLogger(__name__)

ADAPTERS_METHOD = "adapters"
REGISTRY_FILENAME = "registry.jsonl"
MERGED_DIRNAME = "merged"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

EVENT_REGISTER = "register"
EVENT_MERGE = "merge"
EVENT_DELETE = "delete"

# `list-adapters` verdicts. `unknown` means the comparison could not run (a digest was never
# recorded, or the goldset/corpus it names is gone), which is NOT the same as `current`.
VERDICT_CURRENT = "current"
VERDICT_STALE = "stale"
VERDICT_UNKNOWN = "unknown"

# The retrieval knobs recorded from `store_meta.json` at registration and compared by
# `staleness`. An adapter is trained on retrieved CONTEXT: re-embedding or rechunking the same
# corpus leaves `corpus_fingerprint` unchanged while the training contexts no longer exist, so
# these knobs form a third staleness axis beside the goldset and corpus digests.
RETRIEVAL_FINGERPRINT_KEYS = (
    ("embedding_model", "embedding_model"),
    ("strategy", "strategy"),
    ("chunk_size", "size"),
    ("chunk_overlap", "overlap"),
    ("retrieval_mode", "mode"),
)

# Shortest `--adapter` prefix accepted; below this a sha256 prefix is not discriminating enough.
MIN_ADAPTER_ID_PREFIX = 6


@dataclass(frozen=True)
class AdapterEntry:
    """One registered adapter: what trained it, what it scored, and where it lives."""

    adapter_id: str
    base_model: str
    adapter_label: str
    adapter_dir: Path
    dataset_digest: str
    dataset_item_ids: tuple[str, ...] = ()
    dataset_split_counts: dict[str, int] = field(default_factory=dict)
    goldset_digest: str | None = None
    corpus_digest: str | None = None
    goldset_path: str | None = None
    corpus_root: str | None = None
    # RAG-store retrieval knobs at training time (adapter-staleness-retrieval-fingerprint);
    # additive -- entries registered before the field carry None and read `unknown` on that axis.
    retrieval_fingerprint: JsonObject | None = None
    index_dir: str | None = None
    source_run: str | None = None
    eval_summary: JsonObject = field(default_factory=dict)
    created_at: str = ""
    merges: tuple[JsonObject, ...] = ()
    # Position of this adapter's `register` event in the append-only log. `created_at` has
    # second resolution, so two adapters trained in the same second tie; the log order does not.
    # Only entries loaded via `load_registry` carry a meaningful sequence.
    sequence: int = 0

    @property
    def short_id(self) -> str:
        return self.adapter_id[:ADAPTER_DIGEST_SHORT_CHARS]

    @property
    def recency(self) -> tuple[str, int]:
        """Sort key for "which adapter is newest", exact under same-second registrations."""
        return (self.created_at, self.sequence)

    @property
    def resolved_dir(self) -> Path:
        """Absolute adapter directory (committed fixtures record project-relative paths)."""
        return resolve_project_path(self.adapter_dir)

    def as_dict(self) -> JsonObject:
        """The `register` event payload (merge history is carried by its own events)."""
        return {
            "adapter_id": self.adapter_id,
            "base_model": self.base_model,
            "adapter_label": self.adapter_label,
            "adapter_dir": str(self.adapter_dir),
            "dataset_digest": self.dataset_digest,
            "dataset_item_ids": list(self.dataset_item_ids),
            "dataset_split_counts": dict(self.dataset_split_counts),
            "goldset_digest": self.goldset_digest,
            "corpus_digest": self.corpus_digest,
            "goldset_path": self.goldset_path,
            "corpus_root": self.corpus_root,
            "retrieval_fingerprint": (
                dict(self.retrieval_fingerprint) if self.retrieval_fingerprint is not None else None
            ),
            "index_dir": self.index_dir,
            "source_run": self.source_run,
            "eval": dict(self.eval_summary),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class StalenessReport:
    """Whether an entry's recorded evidence still describes the present benchmark."""

    verdict: str
    reasons: tuple[str, ...] = ()

    @property
    def is_stale(self) -> bool:
        return self.verdict == VERDICT_STALE

    def describe(self) -> str:
        return f"{self.verdict}: {'; '.join(self.reasons)}" if self.reasons else self.verdict


def registry_path(data_dir: Path | str) -> Path:
    """`$DATA_DIR/adapters/registry.jsonl` -- the append-only event log."""
    return Path(data_dir) / ADAPTERS_METHOD / REGISTRY_FILENAME


def merged_root(data_dir: Path | str) -> Path:
    """`$DATA_DIR/adapters/merged/` -- merge outputs for the non-LoRA serving backends."""
    return Path(data_dir) / ADAPTERS_METHOD / MERGED_DIRNAME


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def append_event(registry: Path | str, payload: JsonObject) -> None:
    """Append one event. The log is never rewritten, so history cannot be lost by a partial write."""
    path = Path(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_events(registry: Path | str) -> list[JsonObject]:
    """Every event in order; unreadable lines are skipped so one bad append cannot brick the log."""
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
    """Register a trained adapter, digesting the goldset/corpus it was trained against.

    `index_dir` names the RAG store whose retrieval knobs (`store_meta.json`) produced the
    training contexts; they are recorded as the entry's `retrieval_fingerprint`.

    Re-registering an identical adapter is a no-op: the event is only appended when the recorded
    provenance actually changed, so a resumed campaign does not grow the log on every replay.
    """
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
    # Re-read so the returned entry carries its log sequence, which decides supersession.
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
    """Best-effort registration for the orchestrators.

    A self-improvement round or campaign entry must not abort because an injected trainer skipped
    the adapter manifest; the round's own artifacts are still valid. The adapter is simply not
    registered, which the board and `recommend` then treat as uncitable.
    """
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
    """Record that an adapter was merged into a servable artifact for `backend`."""
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
    """Tombstone a garbage-collected adapter; the register event stays in history."""
    append_event(
        registry,
        {
            "event": EVENT_DELETE,
            "adapter_id": adapter_id,
            "reason": reason,
            "created_at": utc_now(),
        },
    )


def resolve_adapter(entries: dict[str, AdapterEntry], adapter: str) -> AdapterEntry:
    """Resolve an adapter id, unique id prefix, label, or directory to its registry entry."""
    wanted = str(adapter).strip()
    if wanted in entries:
        return entries[wanted]
    by_label = [entry for entry in entries.values() if entry.adapter_label == wanted]
    if len(by_label) == 1:
        return by_label[0]
    by_dir = [entry for entry in entries.values() if _same_dir(entry, wanted)]
    if len(by_dir) == 1:
        return by_dir[0]
    if len(wanted) >= MIN_ADAPTER_ID_PREFIX:
        prefixed = [entry for entry in entries.values() if entry.adapter_id.startswith(wanted)]
        if len(prefixed) == 1:
            return prefixed[0]
        if len(prefixed) > 1:
            raise ValueError(
                f"adapter prefix {wanted!r} is ambiguous: "
                f"{', '.join(sorted(entry.short_id for entry in prefixed))}"
            )
    raise ValueError(
        f"adapter {wanted!r} is not registered; `llb list-adapters` shows the registered ids"
    )


def find_by_digest(registry: Path | str, adapter_digest: str) -> AdapterEntry | None:
    """The registry entry for a trained adapter digest, or None when it was never registered."""
    return load_registry(registry).get(str(adapter_digest))


def staleness(
    entry: AdapterEntry,
    *,
    goldset_path: Path | str | None = None,
    corpus_root: Path | str | None = None,
    index_dir: Path | str | None = None,
) -> StalenessReport:
    """Compare recorded training digests against the present goldset/corpus/RAG store.

    Defaults to the goldset/corpus/index the entry itself names, so `list-adapters` needs no
    flags. A missing recorded digest, or a goldset/corpus that no longer exists, yields
    `unknown` -- never a silent `current`. The third axis is the retrieval fingerprint: the
    store's embedder, chunk strategy/size/overlap, or retrieval mode changing since training
    flips the entry `stale` with the changed knob named, because the adapter's training contexts
    no longer exist even when `corpus_fingerprint` is unchanged. An entry registered before the
    fingerprint existed reads `unknown` on that axis, never `current`.
    """
    goldset = goldset_path if goldset_path is not None else entry.goldset_path
    corpus = corpus_root if corpus_root is not None else entry.corpus_root
    index = index_dir if index_dir is not None else entry.index_dir
    reasons: list[str] = []
    stale = False
    unknown = False
    for label, recorded, current in (
        ("goldset", entry.goldset_digest, goldset_digest_for(goldset)),
        ("corpus", entry.corpus_digest, corpus_digest_for(corpus)),
    ):
        if recorded is None or current is None:
            unknown = True
            reasons.append(f"{label} digest unavailable")
        elif recorded != current:
            stale = True
            reasons.append(f"{label} changed since training")
    changed_knobs, fingerprint_unknown = _retrieval_axis(
        entry.retrieval_fingerprint, retrieval_fingerprint_for(index)
    )
    if fingerprint_unknown:
        unknown = True
        reasons.append("retrieval fingerprint unavailable")
    for knob, recorded_value, current_value in changed_knobs:
        stale = True
        reasons.append(
            f"retrieval {knob} changed since training ({recorded_value} -> {current_value})"
        )
    if stale:
        return StalenessReport(VERDICT_STALE, tuple(reasons))
    if unknown:
        return StalenessReport(VERDICT_UNKNOWN, tuple(reasons))
    return StalenessReport(VERDICT_CURRENT)


def _retrieval_axis(
    recorded: JsonObject | None, current: JsonObject | None
) -> tuple[list[tuple[str, object, object]], bool]:
    """Per-knob retrieval comparison: (changed knobs, axis-unknown)."""
    if recorded is None or current is None:
        return [], True
    changed = [
        (knob, recorded.get(knob), current.get(knob))
        for knob, _meta_key in RETRIEVAL_FINGERPRINT_KEYS
        if recorded.get(knob) != current.get(knob)
    ]
    return changed, False


def retrieval_fingerprint_for(index_dir: Path | str | None) -> JsonObject | None:
    """The retrieval knobs recorded in a store's `store_meta.json`, or None when unreadable."""
    if index_dir is None:
        return None
    from llb.rag.store import META_FILE

    path = resolve_project_path(index_dir) / META_FILE
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("[adapters] cannot read store meta %s: %s", path, exc)
        return None
    if not isinstance(meta, dict):
        return None
    return {knob: meta.get(meta_key) for knob, meta_key in RETRIEVAL_FINGERPRINT_KEYS}


def goldset_digest_for(goldset_path: Path | str | None) -> str | None:
    """Digest the goldset exactly as the durable run journal does, or None when it is unreadable."""
    if goldset_path is None:
        return None
    path = resolve_project_path(goldset_path)
    if not path.is_file():
        return None
    from llb.executor.durability import goldset_digest
    from llb.goldset.schema import load_goldset

    try:
        return goldset_digest(load_goldset(path))
    except (OSError, ValueError) as exc:
        _LOG.warning("[adapters] cannot digest goldset %s: %s", path, exc)
        return None


def corpus_digest_for(corpus_root: Path | str | None) -> str | None:
    """Fingerprint the corpus exactly as the stale-store check does, or None when it is gone."""
    if corpus_root is None:
        return None
    root = resolve_project_path(corpus_root)
    if not root.is_dir():
        return None
    from llb.prep.corpus_governance import corpus_fingerprint

    try:
        return corpus_fingerprint(root)
    except OSError as exc:
        _LOG.warning("[adapters] cannot fingerprint corpus %s: %s", root, exc)
        return None


def adapter_rows(entries: dict[str, AdapterEntry]) -> list[JsonObject]:
    """`llb list-adapters` rows, newest first, each carrying its staleness verdict."""
    ordered = sorted(entries.values(), key=lambda entry: entry.recency, reverse=True)
    rows: list[JsonObject] = []
    for entry in ordered:
        report = staleness(entry)
        rows.append(
            {
                "adapter_id": entry.short_id,
                "base_model": entry.base_model,
                "staleness": report.verdict,
                "reasons": list(report.reasons),
                "dataset_digest": entry.dataset_digest[:ADAPTER_DIGEST_SHORT_CHARS],
                "objective_score": (entry.eval_summary or {}).get("objective_score"),
                "delta": (entry.eval_summary or {}).get("delta"),
                "source_run": entry.source_run,
                "adapter_dir": str(entry.adapter_dir),
                "merges": [str(merge.get("backend")) for merge in entry.merges],
                "created_at": entry.created_at,
            }
        )
    return rows


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


def _identity(entry: AdapterEntry) -> JsonObject:
    """Provenance without the wall-clock stamp, so a replayed round re-registers idempotently."""
    payload = entry.as_dict()
    payload.pop("created_at", None)
    return payload


def _same_dir(entry: AdapterEntry, candidate: str) -> bool:
    try:
        return entry.resolved_dir == resolve_project_path(candidate)
    except (OSError, ValueError):
        return False


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
