"""Training-provenance fingerprints and adapter staleness decisions."""

import json
import logging
from pathlib import Path

from llb.core.contracts import JsonObject
from llb.core.paths import resolve_project_path
from llb.finetune.registry.model import (
    RETRIEVAL_FINGERPRINT_KEYS,
    VERDICT_CURRENT,
    VERDICT_STALE,
    VERDICT_UNKNOWN,
    AdapterEntry,
    StalenessReport,
)

_LOG = logging.getLogger(__name__)


def staleness(
    entry: AdapterEntry,
    *,
    goldset_path: Path | str | None = None,
    corpus_root: Path | str | None = None,
    index_dir: Path | str | None = None,
) -> StalenessReport:
    """Compare recorded training digests with the present benchmark and RAG store."""
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


def retrieval_fingerprint_for(index_dir: Path | str | None) -> JsonObject | None:
    """Read retrieval knobs from store metadata, or return None when unavailable."""
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
    """Digest a goldset as the durable run journal does."""
    if goldset_path is None:
        return None
    path = resolve_project_path(goldset_path)
    if not path.is_file():
        return None
    from llb.executor.durability_journal import goldset_digest
    from llb.goldset.schema import load_goldset

    try:
        return goldset_digest(load_goldset(path))
    except (OSError, ValueError) as exc:
        _LOG.warning("[adapters] cannot digest goldset %s: %s", path, exc)
        return None


def corpus_digest_for(corpus_root: Path | str | None) -> str | None:
    """Fingerprint a corpus as the stale-store check does."""
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


def _retrieval_axis(
    recorded: JsonObject | None, current: JsonObject | None
) -> tuple[list[tuple[str, object, object]], bool]:
    if recorded is None or current is None:
        return [], True
    changed = [
        (knob, recorded.get(knob), current.get(knob))
        for knob, _meta_key in RETRIEVAL_FINGERPRINT_KEYS
        if recorded.get(knob) != current.get(knob)
    ]
    return changed, False
