"""Focused durability journal implementation."""

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from llb.core.contracts.runs import DurabilityStatus
from llb.eval import graph as eval_graph
from llb.executor.cases import spans_as_dicts
from llb.core.fsutil import atomic_write_text
from llb.goldset.schema import GoldItem

RagState = eval_graph.RagState

_LOG = logging.getLogger(__name__)

JOURNAL_NAME = "cases.progress.jsonl"

JOURNAL_META_NAME = "cases.progress.meta.json"

_JOURNALED_STATE_KEYS = (
    "retrieved",
    "answer",
    "status",
    "error",
    "usage",
    "retrieve_latency_s",
    "rerank_latency_s",
)


@dataclass
class RetryPolicy:
    """Bounded fault-recovery budget for one run."""

    max_case_retries: int = 2
    retry_backoff_s: float = 1.0
    max_backend_relaunches: int = 1
    backoff_cap_s: float = 30.0


@dataclass
class DurabilityCounters:
    """Mutable fault-recovery tallies accumulated across a run."""

    case_retries: int = 0
    backend_relaunches: int = 0
    resumed_cases: int = 0

    def as_status(self) -> DurabilityStatus:
        return {
            "case_retries": self.case_retries,
            "backend_relaunches": self.backend_relaunches,
            "resumed_cases": self.resumed_cases,
        }


def journal_path(staging_dir: Path | str) -> Path:
    return Path(staging_dir) / JOURNAL_NAME


def journal_meta_path(staging_dir: Path | str) -> Path:
    return Path(staging_dir) / JOURNAL_META_NAME


def drop_journal(staging_dir: Path | str) -> None:
    """Remove the journal + meta from the staging dir before the atomic finalize, so the published
    bundle never carries them (the journal is a resume aid, not a run artifact)."""
    journal_path(staging_dir).unlink(missing_ok=True)
    journal_meta_path(staging_dir).unlink(missing_ok=True)


def _json_default(obj: Any) -> Any:
    """Coerce a non-JSON scalar (e.g. a numpy float retrieval score) so a state always serializes."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


class CaseJournal:
    """Append-only journal of completed cases, keyed by `item_id`.

    Sequential by construction (the runner evaluates one case at a time), so no lock is needed. A
    malformed trailing line -- the expected shape of a killed run -- is skipped on load rather than
    aborting the resume.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._done: dict[str, RagState] = {}

    def load(self) -> int:
        """Load an existing journal (idempotent). Returns the number of journaled cases."""
        if not self.path.is_file():
            return 0
        loaded = 0
        with self.path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    item_id = str(record["item_id"])
                    state = record["state"]
                    if not isinstance(state, dict):
                        raise TypeError("state must be an object")
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    _LOG.warning(
                        "[run-eval] skipping malformed case-journal line %s:%d (%s)",
                        self.path,
                        line_no,
                        exc,
                    )
                    continue
                self._done[item_id] = cast(RagState, state)  # ids are unique per goldset
                loaded += 1
        if loaded:
            _LOG.info("[run-eval] resuming: %d journaled cases in %s", loaded, self.path)
        return loaded

    def get(self, item_id: str) -> RagState | None:
        return self._done.get(item_id)

    def record(self, item_id: str, state: RagState) -> None:
        """Append one completed case (kept minimal to the scored fields). Idempotent per id."""
        if item_id in self._done:
            return
        trimmed = cast(
            RagState,
            {key: value for key, value in dict(state).items() if key in _JOURNALED_STATE_KEYS},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"item_id": item_id, "state": trimmed}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        self._done[item_id] = trimmed


def config_digest(fingerprint: Mapping[str, Any]) -> str:
    """A stable hash of the reproducibility-relevant config (retry knobs are NOT part of it, so a
    run can be resumed with a different retry budget)."""
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def goldset_digest(items: list[GoldItem]) -> str:
    """A stable hash of the exact scored items (id + question + reference + spans), so a resume onto
    a changed goldset is refused."""
    payload = [
        {
            "id": item.id,
            "question": item.question,
            "reference_answer": item.reference_answer,
            "source_spans": spans_as_dicts(item),
        }
        for item in items
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_journal_meta(
    staging_dir: Path | str,
    *,
    config_fingerprint: Mapping[str, Any],
    items: list[GoldItem],
    run_id: str,
    split: str,
) -> None:
    """Pin the determinism-critical identity of a run at start, so `--resume` can refuse a mismatch."""
    meta = {
        "run_id": run_id,
        "split": split,
        "config_digest": config_digest(config_fingerprint),
        "goldset_digest": goldset_digest(items),
        "n_items": len(items),
    }
    atomic_write_text(
        journal_meta_path(staging_dir), json.dumps(meta, ensure_ascii=False, indent=2)
    )


def verify_resume_meta(
    staging_dir: Path | str,
    *,
    config_fingerprint: Mapping[str, Any],
    items: list[GoldItem],
    split: str,
) -> dict[str, Any]:
    """Load the pinned meta and refuse a resume whose config, goldset, or split changed."""
    path = journal_meta_path(staging_dir)
    if not path.is_file():
        raise SystemExit(
            f"[run-eval] cannot resume: no journal meta at {path} "
            "(only an interrupted durable run can be resumed)"
        )
    try:
        meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[run-eval] cannot resume: unreadable journal meta {path} ({exc})")
    mismatched = []
    if meta.get("config_digest") != config_digest(config_fingerprint):
        mismatched.append("config")
    if meta.get("goldset_digest") != goldset_digest(items):
        mismatched.append("goldset")
    if meta.get("split") != split:
        mismatched.append("split")
    if mismatched:
        raise SystemExit(
            "[run-eval] cannot resume: "
            + ", ".join(mismatched)
            + " changed since the interrupted run; start a fresh run instead."
        )
    return meta
