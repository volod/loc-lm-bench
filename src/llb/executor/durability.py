"""Resumable, retrying per-case execution for run-eval (the durable-eval-runner).

`execute_cases` (in `cases.py`) runs every gold item once, in memory: a transient endpoint blip,
a launcher-owned backend crash, or a host restart loses the whole run. This module wraps that loop
so a long eval campaign survives those faults:

- transient per-case transport failures (terminal status `timeout` / `backend_error`) retry with
  capped exponential backoff; a scored answer or a non-transport terminal status (ok / empty /
  malformed / refusal / retrieval_miss) is NEVER retried -- those are real outcomes, not faults;
- each completed case appends its terminal state to an append-only `cases.progress.jsonl` journal in
  the staged run directory, keyed by `item_id`, so `--resume` reuses it instead of re-spending the
  model call;
- when a case exhausts its per-case retries still in a transport failure and the launcher owns a
  serving process, the backend is relaunched a bounded number of times and the case gets another
  full round of attempts -- so a crashed launcher-owned backend recovers.

Everything downstream of the raw terminal state (score_case, retrieval pairs, answers) is recomputed
deterministically on resume, so a resumed run's per-case scores are identical to an uninterrupted
one. A window that reached a terminal state -- including a terminal transport failure after
exhausting retries and relaunches -- IS journaled (done-as-is, matching the ontology extraction
journal); only a hard process kill mid-case leaves a case un-journaled, so resume re-runs it.

Journaling, retry, and relaunch are unit-testable with a fake `runner_fn`, a fake relaunch callable,
and an injected `sleep` -- no endpoint, GPU, or real clock.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT
from llb.contracts import DurabilityStatus
from llb.eval import graph as eval_graph
from llb.executor.cases import CaseBatch, score_case, spans_as_dicts
from llb.fsutil import atomic_write_text
from llb.goldset.schema import GoldItem

RagState = eval_graph.RagState
_LOG = logging.getLogger(__name__)

# Only transport-level failures are retried; every other terminal status is a real case outcome.
RETRYABLE_STATUSES = frozenset({ERR_TIMEOUT, ERR_BACKEND})

JOURNAL_NAME = "cases.progress.jsonl"
JOURNAL_META_NAME = "cases.progress.meta.json"

# Fields of the terminal RagState the journal needs to reproduce scoring, retrieval, and the judge
# record deterministically on resume (context/question are not scored, so they are dropped).
_JOURNALED_STATE_KEYS = ("retrieved", "answer", "status", "error", "usage")


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


def _backoff_seconds(policy: RetryPolicy, attempt: int) -> float:
    """Capped exponential backoff for the `attempt`-th retry (0-indexed)."""
    return float(min(policy.retry_backoff_s * (2**attempt), policy.backoff_cap_s))


def _run_case_with_retry(
    item: GoldItem,
    runner_fn: Callable[[GoldItem], RagState],
    policy: RetryPolicy,
    relaunch: Callable[[], None] | None,
    sleep: Callable[[float], None],
    counters: DurabilityCounters,
) -> RagState:
    """Execute one case, retrying transient transport failures and (bounded) relaunching a dead
    launcher-owned backend. Returns the terminal state (which may still be a transport failure)."""
    relaunches_used = 0
    while True:
        state = runner_fn(item)
        attempt = 0
        while state.get("status") in RETRYABLE_STATUSES and attempt < policy.max_case_retries:
            sleep(_backoff_seconds(policy, attempt))
            counters.case_retries += 1
            attempt += 1
            state = runner_fn(item)
        if state.get("status") not in RETRYABLE_STATUSES:
            return state
        # Per-case retries exhausted, still a transport failure: relaunch the backend and try again.
        if relaunch is not None and relaunches_used < policy.max_backend_relaunches:
            _LOG.warning(
                "[run-eval] case %s still failing (%s) after %d retries; relaunching backend",
                item.id,
                state.get("status"),
                policy.max_case_retries,
            )
            relaunch()
            counters.backend_relaunches += 1
            relaunches_used += 1
            continue
        return state


def execute_cases_durable(
    items: list[GoldItem],
    runner_fn: Callable[[GoldItem], RagState],
    embedder: Any,
    *,
    journal: CaseJournal,
    policy: RetryPolicy,
    relaunch: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    counters: DurabilityCounters | None = None,
) -> tuple[CaseBatch, DurabilityCounters]:
    """Resumable, retrying variant of `execute_cases`.

    Journaled cases are reused verbatim; the rest run through the retry/relaunch loop and journal as
    they complete. Scoring, retrieval pairs, and answers are rebuilt in item order, so the returned
    batch is identical whether or not the run was interrupted.
    """
    counters = counters or DurabilityCounters()
    journal.load()
    rows = []
    retrieval_pairs = []
    answers: list[tuple[GoldItem, str]] = []
    for item in items:
        cached = journal.get(item.id)
        if cached is not None:
            state = cached
            counters.resumed_cases += 1
        else:
            state = _run_case_with_retry(item, runner_fn, policy, relaunch, sleep, counters)
            journal.record(item.id, state)
        spans = spans_as_dicts(item)
        rows.append(score_case(item, state, embedder=embedder))
        retrieval_pairs.append((state.get("retrieved", []), spans))
        answers.append((item, state.get("answer", "")))
    return CaseBatch(rows=rows, retrieval_pairs=retrieval_pairs, answers=answers), counters


def resume_target(
    run_dir_of: Callable[[str], Path],
    staging_dir_of: Callable[[str], Path],
    resume: Path | str,
) -> tuple[str, str, Path, Path]:
    """Resolve a `--resume` handle (canonical run dir OR its hidden staging sibling) to
    `(run_timestamp, run_id, run_dir, staging_dir)`.

    `run_dir_of` / `staging_dir_of` are the config's path builders, so validation of the timestamp
    segment stays in one place.
    """
    name = Path(resume).name
    canonical = name[1:-4] if name.startswith(".") and name.endswith(".tmp") else name
    if "-" not in canonical:
        raise SystemExit(f"[run-eval] cannot resume: '{resume}' is not a run directory")
    run_dir = run_dir_of(canonical)  # validates the segment shape
    staging_dir = staging_dir_of(canonical)
    run_id = canonical.rsplit("-", 1)[1]
    return canonical, run_id, run_dir, staging_dir
