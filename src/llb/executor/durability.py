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

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llb.backends.base import ERR_BACKEND, ERR_TIMEOUT
from llb.executor.cases import CaseBatch, ScoreOptions, score_case, spans_as_dicts
from llb.goldset.schema import GoldItem
from llb.executor.durability_journal import (
    CaseJournal,
    DurabilityCounters,
    RagState,
    RetryPolicy,
    _LOG,
)


# Only transport-level failures are retried; every other terminal status is a real case outcome.
RETRYABLE_STATUSES = frozenset({ERR_TIMEOUT, ERR_BACKEND})


# Fields of the terminal RagState the journal needs to reproduce scoring, retrieval, and the judge
# record deterministically on resume (context/question are not scored, so they are dropped). The
# stage latencies (rerank-context-order) are journaled so a resumed run's rows keep them.


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
    options: ScoreOptions | None = None,
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
        rows.append(score_case(item, state, embedder=embedder, options=options))
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
