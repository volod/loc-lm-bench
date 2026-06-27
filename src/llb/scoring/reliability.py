"""category expansion reliability scoring -- aggregate the typed failure taxonomy into a first-class score.

Every eval case ends in exactly one TYPED status (`llb.eval.common`): ok / empty / malformed /
refusal / retrieval_miss plus the transport tokens timeout / backend_error (and any further typed
failure a backend surfaces: context_truncation / oom / judge_failure). This module rolls those
per-case statuses up into a reliability score (fraction ok) + a per-failure-type breakdown, over
the per-case scores of ANY run (RAG board or a category). Pure + unit-tested.
"""

import json
from collections import Counter
from pathlib import Path

from llb.contracts import ReliabilityReport
from llb.eval.common import OK

# The known typed failure statuses (documentation; the aggregator counts whatever appears).
KNOWN_FAILURES = (
    "empty",
    "malformed",
    "refusal",
    "timeout",
    "backend_error",
    "retrieval_miss",
    "context_truncation",
    "oom",
    "judge_failure",
)


def reliability_report(statuses: list[str]) -> ReliabilityReport:
    """Roll per-case statuses into reliability (fraction ok) + a per-failure-type count."""
    n = len(statuses)
    n_ok = sum(1 for s in statuses if s == OK)
    failures = Counter(s for s in statuses if s != OK)
    return {
        "n": n,
        "n_ok": n_ok,
        "reliability": round(n_ok / n, 6) if n else 0.0,
        "failures": dict(sorted(failures.items())),
    }


def read_case_statuses(run_dir: Path | str) -> list[str]:
    """Extract the per-case `status` column from a run bundle (scores.parquet or scores.jsonl)."""
    run_dir = Path(run_dir)
    parquet = run_dir / "scores.parquet"
    if parquet.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(parquet, columns=["status"])
        return [str(s) for s in table.column("status").to_pylist()]
    jsonl = run_dir / "scores.jsonl"
    if jsonl.exists():
        statuses: list[str] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                statuses.append(str(json.loads(line).get("status", "")))
        return statuses
    raise FileNotFoundError(f"no scores.parquet or scores.jsonl under {run_dir}")
