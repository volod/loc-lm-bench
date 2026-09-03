"""Complete run-bundle records for tests and sample bundles.

A score row and a telemetry report each have a required core -- the columns every producer writes
on every case, and the counters a telemetry pass always records. A test that wants to say
"a case that scored 1.0" should not have to restate the other fifteen, and a fixture that restates
them by hand drifts away from the contract the moment a column joins it. One builder per record
keeps the fixtures and the producers saying the same thing.
"""

from typing import Any

from llb.core.contracts.common import JsonObject
from llb.core.contracts.hardware import TelemetryReport

NEUTRAL_ANSWER_PREVIEW = "answer"


def case_score_row(item_id: str, **overrides: Any) -> JsonObject:
    """One complete `llb.case-score` row: every required column, at a scored-nothing baseline."""
    row: JsonObject = {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": 0.0,
        "token_f1": 0.0,
        "exact": 0.0,
        "contains": 0.0,
        "retrieval_hit": 0.0,
        "first_hit_rank": None,
        "tokens_per_s": 0.0,
        "latency_s": 0.0,
        "completion_tokens": 0,
        "answer_preview": NEUTRAL_ANSWER_PREVIEW,
    }
    row.update(overrides)
    return row


def run_metrics(**overrides: Any) -> JsonObject:
    """One complete `RunMetrics` block: the three headline numbers every run records."""
    metrics: JsonObject = {"objective_score": 0.0, "reliability": 1.0, "tokens_per_s": 0.0}
    metrics.update(overrides)
    return metrics


def retrieval_metrics(**overrides: Any) -> JsonObject:
    """One complete `RetrievalMetrics` block: the population and the two headline rates."""
    metrics: JsonObject = {"n": 0, "k": 0, "recall_at_k": 0.0, "mrr": 0.0}
    metrics.update(overrides)
    return metrics


def run_manifest_payload(**overrides: Any) -> JsonObject:
    """One complete `llb.run-manifest` document, at a ran-nothing baseline.

    Every required field is present, so a fixture states only what its test is about: which model
    a bundle names, what it scored, which split it is.
    """
    payload: JsonObject = {
        "schema_id": "llb.run-manifest",
        "schema_version": "1.0.0",
        "run_id": "fixture-run",
        "run_name": "fixture-run",
        "created_at": "2026-01-01T00:00:00Z",
        "config": {},
        "env": {"python": "3.13.0", "platform": "fixture"},
        "n_cases": 0,
    }
    payload.update(overrides)
    return payload


def case_retrieval_row(item_id: str, **overrides: Any) -> JsonObject:
    """One complete `llb.case-retrieval` row with nothing retrieved and nothing expected."""
    row: JsonObject = {"item_id": item_id, "retrieved": [], "gold_spans": []}
    row.update(overrides)
    return row


def agent_profile_payload(fields: JsonObject, **overrides: Any) -> JsonObject:
    """One complete `llb.agent-profile` document around the fields a test is about."""
    payload: JsonObject = {
        "generated_at": "2026-01-01T00:00:00Z",
        "anchor": {"resolved": False},
        "drift": {},
        "states": {},
        "fields": fields,
        "replay": {},
    }
    payload.update(overrides)
    return payload


def telemetry_report(**overrides: Any) -> TelemetryReport:
    """One complete `TelemetryReport`: the counters a telemetry pass always records."""
    report: TelemetryReport = {
        "steady_tokens_per_s": 0.0,
        "mean_completion_tokens": 0.0,
        "tokens_per_char": 0.0,
        "max_new_tokens": 0,
        "n_warmup": 0,
        "n_measured": 0,
        "n_failed": 0,
        "load_time_s": None,
        "peak_vram_mb": None,
        "requested_context": None,
        "served_context": None,
        "backend": None,
        "gpu_memory_utilization": None,
        "n_gpu_layers": None,
        "gpus": [],
    }
    report.update(overrides)  # type: ignore[typeddict-item]
    return report
