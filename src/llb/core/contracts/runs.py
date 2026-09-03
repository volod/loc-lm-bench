"""Run metrics, artifact paths, durability, the run manifest, and aggregate results.

`RunManifest` is the registered `llb.run-manifest` contract and lives here rather than beside
`persist_run`: the registry declares it, and a family module that had to import the writer to
reach its model would close a cycle through `llb.artifacts`.
"""

import platform
import sys
from datetime import datetime, timezone
from typing import Literal

from pydantic import ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.hardware import ContentionReport, TelemetryReport
from llb.core.contracts.judging import JudgeStatus
from llb.core.contracts.rag import RetrievalMetrics
from llb.core.contracts.results import LeaderboardRow

RUN_MANIFEST_SCHEMA_ID = "llb.run-manifest"


class RunMetrics(TypedDict):
    objective_score: float
    ranking_score: NotRequired[float]
    token_precision: NotRequired[float]
    token_recall: NotRequired[float]
    found_rate: NotRequired[float]
    mean_completion_tokens: NotRequired[float]
    reliability: float
    tokens_per_s: float
    mean_power_w: NotRequired[float]
    tokens_per_watt: NotRequired[float]
    quality_per_watt: NotRequired[float]
    judge_score: NotRequired[float]
    stage_latency: NotRequired[dict[str, float]]
    groundedness: NotRequired[float]
    citation_validity: NotRequired[float]
    citation_coverage: NotRequired[float]
    hallucinated_citation_rate: NotRequired[float]
    # Response-integrity guard (`llb.scoring.answer_guard`), reported beside `reliability` over
    # the same case denominator: the share of cases whose completion leaked deliberation into the
    # answer body despite the backend's thinking-suppression flag, the share that answered in a
    # language other than the question's, and the mean characters the leaks accounted for -- the
    # term that inflates `mean_completion_tokens` and, through it, throughput and cost.
    reasoning_leak_rate: NotRequired[float]
    language_mismatch_rate: NotRequired[float]
    mean_reasoning_leak_chars: NotRequired[float]
    abstention_accuracy: NotRequired[float]
    n_probes: NotRequired[int]
    # Declared answer contract (typed-rag-answer-envelope); present only on an envelope run.
    envelope_conformance: NotRequired[float]
    envelope_schema_invalid_rate: NotRequired[float]
    envelope_malformed_rate: NotRequired[float]
    envelope_repair_rate: NotRequired[float]
    mean_claims: NotRequired[float]
    # Step two of the answer gate (ontology-validated-answer-gate); present only when it ran.
    # `ontology_violation_rate` is the share of cases the gate refused, `validation_checked_rate`
    # the share whose envelope declared a triple the gate could test at all, and
    # `validation_repair_rate` the share that spent the one bounded semantic reprompt.
    ontology_violation_rate: NotRequired[float]
    validation_checked_rate: NotRequired[float]
    validation_repair_rate: NotRequired[float]


class RunEnvironment(TypedDict):
    python: str
    platform: str


class RunPaths(TypedDict):
    manifest: str
    scores: str
    mirror: str
    retrieval: NotRequired[str]
    worksheet: NotRequired[str]
    probes: NotRequired[str]
    insufficient_context_report: NotRequired[str]


class DurabilityStatus(TypedDict):
    """Fault-recovery counters for one evaluation run."""

    case_retries: int
    backend_relaunches: int
    resumed_cases: int


def utc_now() -> str:
    """The manifest timestamp format every run bundle is stamped with."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_env() -> RunEnvironment:
    """Minimal reproducibility environment (GPU/driver added with telemetry in backend telemetry)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


class RunManifest(ArtifactContract):
    """`manifest.json`: the immutable per-run record -- config, environment, headline metrics.

    Every optional block is a lane that may not have run (telemetry, contention, a judge, a
    prompt-system package, a document-lane context window), not a field a reader defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.run-manifest"] = "llb.run-manifest"
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(min_length=1)
    run_name: str = Field(min_length=1)
    split: str | None = None
    created_at: str = Field(default_factory=utc_now)
    config: JsonObject
    env: RunEnvironment = Field(default_factory=capture_env)
    metrics: RunMetrics | None = None
    retrieval: RetrievalMetrics | None = None
    judge: JudgeStatus | None = None
    telemetry: TelemetryReport | None = None
    contention: ContentionReport | None = None
    durability: DurabilityStatus | None = None
    prompt_system_provenance: JsonObject | None = None
    # Set only by a context lane that laid whole documents into the prompt: the declared window,
    # the window the backend was probed as serving, and which of the two bound the skip threshold.
    context_window: JsonObject | None = None
    n_cases: int = Field(default=0, ge=0)


class EvalResult(TypedDict):
    rows: list[LeaderboardRow]
    metrics: RunMetrics
    retrieval: RetrievalMetrics
    paths: RunPaths
    table: str
    telemetry: TelemetryReport | None
    manifest: RunManifest
    run_timestamp: str
