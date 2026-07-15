"""Run metrics, artifact paths, durability, and aggregate evaluation contracts."""

from typing import TYPE_CHECKING

from typing_extensions import NotRequired, TypedDict

from llb.core.contracts.hardware import TelemetryReport
from llb.core.contracts.rag import RetrievalMetrics
from llb.core.contracts.results import LeaderboardRow

if TYPE_CHECKING:
    from llb.tracking.manifest import RunManifest


class RunMetrics(TypedDict):
    objective_score: float
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
    abstention_accuracy: NotRequired[float]
    n_probes: NotRequired[int]


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


class EvalResult(TypedDict):
    rows: list[LeaderboardRow]
    metrics: RunMetrics
    retrieval: RetrievalMetrics
    paths: RunPaths
    table: str
    telemetry: TelemetryReport | None
    manifest: "RunManifest"
    run_timestamp: str
