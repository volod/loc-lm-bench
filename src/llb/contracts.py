"""Shared typed contracts for records crossing package boundaries.

Pydantic models remain the validation boundary for user configuration, gold items, and
run manifests. These TypedDicts document and statically check the lightweight records
passed between retrieval, execution, scoring, telemetry, and persistence.
"""

from typing import TYPE_CHECKING, Any, TypeAlias

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from llb.backends.hardware import Gpu
    from llb.tracking.manifest import RunManifest

JsonObject: TypeAlias = dict[str, Any]


class ChatMessage(TypedDict):
    role: str
    content: str


class SourceSpanRecord(TypedDict):
    doc_id: str
    char_start: int
    char_end: int
    text: str


class ChunkRecord(SourceSpanRecord):
    chunk_id: NotRequired[str]
    parent_id: NotRequired[str]
    matched_child_id: NotRequired[str]
    strategy: NotRequired[str]
    size: NotRequired[int]
    overlap: NotRequired[int]
    metadata: NotRequired[JsonObject]
    retrieval_score: NotRequired[float | None]
    rank: NotRequired[int]


class RagStoreMeta(TypedDict):
    mode: str
    strategy: str
    size: int
    overlap: int
    child_size: int
    embedding_model: str
    n_indexed: int
    n_parents: int
    dim: int


class UsageRecord(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    tokens_per_s: float


class RetrievalMetrics(TypedDict):
    n: int
    k: int
    recall_at_k: float
    mrr: float


RetrievalPair: TypeAlias = tuple[list[ChunkRecord], list[SourceSpanRecord]]


class CorrectnessScores(TypedDict):
    score: float
    token_f1: float
    exact: float
    contains: float
    semantic: NotRequired[float]


class CaseScoreRow(TypedDict):
    item_id: str
    split: str
    status: str
    objective_score: float
    token_f1: float
    exact: float
    contains: float
    retrieval_hit: float
    first_hit_rank: int | None
    tokens_per_s: float
    latency_s: float
    completion_tokens: int
    answer_preview: str
    semantic: NotRequired[float]


class LeaderboardRow(TypedDict):
    rank: int | None
    model: str
    backend: str
    quality: float
    objective: float
    judge: float | None
    reliability: float
    tokens_per_s: float
    peak_vram_mb: float | None
    feasible: bool
    n_cases: int


class BackendMetadata(TypedDict, total=False):
    backend: str
    host: str
    gpu_memory_utilization: float
    served_context: int | None
    tokens_per_s: float
    last_completion_tokens: int
    load_time_s: float | None


class GpuSummary(TypedDict):
    name: str
    total_mb: int
    driver: str


class TelemetryReport(TypedDict):
    steady_tokens_per_s: float
    mean_completion_tokens: float
    tokens_per_char: float
    max_new_tokens: int
    n_warmup: int
    n_measured: int
    n_failed: int
    load_time_s: float | None
    peak_vram_mb: int | None
    requested_context: int | None
    served_context: int | None
    backend: str | None
    gpu_memory_utilization: float | None
    gpus: list[GpuSummary]


class RunMetrics(TypedDict):
    objective_score: float
    reliability: float
    tokens_per_s: float


class RunEnvironment(TypedDict):
    python: str
    platform: str


class JudgeStatus(TypedDict):
    calibration_rho: float | None
    threshold: float
    trusted: bool


class RunPaths(TypedDict):
    manifest: str
    scores: str
    mirror: str
    worksheet: NotRequired[str]


class ValidationReport(TypedDict):
    n: int
    splits: dict[str, int]
    errors: list[str]


class ChunkSummary(TypedDict):
    n: int
    avg: int
    min: int
    max: int


class ModelSpec(TypedDict):
    name: str
    backend: str
    source: str
    min_vram_gb: NotRequired[int | float]
    notes: NotRequired[str]
    license_url: NotRequired[str]
    gated: NotRequired[bool]
    params_b: NotRequired[float]
    quant: NotRequired[str]
    bpw: NotRequired[float]
    n_layers: NotRequired[int]
    kv_dim: NotRequired[int]
    max_context: NotRequired[int]


class PreparedModel(ModelSpec):
    action: str
    reason: str
    status: NotRequired[str]
    detail: NotRequired[str]


class PreparationReport(TypedDict):
    gpus: list["Gpu"]
    max_vram_mb: int
    results: list[PreparedModel]


class ModelPlanRow(TypedDict):
    name: str
    backend: str
    params_b: float | None
    quant: str | None
    weights_mib: float | None
    n_layers: int | None
    ctx_gpu: int
    ctx_max: int
    gpu_layers: int
    verdict: str
    note: str


class CalibrationResult(TypedDict):
    rho: float
    ci_low: float
    ci_high: float
    n: int
    threshold: float
    trusted: bool


class JudgeInputRecord(TypedDict):
    question: str
    answer: str
    contexts: list[str]


class JudgeScore(TypedDict):
    faithfulness: float
    answer_relevancy: float


class SquadAnswers(TypedDict):
    text: list[str]
    answer_start: list[int | None]


class SquadRecord(TypedDict):
    id: str | None
    context: str
    question: str
    answers: SquadAnswers


class RagItemSpec(TypedDict):
    id: str
    doc: str
    answer_span: str
    question: str
    reference_answer: str
    split: str
    provenance: NotRequired[str]
    verified: NotRequired[bool]


class RagDataSpec(TypedDict):
    lang: str
    docs: dict[str, str]
    items: list[RagItemSpec]


class WorksheetItem(TypedDict):
    id: str
    split: str
    question: str
    reference_answer: str


class VramReclaimReport(TypedDict):
    reclaimed: bool
    residual_mb: int
    polls: int


class EvalResult(TypedDict):
    rows: list[LeaderboardRow]
    metrics: RunMetrics
    retrieval: RetrievalMetrics
    paths: RunPaths
    table: str
    telemetry: TelemetryReport | None
    manifest: "RunManifest"
    run_timestamp: str
