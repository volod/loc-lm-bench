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


class PlantedLabelRecord(TypedDict):
    """One planted ground-truth label `prepare-synthetic-corpus` emits for the text-analysis
    benchmark (M5.0 scoring schema). `kind` is a text-analysis sub-task (see
    `llb.scoring.text_analysis`); `value` is the canonical surface string a candidate must
    recover; `aliases` are other accepted surface forms; `doc_id`/`char_start`/`char_end`
    ground the label in the synthetic doc; `attrs` carries kind-specific structure (e.g. a
    trend's direction, a contradiction's paired span ids); `scoring` is "objective" | "judged".
    """

    label_id: str
    kind: str
    value: str
    aliases: NotRequired[list[str]]
    doc_id: NotRequired[str]
    char_start: NotRequired[int]
    char_end: NotRequired[int]
    attrs: NotRequired[JsonObject]
    scoring: NotRequired[str]


class SubtaskScore(TypedDict):
    """Objective recovery score for one text-analysis sub-task over one document (M5.0)."""

    kind: str
    objective: bool
    n_labels: int
    n_pred: int
    matched: list[tuple[str, float]]  # (label_id, credit in {1.0, partial})
    precision: float
    recall: float
    f1: float


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
    judge_score: NotRequired[float]  # per-case judge (mean of faithfulness + answer-relevancy)


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


class BoardRow(TypedDict):
    rank: int | None
    model: str
    backend: str
    tier: str  # "private" (Tier-2) | "screen" (Tier-1); a board never mixes the two
    quality: float  # weighted-blend headline (objective + trusted judge)
    quality_ci: NotRequired[tuple[float, float]]  # bootstrap CI on the per-case headline blend
    objective_ci: NotRequired[tuple[float, float]]  # per-case objective CI (when available)
    semantic_ci: NotRequired[tuple[float, float]]  # per-case semantic CI (when that signal is on)
    judge_ci: NotRequired[tuple[float, float]]  # per-case judge CI (when the judge is trusted/on)
    avg_rank: float  # mean of per-quality-signal ranks (lower is better)
    objective: float
    judge: float | None
    semantic: float | None
    reliability: float
    tokens_per_s: float
    peak_vram_mb: float | None
    pareto: bool  # on (quality up, tok/s up, vram down)
    unresolved: bool  # quality CI overlaps the model ranked just above -> tie not resolved
    feasible: bool
    n_cases: int


class BackendMetadata(TypedDict, total=False):
    backend: str
    host: str
    gpu_memory_utilization: float
    n_gpu_layers: int  # llama.cpp GPU/CPU layer split (-1 == all on GPU)
    ctx_size: int | None  # llama.cpp requested context (`-c`)
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
    n_gpu_layers: int | None  # llama.cpp offload split (None for non-llama.cpp backends)
    gpus: list[GpuSummary]


class RunMetrics(TypedDict):
    objective_score: float
    reliability: float
    tokens_per_s: float
    judge_score: NotRequired[float]  # mean per-case judge, recorded only when the judge is trusted


class RunEnvironment(TypedDict):
    python: str
    platform: str


class JudgeStatus(TypedDict):
    calibration_rho: float | None
    threshold: float
    trusted: bool
    provider: NotRequired[str]
    model: NotRequired[str]
    base_url: NotRequired[str | None]
    prompt_language: NotRequired[str]
    metrics: NotRequired[list[str]]


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
    # Embedding-aware weight estimate (M4.1): under partial quant (w4a16/int4/fp8) the token
    # embedding + norms stay high-precision, so `params_b x bpw` under-counts. The planner reads
    # these from the spec (or a cached config.json) and prices the embedding separately.
    vocab_size: NotRequired[int]
    hidden_size: NotRequired[int]
    tie_word_embeddings: NotRequired[
        bool
    ]  # tied -> output head shares the embedding (counted once)
    embed_bpw: NotRequired[float]  # bits/weight of the high-precision part (default 16)
    # Escape hatch for architectures whose always-high-precision mass is not just the vocab
    # embedding (e.g. Gemma 3n Per-Layer Embeddings): billions of params kept at `embed_bpw`.
    hi_precision_params_b: NotRequired[float]
    # Optional cross-backend serving options for the AvailabilityResolver (M3.2): a map of
    # backend -> source string OR a per-source record carrying its own quant/arch overrides,
    # so the planner prices the actual artifact (e.g. a q4 GGUF, not the vLLM bf16 metadata).
    sources: NotRequired[dict[str, "str | SourceRecord"]]


class SourceRecord(TypedDict):
    """A per-backend artifact for one logical model: its own source + metadata overrides.
    Any field omitted falls back to the parent `ModelSpec` (same architecture, different
    quant/packaging)."""

    source: str
    quant: NotRequired[str]
    bpw: NotRequired[float]
    params_b: NotRequired[float]
    n_layers: NotRequired[int]
    kv_dim: NotRequired[int]
    max_context: NotRequired[int]
    vocab_size: NotRequired[int]
    hidden_size: NotRequired[int]
    tie_word_embeddings: NotRequired[bool]
    embed_bpw: NotRequired[float]
    hi_precision_params_b: NotRequired[float]
    min_vram_gb: NotRequired[int | float]
    gated: NotRequired[bool]
    license_url: NotRequired[str]


class BackendCandidate(TypedDict):
    backend: str
    source: str
    quant: NotRequired[str | None]  # the quant the planner actually priced for this artifact
    gpu_layers: NotRequired[
        int
    ]  # planner GPU/CPU layer split at the planning context (llama.cpp -ngl)
    available: bool
    verdict: str  # planner verdict at the host budget: gpu / offload / no / unknown
    runnable: bool  # available AND the backend can actually serve at that verdict
    reason: str


class ResolvedModel(TypedDict):
    name: str
    chosen_backend: str | None
    chosen_source: str | None
    verdict: str
    candidates: list[BackendCandidate]
    note: str


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


class ScreenTaskResult(TypedDict):
    task: str
    metric: str
    score: float


class ScreenReport(TypedDict):
    model: str
    backend: str
    track: str  # "logprob" (vLLM, MCQ via loglikelihood) | "generation" (generate-until)
    requested_tasks: list[str]
    results: list[ScreenTaskResult]
    covered: list[str]
    missing: list[str]  # requested but absent -> the screen ran only partially
    complete: bool


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


class ResidentProc(TypedDict):
    pid: int
    used_mb: int


class ContentionReport(TypedDict):
    """Pre-launch VRAM-contention guard outcome (M4.2)."""

    total_mb: int
    free_mb: int
    requested_util: float
    safe_util: float  # gpu-memory-utilization to actually use (derated to fit free VRAM)
    target_mb: int  # safe_util x total -- what vLLM may reserve
    weight_floor_mb: int  # embedding-aware weights estimate (M4.1) used for the abort check
    residents: list[ResidentProc]  # other GPU processes holding VRAM
    derated: bool  # True when safe_util < requested_util (contention lowered it)
    fits: bool  # False -> even the derated target cannot hold weights + KV
    action: str  # ok | derate | abort
    note: str


class GpuSample(TypedDict):
    index: int
    temp_c: int | None
    power_w: float | None
    sm_clock_mhz: int | None
    mem_clock_mhz: int | None


class CoolDownReport(TypedDict):
    waited_s: float
    final_temp_c: int | None
    capped: bool


class IsolationOutcome(TypedDict):
    vram_residual_mb: int | None
    vram_verdict: str | None  # reclaimed | leaked | baseline_shift | None (gate skipped)
    cooldown: CoolDownReport
    gpu: list[GpuSample]


class CellResult(TypedDict):
    cell_key: str
    model: str
    backend: str
    status: str  # done | skipped | failed
    run_dir: str | None
    vram_residual_mb: int | None
    cooldown_s: float
    cooldown_capped: bool
    gpu: list[GpuSample]
    detail: str


class SweepReport(TypedDict):
    sweep_id: str
    n_cells: int
    completed: int
    skipped: int
    failed: int
    results: list[CellResult]


class EvalResult(TypedDict):
    rows: list[LeaderboardRow]
    metrics: RunMetrics
    retrieval: RetrievalMetrics
    paths: RunPaths
    table: str
    telemetry: TelemetryReport | None
    manifest: "RunManifest"
    run_timestamp: str
