"""Backend telemetry, GPU isolation, and executor sweep contracts."""

from typing_extensions import NotRequired, TypedDict


class BackendMetadata(TypedDict, total=False):
    backend: str
    host: str
    gpu_memory_utilization: float
    cpu_offload_gb: float | None
    kv_offloading_size_gb: float | None
    n_gpu_layers: int
    ctx_size: int | None
    served_context: int | None
    tokens_per_s: float
    last_completion_tokens: int
    load_time_s: float | None
    sampler: str
    flashinfer_version: str | None
    adapter_path: str | None
    adapter_name: str | None
    max_lora_rank: int | None


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
    n_gpu_layers: int | None
    mean_power_w: NotRequired[float]
    peak_power_w: NotRequired[float]
    power_samples: NotRequired[int]
    tokens_per_watt: NotRequired[float]
    sampler: NotRequired[str]
    flashinfer_version: NotRequired[str | None]
    gpus: list[GpuSummary]


class VramReclaimReport(TypedDict):
    reclaimed: bool
    residual_mb: int
    polls: int


class ResidentProc(TypedDict):
    pid: int
    used_mb: int


class ContentionReport(TypedDict):
    """Pre-launch VRAM-contention guard outcome."""

    total_mb: int
    free_mb: int
    requested_util: float
    safe_util: float
    target_mb: int
    weight_floor_mb: int
    residents: list[ResidentProc]
    derated: bool
    fits: bool
    action: str
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
    vram_verdict: str | None
    cooldown: CoolDownReport
    gpu: list[GpuSample]


class CellResult(TypedDict):
    cell_key: str
    model: str
    backend: str
    status: str
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
