"""Model discovery, planning, preparation, and resolution contracts."""

from typing import TYPE_CHECKING

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from llb.backends.hardware import Gpu


class ModelSpec(TypedDict):
    name: str
    backend: str
    source: str
    min_vram_gb: NotRequired[int | float]
    notes: NotRequired[str]
    license: NotRequired[str]
    license_url: NotRequired[str]
    gated: NotRequired[bool]
    params_b: NotRequired[float]
    quant: NotRequired[str]
    bpw: NotRequired[float]
    n_layers: NotRequired[int]
    kv_layers: NotRequired[int]
    kv_dim: NotRequired[int]
    max_context: NotRequired[int]
    sliding_window: NotRequired[int]
    sliding_window_pattern: NotRequired[int]
    vocab_size: NotRequired[int]
    hidden_size: NotRequired[int]
    tie_word_embeddings: NotRequired[bool]
    embed_bpw: NotRequired[float]
    hi_precision_params_b: NotRequired[float]
    sources: NotRequired[dict[str, "str | SourceRecord | list[str | SourceRecord]"]]


class SourceRecord(TypedDict):
    """Per-backend artifact metadata overriding fields on a logical model spec."""

    source: str
    quant: NotRequired[str]
    bpw: NotRequired[float]
    params_b: NotRequired[float]
    n_layers: NotRequired[int]
    kv_layers: NotRequired[int]
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
    quant: NotRequired[str | None]
    gpu_layers: NotRequired[int]
    available: bool
    verdict: str
    runnable: bool
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
