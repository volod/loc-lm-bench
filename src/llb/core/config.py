"""Canonical run configuration for loc-lm-bench.

One `RunConfig` object flows through the whole vertical: it parameterizes the RAG store,
the eval graph, the scoring, and is recorded verbatim in the run manifest. That single
source keeps a run reproducible -- every knob that affects a score lives here and is
serialized into the manifest.

Defaults target the compile-free RAG core: a small (prebuilt) Ollama model
behind its OpenAI-compatible endpoint, a pinned multilingual embedding, deterministic
decoding. "Compile-free" means no vLLM/flash-attn source build -- the GPU is still used.
Load from YAML with `RunConfig.load(path)`; unset fields fall back to these defaults.
"""

import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llb.core.contracts import JsonObject
from llb.core import env
from llb.core.paths import load_project_env, resolve_data_dir, resolve_project_path

Strategy = Literal[
    "fixed", "sentence", "recursive", "markdown", "semantic", "page", "heading", "late"
]
RetrievalMode = Literal["flat", "parent_child", "hybrid"]
RetrievalBackend = Literal["faiss", "graph"]
RetrievalStrategy = Literal["local_khop", "global_community"]
# Context-order policy (rerank-context-order): how kept chunks are laid into the prompt.
# "rank" = best-first (retrieval/rerank order); "reverse_rank" = best-last.
ContextOrder = Literal["rank", "reverse_rank"]
Backend = Literal["ollama", "vllm", "llamacpp"]

# Pinned UA-capable embedding (Premise 4: validated + pinned, never an Optuna knob).
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# Hybrid retrieval defaults (hybrid-retrieval-uk): dense-vs-lexical RRF weight and the
# per-side candidate depth fed into the fusion.
DEFAULT_FUSION_WEIGHT = 0.5
DEFAULT_FUSION_CANDIDATES = 50
# Reranking defaults (rerank-context-order): candidate pool depth fed into the optional
# cross-encoder before the top_k cut. The default reranker model id lives in `llb.rag.rerank`.
DEFAULT_RERANK_CANDIDATES = 30
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_VLLM_HOST = "http://localhost:8000"
DEFAULT_LLAMACPP_HOST = "http://localhost:8080"
RUN_EVAL_METHOD = "run-eval"


def _environment_value(name: str, default: str) -> str:
    load_project_env()
    return os.environ.get(name, default)


def _optional_environment_value(name: str) -> str | None:
    load_project_env()
    return os.environ.get(name) or None


def _validate_chunk_sizes(
    chunk_overlap: int,
    chunk_size: int,
    retrieval_mode: RetrievalMode,
    child_chunk_size: int,
) -> None:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if retrieval_mode == "parent_child" and chunk_overlap >= child_chunk_size:
        raise ValueError("chunk_overlap must be smaller than child_chunk_size")


def _validate_query_prep(steps: list[str]) -> None:
    """Reject unknown or duplicated query-prep step names (uk-query-processing)."""
    if not steps:
        return
    from llb.rag.query_prep import QUERY_PREP_STEPS

    unknown = [step for step in steps if step not in QUERY_PREP_STEPS]
    if unknown:
        raise ValueError(
            f"unknown query_prep step(s): {unknown}; choose from {list(QUERY_PREP_STEPS)}"
        )
    if len(set(steps)) != len(steps):
        raise ValueError(f"query_prep steps must be unique, got {steps}")


def _validate_http_endpoint_url(url: str, label: str) -> None:
    endpoint = urlsplit(url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError(f"{label} must be an http(s) URL with a host")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError(f"{label} must not contain credentials, query parameters, or a fragment")


def _validate_vllm_host_matches_port(vllm_host: str, vllm_port: int) -> None:
    try:
        endpoint = urlsplit(vllm_host)
        endpoint_port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError(f"invalid vllm_host: {vllm_host}") from exc
    if not endpoint.scheme or not endpoint.hostname:
        raise ValueError(f"invalid vllm_host: {vllm_host}")
    if endpoint_port != vllm_port:
        raise ValueError(f"vllm_host port ({endpoint_port}) must match vllm_port ({vllm_port})")


class RunConfig(BaseModel):
    """Everything needed to reproduce one (model, config) evaluation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Identity
    run_name: str = Field(default="rag-eval", min_length=1)
    seed: int = 13

    # Model + backend (v1: backend resolved per model; RAG core ships Ollama, backend telemetry adds vLLM)
    model: str = Field(default="llama3.2:3b", min_length=1)
    backend: Backend = "ollama"
    ollama_host: str = Field(
        default_factory=lambda: _environment_value(env.OLLAMA_HOST, DEFAULT_OLLAMA_HOST)
    )
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    n_shot: int = Field(default=0, ge=0)

    # vLLM serving (used when backend == "vllm"). gpu_memory_utilization is recorded so peak
    # VRAM is comparable across runs (vLLM pre-reserves a KV-cache fraction).
    vllm_host: str = Field(
        default_factory=lambda: _environment_value(env.VLLM_HOST, DEFAULT_VLLM_HOST)
    )
    vllm_port: int = Field(default=8000, ge=1, le=65535)
    gpu_memory_utilization: float = Field(default=0.85, gt=0, le=1)
    max_model_len: int | None = Field(default=None, ge=1)
    cpu_offload_gb: float | None = Field(default=None, ge=0)
    kv_offloading_size_gb: float | None = Field(default=None, ge=0)
    dtype: str = "auto"
    quantization: str | None = None
    adapter_path: Path | None = None

    # llama.cpp serving (used when backend == "llamacpp"). The GGUF runs via `llama-server`,
    # splitting layers GPU<->CPU: n_gpu_layers is the offload split (-1 == all on GPU; set it to
    # the planner's gpu_layers for an oversized offload model). The served context reuses
    # max_model_len. The port is parsed from llamacpp_host.
    llamacpp_host: str = Field(
        default_factory=lambda: _environment_value(env.LLAMACPP_HOST, DEFAULT_LLAMACPP_HOST)
    )
    n_gpu_layers: int = Field(default=-1, ge=-1)

    # Telemetry: when set, run-eval also measures steady-state tokens/sec + peak VRAM on a
    # fixed prompt set and records it in the manifest (needs a running backend; telemetry hook).
    measure_telemetry: bool = False

    # Retrieval (embedding pinned; chunking + top_k are tunable later via Optuna)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    strategy: Strategy = "recursive"
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=5, ge=1)

    # Retrieval mode. "flat" indexes `chunk_size` chunks directly. "parent_child" indexes
    # small `child_chunk_size` children for precise matching but returns their larger parent
    # (the `chunk_size` chunk) for generation context. "hybrid" indexes like "flat" but also
    # builds a lexical BM25 index beside the vector index and fuses the two rankings with
    # weighted RRF at query time (hybrid-retrieval-uk).
    retrieval_mode: RetrievalMode = "flat"
    child_chunk_size: int = Field(default=400, ge=1)

    # Hybrid fusion knobs (used when retrieval_mode == "hybrid"; recorded in the manifest and
    # the sweep cell fingerprint). `fusion_weight` is the dense share of the weighted RRF
    # (1.0 == dense order, 0.0 == lexical order); `fusion_candidates` is the per-side candidate
    # depth fed into the fusion. `lexical_lemmas` opts the lexical side into Ukrainian
    # lemmatization at index AND query time (pymorphy3, the [lex] extra); the stored chunk
    # text is never altered.
    fusion_weight: float = Field(default=DEFAULT_FUSION_WEIGHT, ge=0, le=1)
    fusion_candidates: int = Field(default=DEFAULT_FUSION_CANDIDATES, ge=1)
    lexical_lemmas: bool = False

    # Rerank + context order (rerank-context-order), both recorded in the manifest and the
    # sweep cell fingerprint. `reranker` names a local cross-encoder (HF id; None == off, the
    # default -- see `llb.rag.rerank.DEFAULT_RERANKER` for the pinned candidate);
    # `rerank_candidates` is the retrieved pool depth fed into it before the `top_k` cut.
    # `context_order` lays the kept chunks into the prompt best-first ("rank") or best-last
    # ("reverse_rank"); it applies with or without a reranker.
    reranker: str | None = None
    rerank_candidates: int = Field(default=DEFAULT_RERANK_CANDIDATES, ge=1)
    context_order: ContextOrder = "rank"

    # Query-side processing lane (uk-query-processing): an ORDERED, opt-in list of query-prep
    # steps applied between the user question and retrieval (never mutating the stored corpus).
    # Empty (the default) is an exact no-op. Valid steps: normalize | typos | glossary | rewrite.
    # `query_glossary_path` points at the `query_glossary.json` the glossary step expands from
    # (built with `build-query-glossary`). Both are recorded in the manifest fingerprint.
    query_prep: list[str] = Field(default_factory=list)
    query_glossary_path: Path | None = None

    # Retrieval backend (GraphRAG backend). "faiss" is the default vector store; "graph" selects the GraphRAG
    # knowledge-graph backend (built from the ontology-assisted drafting extraction). `retrieval_strategy` chooses the
    # span-preserving graph strategy: "local_khop" (entity-link + k-hop subgraph) or
    # "global_community" (the narrative layer over offline-detected communities). Both are recorded
    # in the manifest (via the config fingerprint) so graph-vs-FAISS and local-vs-global runs are
    # comparable. `graph_khop_depth` is the local_khop expansion radius.
    retrieval_backend: RetrievalBackend = "faiss"
    retrieval_strategy: RetrievalStrategy = "local_khop"
    graph_khop_depth: int = Field(default=2, ge=1)
    acl_label: str | None = None

    # Judge gating (Premise 2): demoted to diagnostic below the rho threshold. Both default
    # from the environment (JUDGE_MODEL unset -> no judge runs); an explicit value or CLI flag
    # always wins.
    judge_model: str | None = Field(
        default_factory=lambda: _optional_environment_value(env.JUDGE_MODEL)
    )
    judge_base_url: str | None = Field(
        default_factory=lambda: _optional_environment_value(env.DEEPEVAL_JUDGE_BASE_URL)
    )
    judge_threshold: float = Field(default=0.6, ge=-1, le=1)

    # Add a semantic-similarity correctness signal (uses the pinned embedder; recorded,
    # not blended into the headline score). Off by default -- it embeds every answer.
    score_semantic: bool = False

    # Answer-side RAG quality (groundedness-citation-metrics), all additive columns that never
    # change the headline objective. `cited_answers` swaps in the `[i]`-citation generation prompt
    # and scores citation validity + hallucinated-citation rate. `score_groundedness` records the
    # deterministic groundedness fraction (share of answer claims supported by the retrieved
    # context) per case. `insufficient_context_probes` re-runs N sampled gold items with their gold
    # evidence excluded from retrieval and scores abstention accuracy; probe cases are scored
    # separately and never enter the plain correctness aggregates.
    cited_answers: bool = False
    score_groundedness: bool = False
    insufficient_context_probes: int = Field(default=0, ge=0)

    # Paths (resolved against the project / DATA_DIR, never hardcoded)
    data_dir: Path = Path(".data")
    corpus_root: Path = Path(".data/llb/corpus")
    goldset_path: Path = Path(".data/llb/goldset/goldset_uk.jsonl")

    @model_validator(mode="before")
    @classmethod
    def _resolve_paths(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        values = dict(raw)
        data_dir = resolve_data_dir(values.get("data_dir"))
        values["data_dir"] = data_dir
        values["corpus_root"] = (
            resolve_project_path(values["corpus_root"])
            if values.get("corpus_root") is not None
            else data_dir / "llb" / "corpus"
        )
        values["goldset_path"] = (
            resolve_project_path(values["goldset_path"])
            if values.get("goldset_path") is not None
            else data_dir / "llb" / "goldset" / "goldset_uk.jsonl"
        )
        if values.get("query_glossary_path") is not None:
            values["query_glossary_path"] = resolve_project_path(values["query_glossary_path"])
        if values.get("adapter_path") is not None:
            values["adapter_path"] = resolve_project_path(values["adapter_path"])
        return values

    @model_validator(mode="after")
    def _validate_cross_field_constraints(self) -> "RunConfig":
        _validate_chunk_sizes(
            self.chunk_overlap, self.chunk_size, self.retrieval_mode, self.child_chunk_size
        )
        _validate_query_prep(self.query_prep)
        if self.judge_base_url is not None:
            _validate_http_endpoint_url(self.judge_base_url, "judge_base_url")
        if self.backend == "vllm":
            _validate_vllm_host_matches_port(self.vllm_host, self.vllm_port)
        return self

    def index_dir(self) -> Path:
        """Where the built RAG store (chunks + FAISS index) lives for this config."""
        return self.data_dir / "llb" / "rag"

    def graph_dir(self) -> Path:
        """Where the built GraphRAG store (node/edge JSONL + meta) lives for this config (GraphRAG backend)."""
        return self.data_dir / "llb" / "graph"

    def run_dir(self, run_timestamp: str) -> Path:
        """Per-run artifact root: ``$DATA_DIR/run-eval/<run_timestamp>/``."""
        if not run_timestamp or Path(run_timestamp).name != run_timestamp:
            raise ValueError("run_timestamp must be a non-empty path segment")
        return self.data_dir / RUN_EVAL_METHOD / run_timestamp

    def run_staging_dir(self, run_timestamp: str) -> Path:
        """Hidden sibling used until a complete run bundle is atomically published."""
        final_dir = self.run_dir(run_timestamp)
        return final_dir.with_name(f".{final_dir.name}.tmp")

    @classmethod
    def load(cls, path: Path | str) -> "RunConfig":
        """Load a YAML config; missing keys fall back to defaults."""
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"{path}: cannot load config: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.model_validate(data)

    def fingerprint(self) -> JsonObject:
        """The reproducibility-relevant subset, for the run manifest."""
        return self.model_dump(mode="json")

    def with_overrides(self, **overrides: Any) -> "RunConfig":
        """Return a fully revalidated copy with non-None overrides applied."""
        clean = {key: value for key, value in overrides.items() if value is not None}
        if not clean:
            return self
        values = self.model_dump()
        if "data_dir" in clean:
            if self.corpus_root == self.data_dir / "llb" / "corpus":
                values.pop("corpus_root")
            if self.goldset_path == self.data_dir / "llb" / "goldset" / "goldset_uk.jsonl":
                values.pop("goldset_path")
        return type(self).model_validate({**values, **clean})
