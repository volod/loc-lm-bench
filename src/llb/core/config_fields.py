"""Pydantic field schema and type vocabulary for evaluation runs."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llb.core import env
from llb.core.config_validation import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FUSION_CANDIDATES,
    DEFAULT_FUSION_WEIGHT,
    DEFAULT_LLAMACPP_HOST,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_VLLM_HOST,
    RetrievalMode,
    _environment_value,
    _optional_environment_value,
)

Strategy = Literal[
    "fixed", "sentence", "recursive", "markdown", "semantic", "page", "heading", "late"
]
RetrievalBackend = Literal["faiss", "graph", "fused"]
RetrievalStrategy = Literal["local_khop", "global_community"]
# Span-identity policy of graph-vector fusion (fusion-span-overlap-identity): when do a graph
# evidence span and a vector chunk name ONE candidate both lanes vouch for? "exact" requires
# identical (doc_id, char_start, char_end); "overlap" folds a graph span into the vector chunk
# that contains it. See `src/llb/rag/fusion_spans.py`.
SpanIdentity = Literal["exact", "overlap"]
# Context-order policy (rerank-context-order): how kept chunks are laid into the prompt.
# "rank" = best-first (retrieval/rerank order); "reverse_rank" = best-last.
ContextOrder = Literal["rank", "reverse_rank"]
# Context strategy (rag-vs-long-context-ablation): where the prompt's evidence comes from.
# "rag" retrieves; "closed_book" retrieves nothing; "long_context" lays the item's whole source
# document(s) into the prompt.
ContextStrategy = Literal["rag", "closed_book", "long_context"]
Backend = Literal["ollama", "vllm", "llamacpp"]
# Scorer-policy seam: human review, local DeepEval judge, or budget-capped frontier judge.
ScorerPolicy = Literal["human", "local", "frontier"]

# Pinned UA-capable embedding (Premise 4: validated + pinned, never an Optuna knob).
# Hybrid retrieval defaults (hybrid-retrieval-uk): dense-vs-lexical RRF weight and the
# per-side candidate depth fed into the fusion.
# Reranking defaults (rerank-context-order): candidate pool depth fed into the optional
# cross-encoder before the top_k cut. The default reranker model id lives in `llb.rag.rerank`.


class RunConfigFields(BaseModel):
    """Declarative fields shared by run validation and serialization."""

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

    # Retrieval (embedding is pinned by default; multi-objective tune may sample it)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    strategy: Strategy = "recursive"
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=5, ge=1)
    # Explicit token budget coupling top_k / chunk_size / max_model_len (multi-objective tune).
    context_budget: int | None = Field(default=None, ge=1)

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
    # lemmatization at index AND query time (pymorphy3); the stored chunk text is never altered.
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

    # Context strategy (rag-vs-long-context-ablation): a DIAGNOSTIC lane selector, not a ranking
    # policy -- "rag" (the default) is the leaderboard row. "closed_book" sends no context at all,
    # so the score is what the model already knows; "long_context" lays the item's whole source
    # document(s) into the prompt, and skips (never truncates) an item whose document does not fit
    # the model's usable window. Recorded in the manifest fingerprint like every other knob.
    context_strategy: ContextStrategy = "rag"

    # Query-side processing lane (uk-query-processing): an ORDERED, opt-in list of query-prep
    # steps applied between the user question and retrieval (never mutating the stored corpus).
    # Empty (the default) is an exact no-op. Valid steps: normalize | typos | glossary | rewrite.
    # `query_glossary_path` points at the `query_glossary.json` the glossary step expands from
    # (built with `build-query-glossary`). Both are recorded in the manifest fingerprint.
    query_prep: list[str] = Field(default_factory=list)
    query_glossary_path: Path | None = None
    # Morphology guard for the 'typos' step (morphology-aware-typo-guard): when on, an
    # out-of-vocabulary query token pymorphy3 recognizes as a valid Ukrainian word form is left
    # unchanged (it is an inflection for the lemmatization lane, not a misspelling). Off by
    # default so the pure edit-distance behavior remains explicitly selectable.
    query_prep_typo_guard: bool = False

    # Retrieval backend (GraphRAG backend). "faiss" is the default vector store; "graph" selects the GraphRAG
    # knowledge-graph backend (built from the ontology-assisted drafting extraction). `retrieval_strategy` chooses the
    # span-preserving graph strategy: "local_khop" (entity-link + k-hop subgraph) or
    # "global_community" (the narrative layer over offline-detected communities). Both are recorded
    # in the manifest (via the config fingerprint) so graph-vs-FAISS and local-vs-global runs are
    # comparable. `graph_khop_depth` is the local_khop expansion radius.
    retrieval_backend: RetrievalBackend = "faiss"
    retrieval_strategy: RetrievalStrategy = "local_khop"
    graph_khop_depth: int = Field(default=2, ge=1)
    # Graph share of graph-vector RRF when retrieval_backend="fused". Zero is an exact vector
    # passthrough; one is an exact graph passthrough. Recorded in every run/cell fingerprint.
    graph_weight: float = Field(default=0.3, ge=0, le=1)
    # Per-lane candidate depth fed into graph-vector fusion (the hybrid `fusion_candidates`
    # pattern). None (the default) asks each lane for exactly `top_k`, so a graph-only span can
    # enter the fused result only when it is already in the graph lane's own top_k and any graph
    # candidate that enters displaces a vector candidate one-for-one. A value deeper than `top_k`
    # fuses a larger pool and then cuts to `top_k`, so `graph_weight` controls influence on the
    # ranking instead of seats in the result. Values below `top_k` are lifted to `top_k`.
    graph_fusion_candidates: int | None = Field(default=None, ge=1)
    # Span-identity policy the fusion keys candidates by. "exact" (the default) only lets the two
    # lanes reinforce each other when their boundaries match exactly, which a ~40-character graph
    # mention and an ~800-character chunk essentially never do; "overlap" folds the mention into
    # the chunk that contains it, so the pair becomes one candidate both lanes voted for.
    graph_fusion_span_identity: SpanIdentity = "exact"
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

    # Scorer-policy seam: which judge lane run-eval uses. "local" keeps the DeepEval path;
    # "frontier" requires egress consent + a hard budget cap; "human" skips automated judging.
    scorer_policy: ScorerPolicy = "local"
    scorer_egress_consent: bool = False
    frontier_max_usd: float | None = Field(default=None, gt=0)
    frontier_max_calls: int | None = Field(default=None, ge=1)

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
