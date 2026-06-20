"""Canonical run configuration for loc-lm-bench.

One `RunConfig` object flows through the whole vertical: it parameterizes the RAG store,
the eval graph, the scoring, and is recorded verbatim in the run manifest. That single
source keeps a run reproducible -- every knob that affects a score lives here and is
serialized into the manifest.

Defaults target the compile-free Milestone 1 skeleton: a small (prebuilt) Ollama model
behind its OpenAI-compatible endpoint, a pinned multilingual embedding, deterministic
decoding. "Compile-free" means no vLLM/flash-attn source build -- the GPU is still used.
Load from YAML with `RunConfig.load(path)`; unset fields fall back to these defaults.
"""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Strategy = Literal["fixed", "sentence", "recursive", "markdown", "semantic"]
RetrievalMode = Literal["flat", "parent_child"]
Backend = Literal["ollama", "vllm", "llamacpp"]

# Pinned UA-capable embedding (Premise 4: validated + pinned, never an Optuna knob).
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class RunConfig(BaseModel):
    """Everything needed to reproduce one (model, config) evaluation."""

    # Identity
    run_name: str = "m1-skeleton"
    seed: int = 13

    # Model + backend (v1: backend resolved per model; M1 ships Ollama, M2 adds vLLM)
    model: str = "llama3.2:3b"
    backend: Backend = "ollama"
    ollama_host: str = Field(default_factory=lambda: os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST))
    request_timeout_s: float = 120.0
    max_tokens: int = 512
    temperature: float = 0.0
    n_shot: int = 0  # fixed + recorded across compared models (avoids 0-shot/3-shot mixing)

    # vLLM serving (used when backend == "vllm"). gpu_memory_utilization is recorded so peak
    # VRAM is comparable across runs (vLLM pre-reserves a KV-cache fraction).
    vllm_host: str = "http://localhost:8000"
    vllm_port: int = 8000
    gpu_memory_utilization: float = 0.85  # 16 GB class default (see spec planning table)
    max_model_len: int | None = None
    dtype: str = "auto"
    quantization: str | None = None

    # Telemetry: when set, run-eval also measures steady-state tokens/sec + peak VRAM on a
    # fixed prompt set and records it in the manifest (needs a running backend; M2.2).
    measure_telemetry: bool = False

    # Retrieval (embedding pinned; chunking + top_k are tunable later via Optuna)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    strategy: Strategy = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # Retrieval mode. "flat" indexes `chunk_size` chunks directly. "parent_child" indexes
    # small `child_chunk_size` children for precise matching but returns their larger parent
    # (the `chunk_size` chunk) for generation context.
    retrieval_mode: RetrievalMode = "flat"
    child_chunk_size: int = 400

    # Judge gating (Premise 2): demoted to diagnostic below the rho threshold
    judge_model: str | None = None
    judge_threshold: float = 0.6

    # Add a semantic-similarity correctness signal (uses the pinned embedder; recorded,
    # not blended into the headline score). Off by default -- it embeds every answer.
    score_semantic: bool = False

    # Paths (resolved against the project / DATA_DIR, never hardcoded)
    data_dir: Path = Path(".data")
    corpus_root: Path = Path(".data/llb/corpus")
    goldset_path: Path = Path(".data/llb/goldset/goldset_uk.jsonl")

    def index_dir(self) -> Path:
        """Where the built RAG store (chunks + FAISS index) lives for this config."""
        return self.data_dir / "llb" / "rag"

    def run_dir(self) -> Path:
        """Per-run artifact root: $DATA_DIR/llb/runs/<run_name>/."""
        return self.data_dir / "llb" / "runs" / self.run_name

    @classmethod
    def load(cls, path: Path | str) -> "RunConfig":
        """Load a YAML config; missing keys fall back to defaults."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.model_validate(data)

    def fingerprint(self) -> dict:
        """The reproducibility-relevant subset, for the run manifest."""
        return self.model_dump(mode="json")
