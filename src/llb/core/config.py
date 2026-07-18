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

from pathlib import Path
from typing import Any

import yaml
from pydantic import model_validator

from llb.core.contracts.common import JsonObject
from llb.core.paths import resolve_data_dir, resolve_project_path
from llb.core.config_validation import (
    RUN_EVAL_METHOD,
    _validate_chunk_sizes,
    _validate_http_endpoint_url,
    _validate_query_prep,
    _validate_vllm_host_matches_port,
)

from llb.core.config_fields import RunConfigFields


def _validate_scorer_policy(config: RunConfigFields) -> None:
    """Frontier lane needs a model and a budget; consent is checked at resolve time."""
    if config.scorer_policy == "frontier":
        if not config.judge_model:
            raise ValueError("scorer_policy=frontier requires judge_model")
        if config.frontier_max_usd is None and config.frontier_max_calls is None:
            raise ValueError("scorer_policy=frontier requires frontier_max_usd or frontier_max_calls")
        return
    if config.scorer_egress_consent:
        raise ValueError("scorer_egress_consent can only be set when scorer_policy is frontier")
    if config.frontier_max_usd is not None or config.frontier_max_calls is not None:
        raise ValueError("frontier budgets can only be set when scorer_policy is frontier")


class RunConfig(RunConfigFields):
    """Validated behavior and artifact paths for one evaluation run."""

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
        if self.query_prep_typo_guard and "typos" not in self.query_prep:
            raise ValueError("query_prep_typo_guard needs the 'typos' step in query_prep")
        if self.judge_base_url is not None:
            _validate_http_endpoint_url(self.judge_base_url, "judge_base_url")
        _validate_scorer_policy(self)
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
