"""Typed inputs for ontology-assisted draft execution."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class DraftRequest:
    """Normalized command options passed to the draft workflow."""

    corpus_root: Optional[Path]
    model: Optional[str]
    resume: Optional[Path]
    endpoint: str
    egress_consent: bool
    frontier_stage: str
    local_model: Optional[str]
    backend: str
    base_url: Optional[str]
    max_items: int
    doc_limit: Optional[int]
    seed: int
    extractor: str
    spacy_model: str
    max_tokens: int
    extract_max_chars: Optional[int]
    extract_chunk_overlap: Optional[int]
    concurrency: int
    temperature: float
    timeout: float
    max_usd: Optional[float]
    max_calls: Optional[int]
    no_think: bool
    num_ctx: Optional[int]
    vllm_port: int
    vllm_gpu_memory_utilization: float
    vllm_max_model_len: Optional[int]
    vllm_cpu_offload_gb: Optional[float]
    vllm_kv_offloading_size_gb: Optional[float]
    vllm_dtype: str
    vllm_quantization: Optional[str]
    vllm_startup_timeout: float
    out_dir: Optional[Path]
    verification_sample_size: int
    retrieval_index_dir: Optional[Path]
    retrieval_k: int
    drop_nonretrievable_needles: bool
    coverage_target: Optional[int]
    multi_hop: bool
    chains: bool
    multi_hop_max_paths: int
    multi_hop_bridge_fill: bool
    dedup_against: Optional[str]
    graph_dir: Optional[Path]
    rejection_feedback: Optional[Path]
    require_passed_gates: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DraftRequest":
        payload = {field.name: values[field.name] for field in fields(cls)}
        return cls(**payload)
