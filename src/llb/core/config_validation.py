"""Focused config validation implementation."""

import os
from typing import Literal
from urllib.parse import urlsplit
from llb.core.paths import load_project_env

RetrievalMode = Literal["flat", "parent_child", "hybrid"]

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

DEFAULT_FUSION_WEIGHT = 0.5

DEFAULT_FUSION_CANDIDATES = 50

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
    from llb.rag.query_prep.base import QUERY_PREP_STEPS

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
