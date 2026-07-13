"""CLI endpoint-plan construction, vLLM launch, and frontier consent."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import typer

from llb.cli.helpers import cli_error


@dataclass
class _VllmLaunchOptions:
    port: int
    gpu_memory_utilization: float
    max_model_len: Optional[int]
    cpu_offload_gb: Optional[float]
    kv_offloading_size_gb: Optional[float]
    dtype: str
    quantization: Optional[str]
    startup_timeout: float


def _vllm_host_for_port(default_host: str, port: int) -> str:
    parsed = urlsplit(default_host)
    return f"{parsed.scheme or 'http'}://{parsed.hostname or 'localhost'}:{port}"


def _launch_draft_vllm(model: str, options: _VllmLaunchOptions, log_dir: Path) -> tuple[Any, str]:
    from llb.backends.vllm import VllmLauncher
    from llb.core.config import DEFAULT_VLLM_HOST

    host = _vllm_host_for_port(DEFAULT_VLLM_HOST, options.port)
    launcher = VllmLauncher(
        model,
        host=host,
        port=options.port,
        gpu_memory_utilization=options.gpu_memory_utilization,
        max_model_len=options.max_model_len,
        cpu_offload_gb=options.cpu_offload_gb,
        kv_offloading_size_gb=options.kv_offloading_size_gb,
        dtype=options.dtype,
        quantization=options.quantization,
        startup_timeout=options.startup_timeout,
        log_dir=log_dir,
    )
    typer.echo(
        f"[prepare-goldset-draft] starting vLLM model={model} host={host} port={options.port}"
    )
    launcher.start()
    return launcher, f"{host}/v1"


def _endpoint_config_setup(
    model: str,
    endpoint: str,
    backend: str,
    base_url: Optional[str],
    out_dir: Optional[Path],
    num_ctx: Optional[int],
    vllm_options: _VllmLaunchOptions,
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
    no_think: bool,
    egress_consent: bool = False,
    max_usd: float | None = None,
    max_calls: int | None = None,
) -> tuple[Any, Any, Optional[Path]]:
    from llb.prep.ontology.endpoint_config import (
        DEFAULT_LOCAL_BASE_URL,
        ENDPOINT_FRONTIER,
        ENDPOINT_LOCAL,
        LOCAL_BACKEND_OLLAMA,
        LOCAL_BACKEND_VLLM,
        EndpointConfig,
    )
    from llb.prep.ontology.pipeline.journaling import default_out_dir

    resolved_out = out_dir
    resolved_url = base_url or DEFAULT_LOCAL_BASE_URL
    launcher = None
    if endpoint == ENDPOINT_LOCAL and backend == LOCAL_BACKEND_VLLM and base_url is None:
        resolved_out = resolved_out or default_out_dir()
        launcher, resolved_url = _launch_draft_vllm(model, vllm_options, resolved_out / "vllm")
    try:
        config = EndpointConfig(
            kind=endpoint,
            model=model,
            backend=LOCAL_BACKEND_OLLAMA if endpoint == ENDPOINT_FRONTIER else backend,
            base_url=resolved_url,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            think=False if no_think else None,
            num_ctx=None if backend == LOCAL_BACKEND_VLLM else num_ctx,
            egress_consent=egress_consent,
            max_usd=max_usd,
            max_calls=max_calls,
        )
    except ValueError as exc:
        if launcher is not None:
            launcher.stop()
        cli_error(str(exc))
    return config, launcher, resolved_out


def _endpoint_plan_setup(
    model: str,
    endpoint: str,
    backend: str,
    base_url: Optional[str],
    out_dir: Optional[Path],
    num_ctx: Optional[int],
    vllm_options: _VllmLaunchOptions,
    *,
    frontier_stage: str,
    local_model: Optional[str],
    max_tokens: int,
    temperature: float,
    timeout: float,
    no_think: bool,
    egress_consent: bool,
    max_usd: float | None,
    max_calls: int | None,
) -> tuple[Any, Any, Optional[Path]]:
    from llb.prep.ontology.endpoint_config import ENDPOINT_FRONTIER, EndpointPlan

    if frontier_stage not in ("extraction", "drafting", "both"):
        cli_error("--frontier-stage must be extraction, drafting, or both")
    primary, launcher, resolved_out = _endpoint_config_setup(
        model,
        endpoint,
        backend,
        base_url,
        out_dir,
        num_ctx,
        vllm_options,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        no_think=no_think,
        egress_consent=egress_consent,
        max_usd=max_usd,
        max_calls=max_calls,
    )
    if endpoint != ENDPOINT_FRONTIER or frontier_stage == "both":
        return EndpointPlan.single(primary), launcher, resolved_out
    if not local_model:
        cli_error("mixed frontier routing requires --local-model")
    local, local_launcher, resolved_out = _endpoint_config_setup(
        local_model,
        "local",
        backend,
        base_url,
        resolved_out,
        num_ctx,
        vllm_options,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        no_think=no_think,
    )
    plan = (
        EndpointPlan(extraction=primary, drafting=local)
        if frontier_stage == "extraction"
        else EndpointPlan(extraction=local, drafting=primary)
    )
    return plan, local_launcher, resolved_out


def _confirm_frontier_egress(corpus_root: Path, model: str) -> None:
    prompt = f"Send corpus '{corpus_root}' to Litellm destination '{model}' for frontier drafting?"
    if not typer.confirm(prompt, default=False):
        cli_error("frontier egress was not approved; no provider call was made")
