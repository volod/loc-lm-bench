"""Helpers for the prepare-goldset-draft command: vLLM launch, endpoint setup, resume
overrides, input validation, verification-sample scaffolding, and calibration gating."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast
from urllib.parse import urlsplit

import typer

from llb.cli.helpers import cli_error


@dataclass
class _VllmLaunchOptions:
    """The `--vllm-*` server knobs, grouped so launch logic takes one argument."""

    port: int
    gpu_memory_utilization: float
    max_model_len: Optional[int]
    cpu_offload_gb: Optional[float]
    kv_offloading_size_gb: Optional[float]
    dtype: str
    quantization: Optional[str]
    startup_timeout: float


def _resume_overrides(
    resume: Path,
    corpus_root: Optional[Path],
    model: Optional[str],
    endpoint: str,
    backend: str,
    out_dir: Optional[Path],
) -> tuple[Optional[Path], str, str, str, Path]:
    """Fill unset CLI values from the resumed bundle's journal meta.

    The bundle's journal meta is authoritative for the corpus and endpoint identity; the
    extraction/seed/retrieval settings are re-read inside draft_goldset(resume=True). The
    base URL is intentionally NOT restored so a vLLM resume relaunches a fresh server.
    """
    from llb.prep.ontology import load_journal_meta

    try:
        meta = load_journal_meta(resume)
    except ValueError as exc:
        cli_error(str(exc))
    ep_meta = cast(dict[str, Any], meta.get("endpoint") or {})
    if corpus_root is None:
        corpus_root = Path(str(meta.get("corpus_root")))
    if not model:
        model = str(ep_meta.get("model") or "")
    endpoint = str(ep_meta.get("kind") or endpoint)
    backend = str(ep_meta.get("backend") or backend)
    return corpus_root, model, endpoint, backend, out_dir or resume


def _validate_draft_inputs(
    drop_nonretrievable_needles: bool,
    retrieval_index_dir: Optional[Path],
    graph_dir: Optional[Path],
    rejection_feedback: Optional[Path],
) -> None:
    """Fail fast (exit 2) on option combinations and paths that cannot work."""
    if drop_nonretrievable_needles and retrieval_index_dir is None:
        cli_error("--drop-nonretrievable-needles requires --retrieval-index-dir")
    if retrieval_index_dir is not None and not retrieval_index_dir.is_dir():
        cli_error(f"retrieval index dir not found: {retrieval_index_dir}")
    if graph_dir is not None and not graph_dir.is_dir():
        cli_error(f"graph store dir not found: {graph_dir}")
    if rejection_feedback is not None and not rejection_feedback.is_file():
        cli_error(f"rejection feedback file not found: {rejection_feedback}")


def _launch_draft_vllm(model: str, options: _VllmLaunchOptions, log_dir: Path) -> tuple[Any, str]:
    """Start a vLLM server for the draft run; returns (launcher, base_url)."""
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


def _extraction_adapter(extractor: str, spacy_model: str) -> Any:
    """The opt-in spaCy extraction adapter, or None for the default LLM extractor."""
    if extractor != "spacy":
        return None
    from llb.prep.ontology.spacy_adapter import SpacyExtractionAdapter

    return SpacyExtractionAdapter(model=spacy_model)


def _split_dir_list(value: Optional[str]) -> Optional[list[Path | str]]:
    """Comma-separated dirs -> Path list (None when the option is unset/empty)."""
    if not value:
        return None
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def _draft_endpoint_setup(
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
) -> tuple[Any, Any, Optional[Path]]:
    """Launch vLLM when this command owns the server, then build the endpoint config.

    Returns (endpoint_config, launched_vllm_or_None, resolved_out_dir).
    """
    from llb.prep.ontology import EndpointConfig, default_out_dir
    from llb.prep.ontology.endpoint import (
        DEFAULT_LOCAL_BASE_URL,
        ENDPOINT_LOCAL,
        LOCAL_BACKEND_VLLM,
    )

    resolved_out_dir = out_dir
    base_url_value = base_url or DEFAULT_LOCAL_BASE_URL
    launched_vllm = None
    if endpoint == ENDPOINT_LOCAL and backend == LOCAL_BACKEND_VLLM and base_url is None:
        resolved_out_dir = resolved_out_dir or default_out_dir()
        launched_vllm, base_url_value = _launch_draft_vllm(
            model, vllm_options, resolved_out_dir / "vllm"
        )
    try:
        cfg = EndpointConfig(
            kind=endpoint,
            model=model,
            backend=backend,
            base_url=base_url_value,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            think=False if no_think else None,
            num_ctx=None if backend == LOCAL_BACKEND_VLLM else num_ctx,
        )
    except ValueError as exc:
        if launched_vllm is not None:
            launched_vllm.stop()
        cli_error(str(exc))
    return cfg, launched_vllm, resolved_out_dir


def _write_verification_sample(out_dir: Path, n: int, seed: int) -> None:
    from llb.goldset.verify import build_sample_worksheet

    worksheet = out_dir / "verify_sample.csv"
    sample_size, _strata = build_sample_worksheet(out_dir, worksheet, n=n, seed=seed)
    typer.echo(f"[prepare-goldset-draft] verification sample: {sample_size} rows -> {worksheet}")


def _enforce_calibration_gates(calibration_report: Any, out_dir: Path) -> None:
    """Exit 1 when the required ontology calibration gates failed (--require-passed-gates)."""
    from llb.prep.ontology.artifacts import required_gate_names
    from llb.prep.ontology.constants import PDF_ONTOLOGY_REPORT_FILENAME

    gates = calibration_report.get("gates") if isinstance(calibration_report, dict) else None
    if isinstance(gates, dict) and bool(gates.get("passed")):
        return
    failed: list[str] = []
    if isinstance(gates, dict):
        required = required_gate_names(bool(gates.get("pdf_citation_gate_applicable")))
        failed = [name for name in required if not gates.get(name)]
    detail = ", ".join(failed) if failed else "see report"
    cli_error(
        "ontology calibration gates not passed "
        f"({detail}); inspect {out_dir / PDF_ONTOLOGY_REPORT_FILENAME}",
        code=1,
    )


def _vllm_host_for_port(default_host: str, port: int) -> str:
    parsed = urlsplit(default_host)
    scheme = parsed.scheme or "http"
    hostname = parsed.hostname or "localhost"
    return f"{scheme}://{hostname}:{port}"
