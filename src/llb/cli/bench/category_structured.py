"""Focused category structured implementation."""

from pathlib import Path
from typing import Optional
import typer
from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, load_config


@app.command("bench-structured")
def bench_structured_cmd(
    cases: Path = typer.Option(
        Path("samples/benchmarks/structured_cases_uk.json"),
        help="structured-output cases (JSON array)",
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    data_verified: bool = typer.Option(
        False,
        help="stamp the run as human verification gate-verified for composite-headline eligibility",
    ),
    verification_ref: Optional[str] = typer.Option(
        None,
        help="path or label for the verification worksheet, sample manifest, or accepted ledger",
    ),
) -> None:
    """Score JSON-schema conformance + field accuracy under TIER_STRUCTURED."""
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.bench.structured import StructuredRun, load_cases_file, run_structured

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    st_cases = load_cases_file(cases)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> StructuredRun:
        return run_structured(
            st_cases,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
            data_verified=data_verified,
            verification_ref=verification_ref,
            meter=meter,
        )

    result = drive_with_backend(
        cfg,
        run,
        base_url=base_url,
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    typer.echo(
        f"[bench-structured] field-accuracy={result.score.field_accuracy:.3f} "
        f"conformance={result.score.conformance_rate:.3f}"
    )
    _echo_throughput("bench-structured", meter)
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-structured] manifest -> {result.paths['manifest']}")
