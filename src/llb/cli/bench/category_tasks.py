"""Summarization and structured-output category benchmark commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, load_config


@app.command("bench-summarization")
def bench_summarization_cmd(
    cases: Path = typer.Option(
        Path("samples/benchmarks/summarization_cases_uk.json"),
        help="summarization cases (JSON array)",
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    judge_model: Optional[str] = typer.Option(
        None,
        help="opt-in gated faithfulness judge (recorded alongside coverage, never the headline)",
    ),
    judge_rho: Optional[float] = typer.Option(
        None, help="calibration Spearman rho; the judge is used only when rho >= threshold (0.6)"
    ),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of the judge endpoint"
    ),
    data_verified: bool = typer.Option(
        False,
        help="stamp the run as human verification gate-verified for composite-headline eligibility",
    ),
    verification_ref: Optional[str] = typer.Option(
        None,
        help="path or label for the verification worksheet, sample manifest, or accepted ledger",
    ),
) -> None:
    """Score summaries by pinned-embedder reference coverage under TIER_SUMMARIZATION."""
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.bench.summarization import SummarizationRun, run_summarization
    from llb.bench.summarization_io import load_cases_file

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    sum_cases = load_cases_file(cases)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> SummarizationRun:
        return run_summarization(
            sum_cases,
            model=model,
            backend=backend,
            complete=complete,
            judge_model=judge_model,
            judge_rho=judge_rho,
            judge_base_url=judge_base_url,
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
    typer.echo(f"[bench-summarization] reference-coverage={result.result.objective_score:.3f}")
    if result.faithfulness is not None:
        typer.echo(f"[bench-summarization] faithfulness (gated judge)={result.faithfulness:.3f}")
    _echo_throughput("bench-summarization", meter)
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-summarization] manifest -> {result.paths['manifest']}")
