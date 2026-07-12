"""Agentic, summarization, and structured-output category benchmark commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, load_config


@app.command("bench-agentic")
def bench_agentic_cmd(
    tasks: Path = typer.Option(
        Path("samples/benchmarks/agentic_tasks_uk.json"), help="agentic task set (JSON array)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_steps: int = typer.Option(6, min=1, help="step budget per task"),
    harness: str = typer.Option(
        "loop",
        help="agentic harness: loop (pure) | langgraph ([eval] extra) | crewai ([crewai] extra). "
        "The comparison axis under TIER_AGENTIC; task set + scoring + judge are held fixed.",
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    judge_model: Optional[str] = typer.Option(
        None,
        help="opt-in gated trajectory-quality judge (recorded alongside completion, never the "
        "headline)",
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
    """Score a model's task completion in the deterministic tool-world under TIER_AGENTIC."""
    from llb.bench.agentic import HARNESS_NAMES, AgenticRun, load_tasks_file, run_agentic
    from llb.bench.common import LLMComplete, ThroughputMeter, drive_with_backend

    if harness not in HARNESS_NAMES:
        typer.echo(
            f"[error] unknown --harness '{harness}'; choose one of {', '.join(HARNESS_NAMES)}",
            err=True,
        )
        raise typer.Exit(code=2)
    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    task_set = load_tasks_file(tasks)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> AgenticRun:
        return run_agentic(
            task_set,
            model=model,
            backend=backend,
            complete=complete,
            max_steps=max_steps,
            harness_name=harness,
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
    typer.echo(
        f"[bench-agentic] harness={harness} "
        f"completion-rate={result.result.objective_score:.3f} "
        f"mean-steps={result.mean_steps:.2f} mean-tool-calls={result.mean_tool_calls:.2f}"
    )
    if result.trajectory_quality is not None:
        typer.echo(
            f"[bench-agentic] trajectory-quality (gated judge)={result.trajectory_quality:.3f}"
        )
    if result.judge_diagnostics is not None:
        diag = result.judge_diagnostics
        typer.echo(
            f"[bench-agentic] judge-diagnostics ok={diag['n_ok']} zero={diag['n_zero']} "
            f"reasons={diag['reasons'] or '{}'}"
        )
    _echo_throughput("bench-agentic", meter)
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-agentic] manifest -> {result.paths['manifest']}")


@app.command("bench-agentic-compare")
def bench_agentic_compare_cmd(
    model: str = typer.Option(..., help="the candidate model to compare across harnesses"),
) -> None:
    """Rank one model's agentic runs across its harnesses (loop/langgraph/crewai).

    Reads the persisted `agentic` run bundles, keeps the best run per (model, harness), and ranks
    the harnesses for the chosen model under TIER_AGENTIC -- isolating the harness effect with the
    same bootstrap CIs as the category boards."""
    from llb.board.harnesses import harness_comparison

    cfg = load_config(None)
    rows, table, harnesses = harness_comparison(cfg.data_dir, model)
    if not rows:
        typer.echo(
            f"[bench-agentic-compare] no agentic runs for model '{model}' under {cfg.data_dir}; "
            "run `llb bench-agentic --harness loop|langgraph|crewai ...` first"
        )
        raise typer.Exit(code=2)
    typer.echo(
        f"[bench-agentic-compare] model={model} harnesses={', '.join(sorted(set(harnesses)))}"
    )
    typer.echo(table)


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
    from llb.bench.common import LLMComplete, ThroughputMeter, drive_with_backend
    from llb.bench.summarization import SummarizationRun, load_cases_file, run_summarization

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
    from llb.bench.common import LLMComplete, ThroughputMeter, drive_with_backend
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
