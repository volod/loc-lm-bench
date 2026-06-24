"""Milestone 5 benchmark commands (each category renders under its OWN Tier)."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import best_effort_gpu_readers, load_config


@app.command("bench-text-analysis")
def bench_text_analysis_cmd(
    bundle: Path = typer.Option(
        ..., help="synthetic text-analysis bundle dir (corpus/ + text_analysis_labels.jsonl)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    limit: Optional[int] = typer.Option(None, help="cap the number of documents"),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
    real_corpus: bool = typer.Option(
        False, help="the bundle is a REAL corpus (reported separately from synthetic)"
    ),
) -> None:
    """M5.0/M5.4: score a model's planted-label recovery under TIER_TEXT_ANALYSIS."""
    from llb.bench.common import LLMComplete, drive_with_backend
    from llb.bench.text_analysis import TextAnalysisRun, run_text_analysis

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> TextAnalysisRun:
        return run_text_analysis(
            bundle,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
            limit=limit,
            synthetic=not real_corpus,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-text-analysis] manifest -> {result.paths['manifest']}")


@app.command("bench-security")
def bench_security_cmd(
    cases: Path = typer.Option(
        Path("samples/security_cases_uk.json"), help="security-case set (JSON array)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
) -> None:
    """M5.1: score a model's objective ASR + refusal-appropriateness under TIER_SECURITY."""
    from llb.bench.common import LLMComplete, drive_with_backend
    from llb.bench.security import SecurityRun, load_cases_file, run_security

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    security_cases = load_cases_file(cases)
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> SecurityRun:
        return run_security(
            security_cases,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    s = result.score
    typer.echo(
        f"[bench-security] ASR={s.asr:.3f} defense={s.defense_rate:.3f} "
        f"refusal-appropriateness={s.refusal_appropriateness:.3f} (n_attacks={s.n_attacks})"
    )
    for family, asr in sorted(s.asr_by_family.items()):
        typer.echo(f"[bench-security]   {family:<22} ASR={asr:.3f}")
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-security] manifest -> {result.paths['manifest']}")


@app.command("bench-tooling")
def bench_tooling_cmd(
    catalog: Path = typer.Option(
        Path("samples/tooling_cases_uk.json"), help="tooling bundle (tools + cases JSON)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
) -> None:
    """M5.2: score a model's call-only function-calling correctness under TIER_TOOLING."""
    from llb.bench.common import LLMComplete, drive_with_backend
    from llb.bench.tooling import ToolingRun, load_catalog_file, run_tooling

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    tool_catalog, cases = load_catalog_file(catalog)
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> ToolingRun:
        return run_tooling(
            tool_catalog,
            cases,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    s = result.score
    typer.echo(
        f"[bench-tooling] call-accuracy={s.call_accuracy:.3f} tool-selection="
        f"{s.tool_selection_accuracy:.3f} args-exact={s.argument_exactness:.3f} "
        f"no-hallucinated={s.no_hallucinated_tool_rate:.3f} well-formed={s.well_formed_rate:.3f}"
    )
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-tooling] manifest -> {result.paths['manifest']}")


@app.command("bench-agentic")
def bench_agentic_cmd(
    tasks: Path = typer.Option(
        Path("samples/agentic_tasks_uk.json"), help="agentic task set (JSON array)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_steps: int = typer.Option(6, min=1, help="step budget per task"),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
) -> None:
    """M5.3: score a model's task-completion in the deterministic tool-world under TIER_AGENTIC."""
    from llb.bench.agentic import AgenticRun, load_tasks_file, run_agentic
    from llb.bench.common import LLMComplete, drive_with_backend

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    task_set = load_tasks_file(tasks)
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> AgenticRun:
        return run_agentic(
            task_set,
            model=model,
            backend=backend,
            complete=complete,
            max_steps=max_steps,
            data_dir=cfg.data_dir,
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    typer.echo(
        f"[bench-agentic] completion-rate={result.result.objective_score:.3f} "
        f"mean-steps={result.mean_steps:.2f} mean-tool-calls={result.mean_tool_calls:.2f}"
    )
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-agentic] manifest -> {result.paths['manifest']}")


@app.command("bench-summarization")
def bench_summarization_cmd(
    cases: Path = typer.Option(
        Path("samples/summarization_cases_uk.json"), help="summarization cases (JSON array)"
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
) -> None:
    """M5.4: score summaries by pinned-embedder reference coverage under TIER_SUMMARIZATION."""
    from llb.bench.common import LLMComplete, drive_with_backend
    from llb.bench.summarization import SummarizationRun, load_cases_file, run_summarization

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    sum_cases = load_cases_file(cases)
    vram_reader, pid_reader = best_effort_gpu_readers()

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
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    typer.echo(f"[bench-summarization] reference-coverage={result.result.objective_score:.3f}")
    if result.faithfulness is not None:
        typer.echo(f"[bench-summarization] faithfulness (gated judge)={result.faithfulness:.3f}")
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-summarization] manifest -> {result.paths['manifest']}")


@app.command("bench-structured")
def bench_structured_cmd(
    cases: Path = typer.Option(
        Path("samples/structured_cases_uk.json"), help="structured-output cases (JSON array)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="vLLM/llama.cpp served context window"),
) -> None:
    """M5.4: score JSON-schema conformance + field accuracy under TIER_STRUCTURED."""
    from llb.bench.common import LLMComplete, drive_with_backend
    from llb.bench.structured import StructuredRun, load_cases_file, run_structured

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    st_cases = load_cases_file(cases)
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> StructuredRun:
        return run_structured(
            st_cases, model=model, backend=backend, complete=complete, data_dir=cfg.data_dir
        )

    result = drive_with_backend(
        cfg, run, base_url=base_url, vram_reader=vram_reader, pid_usage_reader=pid_reader
    )
    typer.echo(
        f"[bench-structured] field-accuracy={result.score.field_accuracy:.3f} "
        f"conformance={result.score.conformance_rate:.3f}"
    )
    typer.echo(result.table)
    if result.paths is not None:
        typer.echo(f"[bench-structured] manifest -> {result.paths['manifest']}")


@app.command("bench-reliability")
def bench_reliability_cmd(
    run_dir: Path = typer.Option(..., help="a run bundle dir (scores.parquet / scores.jsonl)"),
) -> None:
    """M5.4: aggregate a run's typed failure taxonomy into a first-class reliability score."""
    from llb.scoring.reliability import read_case_statuses, reliability_report

    report = reliability_report(read_case_statuses(run_dir))
    typer.echo(
        f"[bench-reliability] reliability={report['reliability']:.3f} ({report['n_ok']}/{report['n']})"
    )
    for status, count in report["failures"].items():
        typer.echo(f"[bench-reliability]   {status:<20} {count}")
