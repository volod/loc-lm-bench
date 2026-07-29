"""Agentic category benchmark commands: the task-completion run, the harness comparison,
and the context-management policy comparison.

Three axes over ONE fixed task set + tool world + success checks: `bench-agentic` scores one
cell, `bench-agentic-compare` ranks the HARNESSES, and `bench-agentic-context` ranks the
context-management POLICIES of the loop itself.
"""

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
    from llb.bench.agentic.model import HARNESS_NAMES, AgenticRun
    from llb.bench.agentic.run import load_tasks_file, run_agentic
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend

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
    from llb.bench.agentic.context_budget import resolve_context_budget

    budget = resolve_context_budget(cfg, probe=True)

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
            budget_provenance=budget.provenance(),
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


@app.command("bench-agentic-context")
def bench_agentic_context_cmd(
    tasks: Path = typer.Option(
        Path("samples/benchmarks/agentic_tasks_uk.json"), help="agentic task set (JSON array)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    policies: str = typer.Option(
        "full,observation_cap,keep_last_n,compact",
        help="comma-separated context-management policies to compare (`full` is the baseline)",
    ),
    max_steps: int = typer.Option(6, min=1, help="step budget per task"),
    observation_cap_chars: int = typer.Option(
        800, min=1, help="`observation_cap`: chars kept per observation (head + tail)"
    ),
    keep_last_n: int = typer.Option(3, min=0, help="`keep_last_n`: most-recent steps kept whole"),
    compact_share: float = typer.Option(
        0.5, min=0.0, max=1.0, help="`compact`: share of the usable window that triggers a summary"
    ),
    max_prompt_chars: Optional[int] = typer.Option(
        None,
        help="override the resolved per-step prompt budget (chars); default resolves the served "
        "model's usable window",
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
    """Rank agent-loop context-management policies for one fixed model over one task set."""
    from llb.bench.agentic.context_budget import fixed_budget, resolve_context_budget
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.agentic_context import AgenticContextRun, run_agentic_context
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    task_set = load_tasks_file(tasks)
    policy_list = [p.strip() for p in policies.split(",") if p.strip()]
    if max_prompt_chars is not None:
        budget = fixed_budget(max_prompt_chars)
    else:
        # When Ollama will be asked for an explicit num_ctx, warm the model first so /api/ps
        # reports the window the run will actually serve -- otherwise a stale 4096 resident
        # binds the guard 8x tighter than the backend we are about to request.
        ollama_num_ctx = cfg.max_model_len or cfg.context_budget
        if cfg.backend == "ollama" and ollama_num_ctx:
            from llb.backends.ollama import OllamaLauncher
            from llb.backends.served_window import is_ollama_base_url, native_root

            # Resolve the native host: --base-url may point at the same Ollama daemon's /v1.
            native_host = (
                native_root(base_url)
                if base_url and is_ollama_base_url(base_url, cfg.ollama_host)
                else cfg.ollama_host
            )
            warm = OllamaLauncher(cfg.model, host=native_host, num_ctx=ollama_num_ctx)
            warm.start()
            try:
                warm.ensure_num_ctx(timeout=cfg.request_timeout_s)
            finally:
                warm.stop()
        budget = resolve_context_budget(cfg, probe=True)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> AgenticContextRun:
        return run_agentic_context(
            task_set,
            model=model,
            backend=backend,
            complete=complete,
            policies=policy_list,
            policy_overrides={
                "observation_cap_chars": observation_cap_chars,
                "keep_last_n": keep_last_n,
                "compact_share": compact_share,
            },
            max_steps=max_steps,
            budget=budget,
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
        f"[bench-agentic-context] model={model} policies={','.join(policy_list)} "
        f"prompt-budget={budget.max_prompt_chars or 'unbounded'} chars "
        f"source={budget.budget_source} "
        f"declared={budget.declared_max_model_len or '-'} "
        f"served={budget.served_max_model_len or '-'}"
    )
    _echo_throughput("bench-agentic-context", meter)
    typer.echo(result.table)
    typer.echo(result.recommendation)
    for report in result.reports:
        if report.paths is not None:
            typer.echo(
                f"[bench-agentic-context]   {report.policy:<16} -> {report.paths['manifest']}"
            )


@app.command("bench-agentic-context-compare")
def bench_agentic_context_compare_cmd(
    model: str = typer.Option(..., help="the candidate model to compare across context policies"),
) -> None:
    """Rank one model's persisted agent context-policy runs (full/observation_cap/...).

    Reads the `agentic-context` bundles, keeps the best run per (model, policy), and ranks the
    policies under TIER_AGENTIC -- the context-policy counterpart of `bench-agentic-compare`."""
    from llb.board.agentic_context import agentic_context_comparison

    cfg = load_config(None)
    rows, table, policies = agentic_context_comparison(cfg.data_dir, model)
    if not rows:
        typer.echo(
            f"[bench-agentic-context-compare] no context-policy runs for model '{model}' under "
            f"{cfg.data_dir}; run `llb bench-agentic-context --model {model} ...` first"
        )
        raise typer.Exit(code=2)
    typer.echo(
        f"[bench-agentic-context-compare] model={model} policies={', '.join(sorted(set(policies)))}"
    )
    typer.echo(table)


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
