"""The scored agentic cell: one model, one harness, one context policy, one loop policy.

`bench-agentic` scores ONE cell of the agentic axes over the fixed task set + tool world + success
checks. The harness comparison (`bench-agentic-compare`) and the context-policy comparison
(`bench-agentic-context`) rank their own axis; this command runs the configuration those
comparisons -- and the loop-policy sweep -- recommend, so a recommendation is scored rather than
only re-swept.
"""

from pathlib import Path
from typing import Optional

import typer

from llb.bench.agentic.loop_policy import (
    DEFAULT_MALFORMED_POLICY,
    DEFAULT_REPEAT_FEEDBACK,
    DEFAULT_REPEATED_CALL_POLICY,
)
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
    context_policy: str = typer.Option(
        "full",
        help="agent context-management policy transferred onto harnesses that support it "
        "(full|observation_cap|keep_last_n|compact). loop/langgraph apply it; crewai records "
        "unsupported and still reports the prompt sizes it actually sent.",
    ),
    malformed_policy: str = typer.Option(
        DEFAULT_MALFORMED_POLICY,
        help="controller handling of a malformed tool call (answer|repair_once|strict); the cell "
        "`bench-agentic-loop` recommends, run as a scored cell",
    ),
    repeated_call_policy: str = typer.Option(
        DEFAULT_REPEATED_CALL_POLICY,
        help="controller handling of an identical consecutive tool call (allow|noop)",
    ),
    repeat_feedback: str = typer.Option(
        DEFAULT_REPEAT_FEEDBACK,
        help="observation returned for a suppressed repeat; only reaches the prompt under "
        "--repeated-call-policy noop",
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
    from llb.bench.agentic.context_policy import CONTEXT_POLICIES, ContextPolicy
    from llb.bench.agentic.loop_policy import LoopPolicy
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
    if context_policy not in CONTEXT_POLICIES:
        typer.echo(
            f"[error] unknown --context-policy '{context_policy}'; "
            f"choose one of {', '.join(CONTEXT_POLICIES)}",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        loop_policy = LoopPolicy(
            malformed_call=malformed_policy,
            repeated_call=repeated_call_policy,
            repeat_feedback=repeat_feedback,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    task_set = load_tasks_file(tasks)
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()
    from llb.backends.context_budget import resolve_context_budget

    budget = resolve_context_budget(cfg, probe=True)
    policy = ContextPolicy(name=context_policy)

    def run(complete: LLMComplete) -> AgenticRun:
        return run_agentic(
            task_set,
            model=model,
            backend=backend,
            complete=complete,
            max_steps=max_steps,
            harness_name=harness,
            policy=policy,
            budget=budget,
            loop_policy=loop_policy,
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
    supported = all(ep.context_policy_supported for ep in result.episodes)
    loop_supported = all(ep.loop_policy_supported for ep in result.episodes)
    typer.echo(
        f"[bench-agentic] harness={harness} context-policy={context_policy}"
        f"{'' if supported else ' (unsupported by harness)'} "
        f"completion-rate={result.result.objective_score:.3f} "
        f"mean-steps={result.mean_steps:.2f} mean-tool-calls={result.mean_tool_calls:.2f}"
    )
    typer.echo(
        f"[bench-agentic] loop-policy malformed={malformed_policy} "
        f"repeated-call={repeated_call_policy} repeat-feedback={repeat_feedback}"
        f"{'' if loop_supported else ' (unsupported by harness)'}"
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
