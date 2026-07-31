"""CLI for the paired agent-loop policy sweep."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, load_config


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(item) for item in _csv_strings(value)]
    except ValueError as exc:
        raise typer.BadParameter("expected comma-separated integers") from exc


@app.command("bench-agentic-loop")
def bench_agentic_loop_cmd(
    tasks: Path = typer.Option(
        Path("samples/benchmarks/agentic_tasks_uk.json"), help="agentic task set (JSON array)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    agent_max_steps: str = typer.Option(
        "4,6,10", help="comma-separated step budgets; must include the baseline 6"
    ),
    agent_malformed_policy: str = typer.Option(
        "answer,repair_once,strict",
        help="comma-separated malformed-call policies; must include baseline answer",
    ),
    agent_repeated_call_policy: str = typer.Option(
        "allow,noop",
        help="comma-separated repeated-call policies; must include baseline allow",
    ),
    max_prompt_chars: Optional[int] = typer.Option(
        None, help="override the resolved per-model prompt budget in characters"
    ),
    max_model_len: Optional[int] = typer.Option(None, help="served context window"),
    data_verified: bool = typer.Option(False, help="stamp human-verified task data"),
    verification_ref: Optional[str] = typer.Option(
        None, help="verification worksheet, sample manifest, or accepted ledger"
    ),
) -> None:
    """Sweep loop decision policies and recommend one evidence-backed cell for a fixed model."""
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.agentic_loop_policy import run_agentic_loop_policy
    from llb.bench.agentic_loop_policy_report import AgenticLoopPolicyRun
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.bench._agent_context import resolve_agent_context_budget

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    task_set = load_tasks_file(tasks)
    budget = resolve_agent_context_budget(
        cfg,
        base_url=base_url,
        max_prompt_chars=max_prompt_chars,
    )
    meter = ThroughputMeter()
    vram_reader, pid_reader = best_effort_gpu_readers()

    def run(complete: LLMComplete) -> AgenticLoopPolicyRun:
        return run_agentic_loop_policy(
            task_set,
            model=model,
            backend=backend,
            complete=complete,
            max_steps=_csv_ints(agent_max_steps),
            malformed_policies=_csv_strings(agent_malformed_policy),
            repeated_call_policies=_csv_strings(agent_repeated_call_policy),
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
        f"[bench-agentic-loop] model={model} cells={len(result.reports)} "
        f"tasks={len(task_set)} prompt-budget={budget.max_prompt_chars or 'unbounded'}"
    )
    _echo_throughput("bench-agentic-loop", meter)
    typer.echo(result.table)
    typer.echo(json.dumps(result.recommendation, indent=2, sort_keys=True))
    for report in result.reports:
        if report.paths is not None:
            typer.echo(f"[bench-agentic-loop] {report.cell.cell_id} -> {report.paths['manifest']}")
