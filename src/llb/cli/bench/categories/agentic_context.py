"""The agentic context-management policy comparison.

`bench-agentic-context` ranks the context-management POLICIES of the loop (and any harness that
applies them) for one fixed model over one task set, holding the task set, tool world, scoring, and
judge fixed so the policy is the only variable.
"""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, load_config


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
        800,
        min=1,
        help="chars kept per observation under `observation_cap` and under `compact` live steps",
    ),
    observation_head_share: float = typer.Option(
        0.6,
        min=0.01,
        max=0.99,
        help="fraction of the observation-cap budget kept from the HEAD (rest from the tail)",
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
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.context_policy.run import AgenticContextRun, run_agentic_context
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.bench._agent_context import resolve_agent_context_budget

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    task_set = load_tasks_file(tasks)
    policy_list = [p.strip() for p in policies.split(",") if p.strip()]
    budget = resolve_agent_context_budget(
        cfg,
        base_url=base_url,
        max_prompt_chars=max_prompt_chars,
    )
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
                "observation_head_share": observation_head_share,
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
