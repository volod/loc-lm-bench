"""CLI: active compact versus observation-cap evidence on long transcripts."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, load_config


@app.command("bench-agentic-compact-vs-cap")
def bench_agentic_compact_vs_cap_cmd(
    tasks: Path = typer.Option(..., help="long-transcript agentic task set (JSON array)"),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_steps: int = typer.Option(12, min=1, help="raised step budget per task"),
    observation_cap_chars: int = typer.Option(800, min=1, help="shared live-observation cap"),
    observation_head_share: float = typer.Option(
        0.6, min=0.01, max=0.99, help="shared head share of the live-observation cap"
    ),
    compact_share: float = typer.Option(
        0.5, min=0.01, max=1.0, help="compact trigger as a share of the prompt budget"
    ),
    min_compaction_rate: float = typer.Option(
        0.0,
        min=0.0,
        max=1.0,
        help="predeclared minimum share of compact episodes that must compact",
    ),
    max_prompt_chars: Optional[int] = typer.Option(
        None,
        help="prompt budget override; tighten it when the served window leaves compaction inactive",
    ),
    max_model_len: Optional[int] = typer.Option(None, help="served model context window"),
    data_verified: bool = typer.Option(False, help="stamp human verification-gate provenance"),
    verification_ref: Optional[str] = typer.Option(
        None, help="verification worksheet, manifest, or accepted ledger"
    ),
) -> None:
    """Pair compact directly against observation_cap and fail if compaction never fires."""
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.agentic_compact_vs_cap import (
        run_compact_vs_cap,
    )
    from llb.bench.agentic_compact_vs_cap_report import VERDICT_INACTIVE, CompactVsCapRun
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
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> CompactVsCapRun:
        return run_compact_vs_cap(
            task_set,
            model=model,
            backend=backend,
            complete=complete,
            max_steps=max_steps,
            budget=budget,
            observation_cap_chars=observation_cap_chars,
            observation_head_share=observation_head_share,
            compact_share=compact_share,
            min_compaction_rate=min_compaction_rate,
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
        f"[bench-agentic-compact-vs-cap] model={model} tasks={len(task_set)} "
        f"prompt-budget={budget.max_prompt_chars or 'unbounded'} chars "
        f"compactions={result.n_compactions} "
        f"compacted-episodes={result.n_compacted_episodes}/{len(task_set)} "
        f"required={result.min_compacted_episodes}"
    )
    _echo_throughput("bench-agentic-compact-vs-cap", meter)
    typer.echo(result.table)
    for report in (result.cap, result.compact):
        if report.paths is not None:
            typer.echo(
                f"[bench-agentic-compact-vs-cap] {report.policy} -> {report.paths['manifest']}"
            )
    if result.verdict == VERDICT_INACTIVE:
        raise typer.Exit(code=2)
