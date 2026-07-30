"""CLI: sweep the three agent context-policy constants and pin or expose each."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, load_config


@app.command("bench-agentic-context-sweep")
def bench_agentic_context_sweep_cmd(
    tasks: Path = typer.Option(
        Path("samples/benchmarks/agentic_tasks_uk.json"), help="agentic task set (JSON array)"
    ),
    model: str = typer.Option(..., help="candidate model id (Ollama tag or HF repo id)"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL of a running endpoint (skips launching)"
    ),
    max_steps: int = typer.Option(6, min=1, help="step budget per task"),
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
    """Sweep observation_cap_chars / observation_head_share / keep_last_n and pin or expose each.

    Holds the model and task set fixed, walks the three one-dimensional grids, pairs every
    non-shipped cell against the shipped default, and prints a pin / expose / inapplicable
    verdict per constant. Does not rewrite the shipped defaults.
    """
    from llb.bench.agentic.context_budget import fixed_budget, resolve_context_budget
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.agentic_context_sweep import run_constant_sweep
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend

    cfg = load_config(None, model=model, backend=backend, max_model_len=max_model_len)
    task_set = load_tasks_file(tasks)
    if max_prompt_chars is not None:
        budget = fixed_budget(max_prompt_chars)
    else:
        ollama_num_ctx = cfg.max_model_len or cfg.context_budget
        if cfg.backend == "ollama" and ollama_num_ctx:
            from llb.backends.ollama import OllamaLauncher
            from llb.backends.served_window import is_ollama_base_url, native_root

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

    def run(complete: LLMComplete):
        return run_constant_sweep(
            task_set,
            model=model,
            backend=backend,
            complete=complete,
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
        f"[bench-agentic-context-sweep] model={model} "
        f"prompt-budget={budget.max_prompt_chars or 'unbounded'} chars "
        f"source={budget.budget_source} "
        f"settings={len(result.settings)} verdicts="
        + ",".join(f"{v.axis}={v.verdict}" for v in result.verdicts)
    )
    _echo_throughput("bench-agentic-context-sweep", meter)
    typer.echo(result.table)
