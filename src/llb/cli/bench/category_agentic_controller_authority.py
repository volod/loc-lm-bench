"""CLI entry point for structural controller-channel authority evidence."""

import json
from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-loop-controller-channel-authority")
def bench_agentic_loop_controller_channel_authority_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_controller_channel_authority_design.json"),
        "--design",
        help="immutable authority text, role serialization, seeds, and gates",
    ),
    tasks_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_controller_channel_authority.json"),
        "--tasks",
        help="fresh balanced controller-channel holdout ledger",
    ),
) -> None:
    """Compare identical authority text as observation versus controller message."""
    from llb.backends.ollama import OllamaLauncher, list_models
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.agentic_controller_authority import (
        analyze_channel_authority,
        validate_channel_authority_design,
    )
    from llb.bench.agentic_controller_authority_model import ChannelSeedRun
    from llb.bench.agentic_controller_authority_report import (
        format_channel_authority_table,
        persist_channel_authority,
    )
    from llb.bench.agentic_controller_authority_run import run_channel_authority_seed
    from llb.bench.agentic_loop_policy_power import load_repeat_power_design
    from llb.bench.common_backend import ThroughputMeter, launcher_chat
    from llb.cli.bench._agent_context import resolve_agent_context_budget
    from llb.cli.helpers import cli_error, load_config

    design = load_repeat_power_design(design_path)
    tasks = load_tasks_file(tasks_path)
    try:
        validate_channel_authority_design(design, tasks)
    except ValueError as exc:
        cli_error(str(exc))
    roster = cast(list[dict[str, object]], design["roster"])
    if any(row["backend"] != "ollama" for row in roster):
        cli_error("the committed controller-channel evidence roster requires Ollama")
    sampling = cast(dict[str, object], design["sampling"])
    temperature = float(cast(float, sampling["temperature"]))
    max_tokens = int(cast(int, sampling["max_tokens"]))
    max_model_len = int(cast(int, design["max_model_len"]))
    first_model = cast(str, roster[0]["model"])
    host_cfg = load_config(
        None,
        model=first_model,
        backend="ollama",
        max_model_len=max_model_len,
    )
    installed = set(list_models(host_cfg.ollama_host))
    missing = [str(row["model"]) for row in roster if str(row["model"]) not in installed]
    if missing:
        cli_error(
            f"controller-channel model is not installed at {host_cfg.ollama_host}: {missing[0]}"
        )
    fixed = cast(dict[str, object], design["fixed_policy"])
    seed_runs: list[ChannelSeedRun] = []
    for roster_row in roster:
        model = cast(str, roster_row["model"])
        backend = cast(str, roster_row["backend"])
        for seed in cast(list[int], design["run_seeds"]):
            typer.echo(f"[controller-channel] model={model} seed={seed} temperature={temperature}")
            cfg = load_config(
                None,
                model=model,
                backend=backend,
                max_model_len=max_model_len,
                seed=seed,
                temperature=temperature,
            )
            budget = resolve_agent_context_budget(cfg, base_url=None, max_prompt_chars=None)
            meter = ThroughputMeter()
            launcher = OllamaLauncher(
                model,
                host=cfg.ollama_host,
                num_ctx=max_model_len,
                seed=seed,
            )
            with launcher:
                chat = launcher_chat(
                    launcher,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=cfg.request_timeout_s,
                    meter=meter,
                )
                seed_run = run_channel_authority_seed(
                    tasks,
                    seed=seed,
                    model=model,
                    backend=backend,
                    chat=chat,
                    budget=budget,
                    max_steps=int(cast(int, fixed["max_steps"])),
                    design=design,
                    data_dir=cfg.data_dir,
                    meter=meter,
                )
            if meter.calls == 0:
                raise RuntimeError(f"backend returned no generations for {model} seed {seed}")
            seed_runs.append(seed_run)
            typer.echo(
                f"[controller-channel] model={model} seed={seed} "
                f"throughput={meter.tokens_per_s:.1f} tok/s"
            )

    analysis = analyze_channel_authority(design, seed_runs)
    table = format_channel_authority_table(analysis)
    paths = persist_channel_authority(design, analysis, data_dir=host_cfg.data_dir, table=table)
    typer.echo(table)
    typer.echo(json.dumps(analysis, indent=2, sort_keys=True))
    typer.echo(f"[controller-channel] aggregate -> {paths['manifest']}")
