"""CLI orchestration for the seeded cross-family repeat-feedback study."""

import json
from pathlib import Path
from typing import cast

import typer

from llb.cli.app import app


@app.command("bench-agentic-loop-repeat-feedback-generalization")
def bench_agentic_loop_repeat_feedback_generalization_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_feedback_generalization_design.json"),
        "--design",
        help="predeclared roster, seeds, sampling, and cross-family adoption rule",
    ),
    tasks_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_repeat_power_uk.json"),
        "--tasks",
        help="fixed powered repetition-prone task ledger",
    ),
    data_verified: bool = typer.Option(False, help="stamp human-verified task data"),
    verification_ref: str | None = typer.Option(
        None, help="verification worksheet, sample manifest, or accepted ledger"
    ),
) -> None:
    """Run every predeclared model-family/seed cell and persist one aggregate decision."""
    from llb.backends.ollama import list_models
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.agentic_loop_feedback_generalization import (
        FeedbackSeedRun,
        analyze_feedback_generalization,
        validate_feedback_generalization_design,
    )
    from llb.bench.agentic_loop_feedback_generalization_report import (
        format_feedback_generalization_table,
        persist_feedback_generalization,
    )
    from llb.bench.agentic_loop_policy import run_agentic_loop_policy
    from llb.bench.agentic_loop_policy_power import load_repeat_power_design
    from llb.bench.agentic_loop_policy_report import AgenticLoopPolicyRun
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.bench._agent_context import resolve_agent_context_budget
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    design = load_repeat_power_design(design_path)
    tasks = load_tasks_file(tasks_path)
    try:
        validate_feedback_generalization_design(design, tasks)
    except ValueError as exc:
        cli_error(str(exc))
    roster = cast(list[dict[str, object]], design["roster"])
    seeds = cast(list[int], design["run_seeds"])
    variants = cast(list[str], design["repeat_feedback_variants"])
    fixed = cast(dict[str, object], design["fixed_policy"])
    sampling = cast(dict[str, object], design["sampling"])
    temperature = float(cast(float, sampling["temperature"]))
    max_model_len = int(cast(int, design["max_model_len"]))
    available = set(list_models())
    missing = [cast(str, row["model"]) for row in roster if row["model"] not in available]
    if missing:
        cli_error(f"generalization roster models are not installed: {missing}")

    vram_reader, pid_reader = best_effort_gpu_readers()
    seed_runs: list[FeedbackSeedRun] = []
    for row in roster:
        family = cast(str, row["model_family"])
        model = cast(str, row["model"])
        backend = cast(str, row["backend"])
        for seed in seeds:
            typer.echo(
                f"[feedback-generalization] family={family} model={model} seed={seed} "
                f"temperature={temperature}"
            )
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

            def run(complete: LLMComplete) -> AgenticLoopPolicyRun:
                return run_agentic_loop_policy(
                    tasks,
                    model=model,
                    backend=backend,
                    complete=complete,
                    max_steps=[int(cast(int, fixed["max_steps"]))],
                    malformed_policies=[cast(str, fixed["malformed_call"])],
                    repeated_call_policies=cast(list[str], fixed["repeated_call"]),
                    repeated_feedback_variants=variants,
                    budget=budget,
                    data_dir=cfg.data_dir,
                    run_name=f"{design['study_id']}-{family}-seed={seed}",
                    data_verified=data_verified,
                    verification_ref=verification_ref,
                    meter=meter,
                    repeat_feedback_design=design,
                    model_family=family,
                    run_seed=seed,
                )

            result = drive_with_backend(
                cfg,
                run,
                vram_reader=vram_reader,
                pid_usage_reader=pid_reader,
                meter=meter,
            )
            if meter.calls == 0:
                raise RuntimeError(
                    f"backend returned no successful generations for {family}/{seed}"
                )
            analysis = result.repeat_feedback_analysis
            if analysis is None:
                raise RuntimeError("generalization run produced no repeat-feedback analysis")
            manifests = {
                report.cell.cell_id: report.paths["manifest"]
                for report in result.reports
                if report.paths is not None
            }
            seed_runs.append(FeedbackSeedRun(family, model, seed, analysis, manifests))
            variant = cast(dict[str, dict[str, object]], analysis["variants"])[variants[1]]
            typer.echo(
                f"[feedback-generalization] family={family} seed={seed} "
                f"completion={cast(float, variant['completion_rate']):.3f} "
                f"supports={str(variant['supports_variant']).lower()} "
                f"throughput={meter.tokens_per_s:.1f} tok/s"
            )

    analysis = analyze_feedback_generalization(design, seed_runs)
    table = format_feedback_generalization_table(analysis)
    paths = persist_feedback_generalization(
        design,
        analysis,
        data_dir=load_config(None).data_dir,
        task_digest=cast(str, cast(dict[str, object], design["reference"])["task_set_digest"]),
        table=table,
    )
    typer.echo(table)
    typer.echo(json.dumps(analysis, indent=2, sort_keys=True))
    typer.echo(f"[feedback-generalization] aggregate -> {paths['manifest']}")
