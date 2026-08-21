"""Shared orchestration for seeded task-neutral repeat-feedback studies."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import typer

DesignValidator = Callable[..., None]
StudyAnalyzer = Callable[..., dict[str, object]]
StudyPersister = Callable[..., Any]


def run_neutral_feedback_study(
    design_path: Path,
    tasks_path: Path,
    *,
    data_verified: bool,
    verification_ref: str | None,
    validate_design: DesignValidator,
    analyze: StudyAnalyzer,
    persist: StudyPersister,
    log_label: str,
) -> None:
    """Run one predeclared Gemma notice across its exact two-seed grid."""
    from llb.backends.ollama import list_models
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.loop_feedback.transfer import FeedbackTransferRun
    from llb.bench.loop_feedback.transfer_report import format_feedback_transfer_table
    from llb.bench.loop_policy.run import run_agentic_loop_policy
    from llb.bench.loop_policy.power import load_repeat_power_design
    from llb.bench.loop_policy.report import AgenticLoopPolicyRun
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.bench._agent_context import resolve_agent_context_budget
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    design = load_repeat_power_design(design_path)
    tasks = load_tasks_file(tasks_path)
    try:
        validate_design(design, tasks)
    except ValueError as exc:
        cli_error(str(exc))
    roster = cast(list[dict[str, object]], design["roster"])
    roster_row = roster[0]
    model = cast(str, roster_row["model"])
    backend = cast(str, roster_row["backend"])
    seeds = cast(list[int], design["run_seeds"])
    variants = cast(list[str], design["repeat_feedback_variants"])
    fixed = cast(dict[str, object], design["fixed_policy"])
    sampling = cast(dict[str, object], design["sampling"])
    temperature = float(cast(float, sampling["temperature"]))
    max_model_len = int(cast(int, design["max_model_len"]))
    if model not in set(list_models()):
        cli_error(f"{log_label} Gemma model is not installed: {model}")

    vram_reader, pid_reader = best_effort_gpu_readers()
    seed_runs: list[FeedbackTransferRun] = []
    for seed in seeds:
        typer.echo(f"[{log_label}] model={model} seed={seed} temperature={temperature}")
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
                run_name=f"{design['study_id']}-seed={seed}",
                data_verified=data_verified,
                verification_ref=verification_ref,
                meter=meter,
                repeat_feedback_design=design,
                model_family="gemma",
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
            raise RuntimeError(f"backend returned no Gemma generations for seed {seed}")
        analysis = result.repeat_feedback_analysis
        if analysis is None:
            raise RuntimeError(f"{log_label} run produced no feedback analysis")
        manifests = {
            report.cell.cell_id: report.paths["manifest"]
            for report in result.reports
            if report.paths is not None
        }
        seed_runs.append(FeedbackTransferRun(seed, model, analysis, manifests))
        variant = cast(dict[str, dict[str, object]], analysis["variants"])[variants[1]]
        typer.echo(
            f"[{log_label}] seed={seed} "
            f"completion={cast(float, variant['completion_rate']):.3f} "
            f"response={cast(float, cast(dict[str, object], variant['redirect'])['response_rate']):.3f} "
            f"throughput={meter.tokens_per_s:.1f} tok/s"
        )

    aggregate = analyze(design, seed_runs)
    table = format_feedback_transfer_table(aggregate)
    task_digest = cast(str, cast(dict[str, object], design["reference"])["task_set_digest"])
    paths = persist(
        design,
        aggregate,
        data_dir=load_config(None).data_dir,
        task_digest=task_digest,
        table=table,
    )
    typer.echo(table)
    typer.echo(json.dumps(aggregate, indent=2, sort_keys=True))
    typer.echo(f"[{log_label}] aggregate -> {paths['manifest']}")
