"""CLI orchestration for the seeded family-adapted repeat-feedback study."""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from llb.bench.agentic_design_fields import as_float
from llb.cli.app import app

if TYPE_CHECKING:
    from llb.bench.agentic_loop_feedback_adaptation import FeedbackAdaptationRun
    from llb.bench.agentic_loop_policy_report import AgenticLoopPolicyRun
    from llb.bench.common import LLMComplete


@dataclass(frozen=True, slots=True)
class _AdaptationPlan:
    """What the design fixes for every family/seed cell of the family-adaptation study.

    Resolved once from the design so each cell is driven from one record rather than from locals
    captured by a closure -- and so no cell can read a different policy or temperature than its
    neighbour.
    """

    design: dict[str, object]
    roster: list[dict[str, object]]
    seeds: list[int]
    fixed: dict[str, object]
    temperature: float
    max_model_len: int

    @classmethod
    def of(cls, design: dict[str, object]) -> "_AdaptationPlan":
        """Read the predeclared roster, seeds, policy, and sampling out of the design."""
        from llb.bench.agentic_design_fields import as_int, as_ints, as_mapping, as_rows

        return cls(
            design=design,
            roster=as_rows(design, "roster"),
            seeds=as_ints(design, "run_seeds"),
            fixed=as_mapping(design, "fixed_policy"),
            temperature=as_float(as_mapping(design, "sampling"), "temperature"),
            max_model_len=as_int(design, "max_model_len"),
        )


def _run_adaptation_cell(
    complete: "LLMComplete",
    *,
    plan: _AdaptationPlan,
    tasks: list[Any],
    family: str,
    model: str,
    backend: str,
    seed: int,
    candidate: str,
    budget: Any,
    data_dir: Path,
    meter: Any,
    data_verified: bool,
    verification_ref: str | None,
) -> "AgenticLoopPolicyRun":
    """Run one family/seed cell against that family's OWN candidate notice."""
    from llb.bench.agentic_design_fields import as_int, as_str, as_strs
    from llb.bench.agentic_loop_policy import run_agentic_loop_policy

    return run_agentic_loop_policy(
        tasks,
        model=model,
        backend=backend,
        complete=complete,
        max_steps=[as_int(plan.fixed, "max_steps")],
        malformed_policies=[as_str(plan.fixed, "malformed_call")],
        repeated_call_policies=as_strs(plan.fixed, "repeated_call"),
        repeated_feedback_variants=["current", candidate],
        budget=budget,
        data_dir=data_dir,
        run_name=f"{plan.design['study_id']}-{family}-seed={seed}",
        data_verified=data_verified,
        verification_ref=verification_ref,
        meter=meter,
        repeat_feedback_design=plan.design,
        model_family=family,
        run_seed=seed,
    )


def _adaptation_seed_run(
    plan: _AdaptationPlan,
    tasks: list[Any],
    *,
    row: dict[str, object],
    seed: int,
    data_verified: bool,
    verification_ref: str | None,
    readers: tuple[Any, Any],
) -> "FeedbackAdaptationRun":
    """Drive one family/seed cell on a real backend and read the analysis it produced."""
    from llb.bench.agentic_loop_feedback_adaptation import FeedbackAdaptationRun
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.cli.bench._agent_context import resolve_agent_context_budget
    from llb.cli.helpers import load_config

    family = cast(str, row["model_family"])
    model = cast(str, row["model"])
    backend = cast(str, row["backend"])
    candidate = cast(str, row["candidate_feedback_variant"])
    typer.echo(
        f"[feedback-adaptation] family={family} model={model} candidate={candidate} "
        f"seed={seed} temperature={plan.temperature}"
    )
    cfg = load_config(
        None,
        model=model,
        backend=backend,
        max_model_len=plan.max_model_len,
        seed=seed,
        temperature=plan.temperature,
    )
    meter = ThroughputMeter()
    vram_reader, pid_reader = readers
    result = drive_with_backend(
        cfg,
        partial(
            _run_adaptation_cell,
            plan=plan,
            tasks=tasks,
            family=family,
            model=model,
            backend=backend,
            seed=seed,
            candidate=candidate,
            budget=resolve_agent_context_budget(cfg, base_url=None, max_prompt_chars=None),
            data_dir=cfg.data_dir,
            meter=meter,
            data_verified=data_verified,
            verification_ref=verification_ref,
        ),
        vram_reader=vram_reader,
        pid_usage_reader=pid_reader,
        meter=meter,
    )
    if meter.calls == 0:
        raise RuntimeError(f"backend returned no generations for {family}/{seed}")
    analysis = result.repeat_feedback_analysis
    if analysis is None:
        raise RuntimeError("family-adaptation run produced no feedback analysis")
    manifests = {
        report.cell.cell_id: report.paths["manifest"]
        for report in result.reports
        if report.paths is not None
    }
    variant = cast(dict[str, dict[str, object]], analysis["variants"])[candidate]
    typer.echo(
        f"[feedback-adaptation] family={family} seed={seed} "
        f"completion={cast(float, variant['completion_rate']):.3f} "
        f"supports={str(variant['supports_variant']).lower()} "
        f"throughput={meter.tokens_per_s:.1f} tok/s"
    )
    return FeedbackAdaptationRun(family, model, seed, candidate, analysis, manifests)


def _check_roster_installed(roster: list[dict[str, object]]) -> None:
    """Refuse before any run when a declared model is not on this host."""
    from llb.backends.ollama import list_models
    from llb.cli.helpers import cli_error

    available = set(list_models())
    missing = [cast(str, row["model"]) for row in roster if row["model"] not in available]
    if missing:
        cli_error(f"family-adaptation roster models are not installed: {missing}")


@app.command("bench-agentic-loop-repeat-feedback-family-adaptation")
def bench_agentic_loop_repeat_feedback_family_adaptation_cmd(
    design_path: Path = typer.Option(
        Path("samples/benchmarks/agentic_loop_feedback_family_adaptation_design.json"),
        "--design",
        help="predeclared family candidates, wording, seeds, and adoption gates",
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
    """Run each predeclared family candidate on two seeds and persist stable routes."""
    from llb.bench.agentic.run import load_tasks_file
    from llb.bench.agentic_loop_feedback_adaptation import (
        analyze_feedback_adaptation,
        validate_feedback_adaptation_design,
    )
    from llb.bench.agentic_loop_feedback_adaptation_report import (
        format_feedback_adaptation_table,
        persist_feedback_adaptation,
    )
    from llb.bench.agentic_loop_policy_power import load_repeat_power_design
    from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config

    design = load_repeat_power_design(design_path)
    tasks = load_tasks_file(tasks_path)
    try:
        validate_feedback_adaptation_design(design, tasks)
    except ValueError as exc:
        cli_error(str(exc))
    plan = _AdaptationPlan.of(design)
    _check_roster_installed(plan.roster)
    readers = best_effort_gpu_readers()
    seed_runs = [
        _adaptation_seed_run(
            plan,
            tasks,
            row=row,
            seed=seed,
            data_verified=data_verified,
            verification_ref=verification_ref,
            readers=readers,
        )
        for row in plan.roster
        for seed in plan.seeds
    ]
    analysis = analyze_feedback_adaptation(design, seed_runs)
    table = format_feedback_adaptation_table(analysis)
    paths = persist_feedback_adaptation(
        design,
        analysis,
        data_dir=load_config(None).data_dir,
        task_digest=cast(str, cast(dict[str, object], design["reference"])["task_set_digest"]),
        table=table,
    )
    typer.echo(table)
    typer.echo(json.dumps(analysis, indent=2, sort_keys=True))
    typer.echo(f"[feedback-adaptation] aggregate -> {paths['manifest']}")
