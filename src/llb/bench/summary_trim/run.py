"""Run every workload under both trim strategies for one model family.

Both arms of ONE task run adjacently and which arm goes first alternates with the task index, so
arm order is balanced across the set instead of aligned with the treatment. That is the whole
difference from the study's first reading: what is measured, how cases pair, and how strata are
counted are unchanged, and only the order the serving endpoint saw the episodes in moved.
"""

from dataclasses import dataclass, field
from typing import cast

from llb.backends.context_budget import fixed_budget
from llb.bench.agentic.context_policy import POLICY_COMPACT, ContextPolicy
from llb.bench.context_policy.interleave import ORDER_ALTERNATING, run_arms_interleaved
from llb.bench.context_policy.report import PolicyReport
from llb.bench.context_policy.run import task_set_digest
from llb.bench.summary_trim.design import ARMS, workload_geometry, workloads
from llb.bench.summary_trim.workloads import (
    workload_case_metadata,
    workload_tasks,
)
from llb.bench.common import LLMComplete


@dataclass(slots=True)
class FamilyRun:
    """One family's rows across every workload and arm, plus the reports to persist."""

    model_family: str
    model: str
    backend: str
    rows: list[dict[str, object]] = field(default_factory=list)
    reports: dict[tuple[str, str], PolicyReport] = field(default_factory=dict)
    analysis: dict[str, object] = field(default_factory=dict)
    tokens_per_s: float = 0.0
    # Every episode this family executed, in execution order, with the arm and the position it
    # ran in. It is what proves the schedule was balanced rather than asserted.
    schedule: list[dict[str, object]] = field(default_factory=list)
    order_offset: int = 0


def run_summary_trim_family(
    design: dict[str, object],
    candidate: dict[str, object],
    *,
    complete: LLMComplete,
    order_offset: int = 0,
) -> FamilyRun:
    """Walk every workload with both arms of each task adjacent and their order alternating.

    The rotation CARRIES across workloads, so a workload with an odd task count does not hand its
    remainder to the same arm every time; `order_offset` sets the starting phase, which the CLI
    flips per family so the one leftover first position cancels over the run rather than
    accumulating on the shipped default.
    """
    held = cast(dict[str, object], design["held_fixed"])
    run = FamilyRun(
        model_family=cast(str, candidate["model_family"]),
        model=cast(str, candidate["model"]),
        backend=cast(str, candidate["backend"]),
        order_offset=order_offset,
    )
    offset = order_offset
    for workload in workloads(design):
        tasks = workload_tasks(workload)
        digest = task_set_digest(tasks)
        metadata = workload_case_metadata(workload)
        interleaved = run_arms_interleaved(
            tasks,
            {arm: _policy(workload, held, arm) for arm in ARMS},
            backend=run.backend,
            complete=complete,
            max_steps=int(cast(int, workload["max_steps"])),
            budget=fixed_budget(int(cast(int, workload["max_prompt_chars"]))),
            preserve_memory_markers=bool(held["preserve_memory_markers"]),
            offset=offset,
        )
        offset += len(tasks)
        name = cast(str, workload["workload"])
        order = _order_by_case(interleaved.schedule)
        run.schedule.extend({"workload": name, **row} for row in interleaved.schedule)
        for arm in ARMS:
            report = interleaved.reports[arm]
            run.reports[(name, arm)] = report
            run.rows.append(_arm_row(workload, arm, report, digest, metadata, order))
    return run


def _order_by_case(schedule: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    """Where each (case, arm) episode sat in its task's pair, keyed the way a case row is."""
    return {(cast(str, row["item_id"]), cast(str, row["arm"])): row for row in schedule}


def _policy(workload: dict[str, object], held: dict[str, object], arm: str) -> ContextPolicy:
    geometry = workload_geometry(workload, held)
    return ContextPolicy(
        name=POLICY_COMPACT,
        observation_cap_chars=int(cast(int, geometry["observation_cap_chars"])),
        observation_head_share=float(cast(float, geometry["observation_head_share"])),
        compact_share=float(cast(float, geometry["compact_share"])),
        summary_input_cap=str(cast(str, geometry["summary_input_cap"])),
        summary_trim_strategy=arm,
    )


def _arm_row(
    workload: dict[str, object],
    arm: str,
    report: PolicyReport,
    digest: str,
    metadata: dict[str, dict[str, object]],
    order: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    """One (workload, arm) row: the four compared quantities, per case and in aggregate."""
    cases = [
        {
            "item_id": row["item_id"],
            "success": bool(episode.success),
            "status": episode.status,
            "measured_folds": episode.telemetry.n_compactions,
            # The transcript the FIRST fold offered the summarizer. The two arms are byte-identical
            # up to and including this value, so it is what pairs a case across arms.
            "first_fold_input_chars": (
                episode.telemetry.summary_fold_input_chars[0]
                if episode.telemetry.summary_fold_input_chars
                else 0
            ),
            "summary_input_chars": episode.telemetry.summary_input_chars,
            "summary_input_elided_chars": episode.telemetry.summary_input_elided_chars,
            "summary_prompt_chars": episode.telemetry.compaction_prompt_chars,
            "model_input_prompt_chars": episode.telemetry.model_input_prompt_chars,
            "n_steps": episode.n_steps,
            # Where this episode ran in its task's pair, so a reading can ask whether an outcome
            # tracks the POSITION rather than the arm.
            "order_position": order[(cast(str, row["item_id"]), arm)]["position"],
            "first_arm": order[(cast(str, row["item_id"]), arm)]["first_arm"],
            **metadata.get(cast(str, row["item_id"]), {}),
        }
        for row, episode in zip(report.rows, report.episodes, strict=True)
    ]
    return {
        "workload": workload["workload"],
        "arm": arm,
        "arm_order_policy": ORDER_ALTERNATING,
        "n_first_position": sum(1 for case in cases if case["order_position"] == 1),
        "max_prompt_chars": workload["max_prompt_chars"],
        "compact_share": workload["compact_share"],
        "task_set_digest": digest,
        "n_tasks": len(cases),
        "completion": report.result.objective_score,
        "n_context_overflow": report.n_context_overflow,
        "cases": cases,
    }
