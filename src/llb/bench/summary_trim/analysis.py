"""Assemble one adoption verdict from the family runs, the audit, and the provenance scope.

The study is not allowed to recommend a default change on its own measurements. Promoting the
shipped value of a `ContextPolicy` field retires whatever published cells the move changes, so the
model-free policy-change audit runs HERE, under the pinned policy the published numbers stand
under, and its answer is an input to the verdict beside the completion and cost readings. The
registered published values whose arithmetic declares the field are listed the same way, so the
recommendation names what a default change would cost rather than leaving it to be discovered.
"""

from typing import cast

from llb.bench.agentic.context_policy import (
    SUMMARY_TRIM_HEAD_TAIL,
    SUMMARY_TRIM_PER_ENTRY_HEAD,
)
from llb.bench.summary_trim.design import probe_workload, workloads
from llb.bench.summary_trim.workloads import BUILDER_EVIDENCE_STRATA
from llb.bench.summary_trim.adoption import adoption_reading
from llb.bench.summary_trim.reading import family_reading
from llb.bench.summary_trim.run import FamilyRun

POLICY_FIELD = "summary_trim_strategy"


def audit_default_change() -> dict[str, object]:
    """Replay every published cell under both trim strategies, at the pinned policy, with no GPU."""
    from llb.bench.policy_change.audit import PolicyChange, audit_policy_change
    from llb.bench.policy_change.audit_report import policy_change_summary
    from llb.bench.policy_change.geometry import load_audited_designs
    from llb.bench.policy_change.pin_gate import PINS_PATH, load_policy_pins
    from llb.bench.published_value.operations.scope import policy_affected_published_values
    from llb.core.paths import PROJECT_ROOT

    pins = load_policy_pins(PROJECT_ROOT / PINS_PATH)
    pinned = {field: pin.value for field, pin in pins.pins.items()}
    change = PolicyChange.of(POLICY_FIELD, SUMMARY_TRIM_HEAD_TAIL, SUMMARY_TRIM_PER_ENTRY_HEAD)
    summary = policy_change_summary(
        audit_policy_change(load_audited_designs(), change, pinned=pinned), change
    )
    affected = policy_affected_published_values(PROJECT_ROOT, change.fields)
    return {
        "change": change.label,
        "pinned_policy": pinned,
        "n_cells": summary["n_cells"],
        "n_prompt_invariant": summary["n_prompt_invariant"],
        "n_invalidated": summary["n_invalidated"],
        "invalidated_cells": [
            f"{row['study_kind']} {row['cell_id']}"
            for row in cast(list[dict[str, object]], summary["invalidated"])
        ],
        "affected_published_values": [row.named() for row in affected],
        "invariant": int(cast(int, summary["n_invalidated"])) == 0,
    }


def analyze_summary_trim_runs(
    design: dict[str, object], runs: list[FamilyRun], *, audit: dict[str, object] | None = None
) -> dict[str, object]:
    """Read every qualified family, then decide what the study licenses."""
    held = cast(dict[str, object], design["held_fixed"])
    required = int(cast(int, design["required_qualified_families"]))
    audit = audit if audit is not None else audit_default_change()
    families = []
    for run in runs:
        reading = family_reading(run.rows)
        eligible, reason = family_eligibility(design, run)
        run.analysis = {
            "model_family": run.model_family,
            "model": run.model,
            "backend": run.backend,
            "eligible": eligible,
            "eligibility_reason": reason,
            "tokens_per_s": run.tokens_per_s,
            **reading,
        }
        families.append(run.analysis)
    qualified = [row for row in families if row["eligible"]][:required]
    verdict, reason = adoption_reading(
        qualified,
        required_families=required,
        audit_invariant=cast(bool, audit["invariant"]),
        required_middle_pairs=_declared_stratum_size(design),
    )
    return {
        "study_id": design["study_id"],
        "study_kind": design["study_kind"],
        "seed": design["seed"],
        "arms": design["arms"],
        "arm_order": design.get("arm_order"),
        "required_qualified_families": required,
        "declared_geometry": [
            {"workload": workload["workload"], **probe_workload(workload, held)}
            for workload in workloads(design)
        ],
        "families": families,
        "qualified_models": [row["model"] for row in qualified],
        "policy_change_audit": audit,
        "adoption_reading": verdict,
        "adoption_reason": reason,
        "changes_shipped_default": False,
    }


def _declared_stratum_size(design: dict[str, object]) -> int:
    """How many cases the recovery claim needs per stratum -- the design's own declared count.

    A run whose middle stratum ends smaller than this did not put those episodes into the regime
    under test, so it is reported as under-powered rather than as evidence either way.
    """
    return next(
        (
            int(cast(int, workload["n_tasks"]))
            for workload in workloads(design)
            if workload["task_builder"] == BUILDER_EVIDENCE_STRATA
        ),
        0,
    )


def family_eligibility(design: dict[str, object], run: FamilyRun) -> tuple[bool, str]:
    """A family qualifies by completing the elision-free control under the SHIPPED trim.

    The control is the workload whose fold fits the summarize-input bound: both arms render the
    identical prompt there, so a family that cannot complete it cannot attribute any later
    difference to the trim strategy rather than to the walk.
    """
    held = cast(dict[str, object], design["held_fixed"])
    name = str(held["qualifying_workload"])
    floor = float(cast(float, held["minimum_control_completion"]))
    row = next(
        (
            item
            for item in run.rows
            if item["workload"] == name and item["arm"] == SUMMARY_TRIM_HEAD_TAIL
        ),
        None,
    )
    if row is None:
        return False, f"the {name!r} control did not run under {SUMMARY_TRIM_HEAD_TAIL!r}"
    completion = float(cast(float, row["completion"]))
    overflows = int(cast(int, row["n_context_overflow"]))
    ok = completion >= floor and overflows == 0
    return ok, (
        f"control={name} completion={completion:.3f} (floor {floor:.3f}); overflows={overflows}"
    )
