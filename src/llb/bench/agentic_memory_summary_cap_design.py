"""Design contract for the summarize-input-cap study: two arms over ONE fold-step ladder.

The placement rules are the fold-step study's, unchanged -- the ladder must still fold where it
claims, on adjacent reachable steps, with guards spanning each step's interval and straddling the
step change closely -- because the residual this study drives to zero is only meaningful on the
scale that ladder measures it.

What is new is checkable with no model too: the reference arm must actually ELIDE the folded
transcript on at least one declared guard, and the step-aligned arm must elide nowhere and offer
the summarizer a bit-identical input inside each step. Both come from the deterministic compact
walk in `agentic_memory_boundary_probe.compact_fold_input_probe`, so a design that has no elision
to price -- or an arm that does not do what its name says -- is refused before a GPU is warmed.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic.context import SUMMARY_INPUT_CAP_TRIGGER, SUMMARY_INPUT_CAPS
from llb.bench.agentic_memory_boundary_probe import cap_prompt_sequence, compact_fold_input_probe
from llb.bench.agentic_memory_fold_step_ladder import measured_cap_peak
from llb.bench.agentic_memory_fold_step_placement import (
    step_guards,
    validate_ladder_shape,
    validate_step_cells,
    validate_step_changes,
    validate_window,
)
from llb.bench.agentic_memory_summary_cap_reading import (
    ARM_ROLES,
    CAP_METRIC,
    CAP_RULE,
    REPORTING_CONFIDENCE,
    ROLE_REFERENCE,
    ROLE_STEP_ALIGNED,
    STUDY_KIND,
)
from llb.bench.agentic_memory_transfer import load_transfer_design
from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs


def load_summary_cap_design(path: Path | str) -> dict[str, object]:
    """Reuse the strict JSON loader; summarize-input-cap validation runs separately."""
    return load_transfer_design(path)


def validate_summary_cap_design(design: dict[str, object]) -> None:
    """Refuse a design that cannot separate a step-aligned cap from the trigger it replaces."""
    if design.get("schema_version") != 1 or design.get("study_kind") != STUDY_KIND:
        raise ValueError("summarize-input-cap design schema or study kind is invalid")
    if float(cast(float, design.get("reporting_confidence", 0.0))) != REPORTING_CONFIDENCE:
        raise ValueError(f"summarize-input-cap reporting confidence must be {REPORTING_CONFIDENCE}")

    held = cast(dict[str, object], design.get("held_fixed", {}))
    share = float(cast(float, held.get("compact_share", 0.0)))
    if not held.get("model") or not held.get("model_family") or held.get("backend") != "ollama":
        raise ValueError("the summarize-input-cap study must pin one local Ollama model family")
    if not 0.0 < share <= 1.0:
        raise ValueError("summarize-input-cap compact_share is invalid")
    if int(cast(int, held.get("n_tasks", 0))) < minimum_discordant_pairs(REPORTING_CONFIDENCE):
        raise ValueError("summarize-input-cap cells cannot reach the 97.5% exact-sign-test floor")
    if any(
        not 0.0 <= float(cast(float, held.get(floor, -1.0))) <= 1.0
        for floor in ("minimum_compaction_rate", "minimum_cell_completion")
    ):
        raise ValueError("summarize-input-cap activation or completion floor is invalid")
    if "summary_input_cap" in held:
        raise ValueError("the summarize-input cap is this study's ARM, not a held-fixed setting")

    control = cast(dict[str, object], design.get("control_recheck", {}))
    threshold = float(cast(float, control.get("minimum_completion_rate", 0.0)))
    if (
        int(cast(int, control.get("n_tasks", 0))) < 1
        or int(cast(int, control.get("depth", 0))) < 3
        or not 0.0 < threshold <= 1.0
    ):
        raise ValueError("summarize-input-cap control recheck contract is invalid")

    rule = cast(dict[str, object], design.get("cap_rule", {}))
    _validate_cap_rule(rule)
    _validate_arms(cast(list[dict[str, object]], design.get("arms", [])))
    _validate_ladder(design, held, rule)


def summary_cap_prompt_sequence(design: dict[str, object]) -> list[int]:
    """The deterministic per-step cap prompt sizes behind the tested depth (no model, no GPU)."""
    held = cast(dict[str, object], design["held_fixed"])
    return cap_prompt_sequence(
        depth=summary_cap_depth(design),
        n_tasks=int(cast(int, held["n_tasks"])),
        pad_chars=int(cast(int, held["pad_chars"])),
        max_steps_margin=int(cast(int, held["max_steps_margin"])),
        observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
        observation_head_share=float(cast(float, held["observation_head_share"])),
    )


def summary_cap_depth(design: dict[str, object]) -> int:
    """The one depth this study re-runs -- a ladder, not a surface."""
    return int(cast(int, cast(dict[str, object], design["ladder"])["depth"]))


def declared_cells(design: dict[str, object]) -> list[dict[str, object]]:
    """Every declared cell, once per arm, in the order the run walks them.

    Arm-major order: one arm's whole ladder, then the other's. The two arms differ ONLY in the
    summarize-input bound, so the cell ids are shared and the arm id is what disambiguates them.
    """
    ladder = cast(dict[str, object], design["ladder"])
    depth = int(cast(int, ladder["depth"]))
    return [
        {
            **cell,
            "depth": depth,
            "fold_step": step["fold_step"],
            "arm_id": arm["arm_id"],
            "summary_input_cap": arm["summary_input_cap"],
            "role": arm["role"],
        }
        for arm in cast(list[dict[str, object]], design["arms"])
        for step in cast(list[dict[str, object]], ladder["steps"])
        for cell in cast(list[dict[str, object]], step["cells"])
    ]


def arm_fold_input_probes(design: dict[str, object]) -> list[dict[str, object]]:
    """The model-free summarizer-input prediction for every arm/guard pair in the ladder."""
    held = cast(dict[str, object], design["held_fixed"])
    depth = summary_cap_depth(design)
    share = float(cast(float, held["compact_share"]))
    return [
        {
            "arm_id": arm["arm_id"],
            "summary_input_cap": arm["summary_input_cap"],
            "cell_id": cell["cell_id"],
            "fold_step": cell["fold_step"],
            "max_prompt_chars": cell["max_prompt_chars"],
            **compact_fold_input_probe(
                depth=depth,
                n_tasks=int(cast(int, held["n_tasks"])),
                max_prompt_chars=int(cast(int, cell["max_prompt_chars"])),
                compact_share=share,
                summary_input_cap=cast(str, arm["summary_input_cap"]),
                pad_chars=int(cast(int, held["pad_chars"])),
                max_steps_margin=int(cast(int, held["max_steps_margin"])),
                observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
                observation_head_share=float(cast(float, held["observation_head_share"])),
            ),
        }
        for arm in cast(list[dict[str, object]], design["arms"])
        for cell in declared_cells(design)
        if cell["arm_id"] == arm["arm_id"]
    ]


def _validate_cap_rule(rule: dict[str, object]) -> None:
    tolerance = float(cast(float, rule.get("residual_tolerance_tokens", -1.0)))
    fraction = float(cast(float, rule.get("within_step_cap_cost_fraction", 0.0)))
    span = float(cast(float, rule.get("minimum_within_step_guard_span_fraction", 0.0)))
    gap = int(cast(int, rule.get("maximum_step_change_guard_gap_chars", 0)))
    if (
        rule.get("rule") != CAP_RULE
        or rule.get("metric") != CAP_METRIC
        or rule.get("requires_elided_reference_arm") is not True
        or not 0.0 <= tolerance <= 5.0
        or not 0.0 < fraction <= 0.1
        or not 0.0 < span <= 1.0
        or gap < 1
    ):
        raise ValueError("the summarize-input-cap study must predeclare its rule and its bounds")


def _validate_arms(arms: list[dict[str, object]]) -> None:
    """Exactly two arms differing ONLY in the summarize-input bound, one of them the shipped one."""
    ids = [str(arm.get("arm_id", "")) for arm in arms]
    caps = [str(arm.get("summary_input_cap", "")) for arm in arms]
    roles = [str(arm.get("role", "")) for arm in arms]
    if len(arms) != 2 or not all(ids) or len(set(ids)) != 2:
        raise ValueError("the summarize-input-cap study needs exactly two uniquely named arms")
    if any(cap not in SUMMARY_INPUT_CAPS for cap in caps) or len(set(caps)) != 2:
        raise ValueError(
            f"each arm must pin a distinct summarize-input cap from {SUMMARY_INPUT_CAPS}"
        )
    if sorted(roles) != sorted(ARM_ROLES):
        raise ValueError(f"the two arms must carry the roles {ARM_ROLES}")
    reference = next(arm for arm in arms if arm["role"] == ROLE_REFERENCE)
    if reference["summary_input_cap"] != SUMMARY_INPUT_CAP_TRIGGER:
        raise ValueError(
            "the reference arm must pin the trigger cap the published fold-step residual was "
            "measured under"
        )


def _validate_ladder(
    design: dict[str, object], held: dict[str, object], rule: dict[str, object]
) -> None:
    ladder = cast(dict[str, object], design.get("ladder", {}))
    depth = int(cast(int, ladder.get("depth", 0)))
    label = f"the depth {depth} ladder"
    if depth < 3:
        raise ValueError("the summarize-input-cap ladder needs a depth of at least 3")
    sequence = summary_cap_prompt_sequence(design)
    peak = measured_cap_peak(sequence, geometry=label)
    steps = cast(list[dict[str, object]], ladder.get("steps", []))
    validate_ladder_shape(label, steps, sequence)
    for step in steps:
        validate_step_cells(
            label,
            step,
            sequence=sequence,
            compact_share=float(cast(float, held["compact_share"])),
            peak_prompt_chars=peak,
            minimum_guard_span_fraction=float(
                cast(float, rule["minimum_within_step_guard_span_fraction"])
            ),
        )
    validate_step_changes(label, steps, int(cast(int, rule["maximum_step_change_guard_gap_chars"])))
    validate_window(
        int(cast(int, held["max_model_len"])), max(step_guards(step)[-1] for step in steps)
    )
    _validate_arm_elision(design, rule)


def _validate_arm_elision(design: dict[str, object], rule: dict[str, object]) -> None:
    """Each arm must do what its name says, and the reference arm must have an elision to price."""
    probes = arm_fold_input_probes(design)
    aligned_arm = next(
        arm
        for arm in cast(list[dict[str, object]], design["arms"])
        if arm["role"] == ROLE_STEP_ALIGNED
    )
    aligned = [probe for probe in probes if probe["arm_id"] == aligned_arm["arm_id"]]
    reference = [probe for probe in probes if probe["arm_id"] != aligned_arm["arm_id"]]
    # One fold is what makes the probe's prediction EXACT: a second fold carries the running summary
    # into the summarize input, and the probe answers with a constant summary the run will not.
    if any(int(cast(int, probe["n_compactions"])) != 1 for probe in probes):
        raise ValueError(
            "every declared guard must fold exactly once under perfect play, or the summarizer "
            "input the probe predicts is not the one the run offers"
        )
    if any(int(cast(int, probe["summary_input_elided_chars"])) > 0 for probe in aligned):
        raise ValueError(
            "the step-aligned arm still elides the folded transcript, so it is not step-aligned"
        )
    for step in {int(cast(int, probe["fold_step"])) for probe in aligned}:
        offered = {
            int(cast(int, probe["summary_input_chars"]))
            for probe in aligned
            if probe["fold_step"] == step
        }
        if len(offered) != 1:
            raise ValueError(
                f"the step-aligned arm offers the summarizer {sorted(offered)} chars inside fold "
                f"step {step}, so its input is not fixed by the fold step"
            )
    elided = max(int(cast(int, probe["summary_input_elided_chars"])) for probe in reference)
    if rule["requires_elided_reference_arm"] and elided <= 0:
        raise ValueError(
            "the reference arm elides nothing on this ladder, so the study has no trimmed span to "
            "price against completion"
        )
