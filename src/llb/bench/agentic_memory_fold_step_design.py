"""Design contract for the fold-step crossover: a grid one fold step apart, checked with no model.

A grid can only tell a step change apart from a smooth slide when its cells are placed against the
deterministic step ladder rather than against round guard numbers. The placement rules themselves
live in `agentic_memory_fold_step_placement.py` (they are shared with the summarize-input-cap
study); this module is the fold-step design's own contract -- the pinned family, the control
recheck, the step rule, and one ladder per tested depth placed against those rules.
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_boundary_probe import cap_prompt_sequence
from llb.bench.agentic_memory_fold_step_ladder import measured_cap_peak
from llb.bench.agentic_memory_fold_step_placement import (
    EXPECTED_SIDES,
    step_guards,
    validate_ladder_shape,
    validate_step_cells,
    validate_step_changes,
    validate_window,
)
from llb.bench.agentic_memory_fold_step_reading import (
    REPORTING_CONFIDENCE,
    STEP_METRIC,
    STEP_RULE,
    STUDY_KIND,
)
from llb.bench.agentic_memory_transfer import load_transfer_design
from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs

__all__ = [
    "EXPECTED_SIDES",
    "declared_cells",
    "fold_step_cap_peaks",
    "fold_step_prompt_sequences",
    "load_fold_step_design",
    "validate_fold_step_design",
]


def load_fold_step_design(path: Path | str) -> dict[str, object]:
    """Reuse the strict JSON loader; fold-step-specific validation runs separately."""
    return load_transfer_design(path)


def validate_fold_step_design(design: dict[str, object]) -> None:
    """Refuse a grid that cannot separate a step change from a guard-axis slide."""
    if design.get("schema_version") != 1 or design.get("study_kind") != STUDY_KIND:
        raise ValueError("fold-step crossover design schema or study kind is invalid")
    confidence = float(cast(float, design.get("reporting_confidence", 0.0)))
    if confidence != REPORTING_CONFIDENCE:
        raise ValueError(f"fold-step crossover reporting confidence must be {REPORTING_CONFIDENCE}")

    held = cast(dict[str, object], design.get("held_fixed", {}))
    share = float(cast(float, held.get("compact_share", 0.0)))
    if not held.get("model") or not held.get("model_family") or held.get("backend") != "ollama":
        raise ValueError("the fold-step crossover study must pin one local Ollama model family")
    if not 0.0 < share <= 1.0:
        raise ValueError("fold-step crossover compact_share is invalid")
    if int(cast(int, held.get("n_tasks", 0))) < minimum_discordant_pairs(confidence):
        raise ValueError("fold-step crossover cells cannot reach the 97.5% exact-sign-test floor")
    if any(
        not 0.0 <= float(cast(float, held.get(floor, -1.0))) <= 1.0
        for floor in ("minimum_compaction_rate", "minimum_cell_completion")
    ):
        raise ValueError("fold-step crossover activation or completion floor is invalid")

    control = cast(dict[str, object], design.get("control_recheck", {}))
    threshold = float(cast(float, control.get("minimum_completion_rate", 0.0)))
    if (
        int(cast(int, control.get("n_tasks", 0))) < 1
        or int(cast(int, control.get("depth", 0))) < 3
        or not 0.0 < threshold <= 1.0
    ):
        raise ValueError("fold-step crossover control recheck contract is invalid")

    rule = cast(dict[str, object], design.get("step_rule", {}))
    _validate_step_rule(rule)
    _validate_ladders(cast(list[dict[str, object]], design.get("ladders", [])), held, rule)


def fold_step_prompt_sequences(design: dict[str, object]) -> dict[int, list[int]]:
    """The deterministic per-step cap prompt sizes behind every tested depth (no model, no GPU)."""
    held = cast(dict[str, object], design["held_fixed"])
    return {
        int(cast(int, ladder["depth"])): _prompt_sequence(int(cast(int, ladder["depth"])), held)
        for ladder in cast(list[dict[str, object]], design["ladders"])
    }


def fold_step_cap_peaks(design: dict[str, object]) -> dict[int, int]:
    """The deterministic cap peak prompt behind every tested depth -- the sequence's own maximum."""
    return {
        depth: measured_cap_peak(sequence, geometry=f"depth {depth}")
        for depth, sequence in fold_step_prompt_sequences(design).items()
    }


def declared_cells(design: dict[str, object]) -> list[dict[str, object]]:
    """Every declared cell in ladder order, carrying the depth and fold step it belongs to."""
    return [
        {**cell, "depth": int(cast(int, ladder["depth"])), "fold_step": step["fold_step"]}
        for ladder in cast(list[dict[str, object]], design["ladders"])
        for step in cast(list[dict[str, object]], ladder["steps"])
        for cell in cast(list[dict[str, object]], step["cells"])
    ]


def _validate_step_rule(rule: dict[str, object]) -> None:
    fraction = float(cast(float, rule.get("within_step_cap_cost_fraction", 0.0)))
    span = float(cast(float, rule.get("minimum_within_step_guard_span_fraction", 0.0)))
    gap = int(cast(int, rule.get("maximum_step_change_guard_gap_chars", 0)))
    if (
        rule.get("rule") != STEP_RULE
        or rule.get("metric") != STEP_METRIC
        or rule.get("requires_adjacent_fold_steps") is not True
        or not 0.0 < fraction <= 0.1
        or not 0.0 < span <= 1.0
        or gap < 1
    ):
        raise ValueError(
            "the fold-step study must predeclare the step rule and its placement bounds"
        )


def _validate_ladders(
    ladders: list[dict[str, object]], held: dict[str, object], rule: dict[str, object]
) -> None:
    depths = [int(cast(int, ladder.get("depth", 0))) for ladder in ladders]
    if not ladders or len(set(depths)) != len(depths) or any(depth < 3 for depth in depths):
        raise ValueError("the fold-step study needs at least one ladder per unique depth")
    seen: set[str] = set()
    for ladder in ladders:
        _validate_ladder(ladder, held, rule, seen)


def _validate_ladder(
    ladder: dict[str, object], held: dict[str, object], rule: dict[str, object], seen: set[str]
) -> None:
    depth = int(cast(int, ladder["depth"]))
    label = f"depth {depth}"
    share = float(cast(float, held["compact_share"]))
    sequence = _prompt_sequence(depth, held)
    peak = measured_cap_peak(sequence, geometry=label)
    steps = cast(list[dict[str, object]], ladder.get("steps", []))
    validate_ladder_shape(label, steps, sequence)
    for step in steps:
        validate_step_cells(
            label,
            step,
            sequence=sequence,
            compact_share=share,
            peak_prompt_chars=peak,
            minimum_guard_span_fraction=float(
                cast(float, rule["minimum_within_step_guard_span_fraction"])
            ),
            seen=seen,
        )
    validate_step_changes(label, steps, int(cast(int, rule["maximum_step_change_guard_gap_chars"])))
    validate_window(
        int(cast(int, held["max_model_len"])), max(step_guards(step)[-1] for step in steps)
    )


def _prompt_sequence(depth: int, held: dict[str, object]) -> list[int]:
    return cap_prompt_sequence(
        depth=depth,
        n_tasks=int(cast(int, held["n_tasks"])),
        pad_chars=int(cast(int, held["pad_chars"])),
        max_steps_margin=int(cast(int, held["max_steps_margin"])),
        observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
        observation_head_share=float(cast(float, held["observation_head_share"])),
    )
