"""Design contract for the fold-step crossover: a grid one fold step apart, checked with no model.

A grid can only tell a step change apart from a smooth slide when its cells are placed against the
deterministic step ladder rather than against round guard numbers. Every placement rule here is
therefore checkable before a GPU is warmed: each declared cell must fold at the step it claims, the
tested steps must be ADJACENT on the ladder, the guards inside one step must span that step's whole
interval (otherwise "same step, same cost" is measured over two nearly identical guards), and the
guards on either side of a step change must sit within a few chars of it (otherwise a flip is not
localized to the step change at all).
"""

from pathlib import Path
from typing import cast

from llb.bench.agentic_memory_boundary_gate import SIDE_CAP_CHEAPER, SIDE_COMPACT_CHEAPER
from llb.bench.agentic_memory_boundary_probe import (
    cap_prompt_sequence,
    compaction_trigger_chars,
    first_fold_step,
    fold_step_guard_interval,
    guard_is_cap_fitting,
    reachable_fold_steps,
)
from llb.bench.agentic_memory_fold_step_reading import (
    REPORTING_CONFIDENCE,
    STEP_METRIC,
    STEP_RULE,
    STUDY_KIND,
)
from llb.bench.agentic_memory_transfer import load_transfer_design
from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs

EXPECTED_SIDES = (SIDE_COMPACT_CHEAPER, SIDE_CAP_CHEAPER)


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
    return {depth: max(sequence) for depth, sequence in fold_step_prompt_sequences(design).items()}


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
    share = float(cast(float, held["compact_share"]))
    sequence = _prompt_sequence(depth, held)
    peak = max(sequence)
    steps = cast(list[dict[str, object]], ladder.get("steps", []))
    declared = [int(cast(int, step.get("fold_step", 0))) for step in steps]
    if len(steps) < 2 or declared != sorted(declared) or len(set(declared)) != len(declared):
        raise ValueError(f"depth {depth} needs two or more fold steps in increasing order")
    reachable = reachable_fold_steps(sequence)
    positions = [reachable.index(step) for step in declared if step in reachable]
    if len(positions) != len(declared) or positions != list(
        range(positions[0], positions[0] + len(positions))
    ):
        raise ValueError(
            f"depth {depth} must test fold steps that are ADJACENT on the reachable ladder "
            f"{reachable}, got {declared}"
        )
    sides = [str(step.get("expected_side", "")) for step in steps]
    if set(sides) != set(EXPECTED_SIDES) or sides != sorted(sides, key=EXPECTED_SIDES.index):
        raise ValueError(
            f"depth {depth} must predeclare compact-cheaper fold steps below cap-cheaper ones"
        )
    for step in steps:
        _validate_step(depth, step, sequence, share, peak, rule, seen)
    _validate_step_changes(depth, steps, rule)
    _validate_window(held, max(_guards(step)[-1] for step in steps))


def _validate_step(
    depth: int,
    step: dict[str, object],
    sequence: list[int],
    share: float,
    peak: int,
    rule: dict[str, object],
    seen: set[str],
) -> None:
    fold_step = int(cast(int, step["fold_step"]))
    cells = cast(list[dict[str, object]], step.get("cells", []))
    ids = [str(cell.get("cell_id", "")) for cell in cells]
    guards = _guards(step)
    if len(cells) < 2 or not all(ids) or seen & set(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"depth {depth} fold step {fold_step} needs two or more unique cells")
    seen |= set(ids)
    if any(cell.get("expected_side") != step["expected_side"] for cell in cells):
        raise ValueError(
            f"depth {depth} fold step {fold_step} cells must predeclare the step's own side"
        )
    if len(set(guards)) != len(guards):
        raise ValueError(f"depth {depth} fold step {fold_step} must test distinct guards")
    for guard in guards:
        predicted = first_fold_step(sequence, compaction_trigger_chars(guard, share))
        if predicted != fold_step:
            raise ValueError(
                f"depth {depth} guard {guard} folds at step {predicted}, not the declared "
                f"{fold_step}"
            )
        if not guard_is_cap_fitting(guard, peak, share):
            raise ValueError(
                f"depth {depth} guard {guard} falls outside the usable band around a {peak}-char "
                "cap peak, where cap fits and compact still activates"
            )
    low, high = fold_step_guard_interval(sequence, fold_step, share)
    required = float(cast(float, rule["minimum_within_step_guard_span_fraction"])) * (high - low)
    if guards[-1] - guards[0] < required:
        raise ValueError(
            f"depth {depth} fold step {fold_step} guards span {guards[-1] - guards[0]} chars of "
            f"its [{low}, {high}) interval, under the declared {required:.0f}-char minimum"
        )


def _validate_step_changes(
    depth: int, steps: list[dict[str, object]], rule: dict[str, object]
) -> None:
    """Each step change must be straddled closely, or a flip is not localized to the step change."""
    allowed = int(cast(int, rule["maximum_step_change_guard_gap_chars"]))
    for low, high in zip(steps, steps[1:]):
        gap = _guards(high)[0] - _guards(low)[-1]
        if not 0 < gap <= allowed:
            raise ValueError(
                f"depth {depth} straddles the step "
                f"{low['fold_step']}->{high['fold_step']} change by {gap} chars, over the declared "
                f"{allowed}-char maximum"
            )


def _validate_window(held: dict[str, object], max_guard: int) -> None:
    from llb.optimize.tuning_space import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS

    required = int(max_guard / CHARS_PER_TOKEN) + PROMPT_HEADROOM_TOKENS
    if int(cast(int, held["max_model_len"])) < required:
        raise ValueError(
            f"declared max_model_len cannot carry the widest guard ({max_guard} chars needs about "
            f"{required} tokens)"
        )


def _guards(step: dict[str, object]) -> list[int]:
    return sorted(
        int(cast(int, cell.get("max_prompt_chars", 0)))
        for cell in cast(list[dict[str, object]], step.get("cells", []))
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
