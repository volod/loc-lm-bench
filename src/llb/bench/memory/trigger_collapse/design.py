"""Design contract for the trigger/guard collapse: what a family must be, before it runs.

A family only separates the trigger axis from the guard axis when its cells realize ONE trigger
through different guards (or one guard through different triggers), and only measures a cost when
every pair stays inside the deterministic cap-fitting band. Both are checkable with no model, so a
grid that could not answer the question is refused before a GPU is warmed.
"""

from pathlib import Path
from typing import cast

from llb.bench.memory.boundary.probe import cap_peak_prompt_chars, cap_prompt_sequence
from llb.bench.memory.fold_step.ladder import compaction_trigger_chars, guard_is_cap_fitting
from llb.bench.memory.transfer.run import load_transfer_design
from llb.bench.memory.trigger_collapse.reading import (
    EQUIVALENCE_METRIC,
    EQUIVALENCE_RULE,
    KIND_EQUAL_GUARD,
    KIND_EQUAL_TRIGGER,
    REPORTING_CONFIDENCE,
    STUDY_KIND,
)
from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs


def load_collapse_design(path: Path | str) -> dict[str, object]:
    """Reuse the strict JSON loader; collapse-specific validation runs separately."""
    return load_transfer_design(path)


def validate_collapse_design(design: dict[str, object]) -> None:
    """Refuse a grid that cannot separate the trigger axis from the guard axis."""
    if design.get("schema_version") != 1 or design.get("study_kind") != STUDY_KIND:
        raise ValueError("trigger-collapse design schema or study kind is invalid")
    confidence = float(cast(float, design.get("reporting_confidence", 0.0)))
    if confidence != REPORTING_CONFIDENCE:
        raise ValueError(f"trigger-collapse reporting confidence must be {REPORTING_CONFIDENCE}")

    held = cast(dict[str, object], design.get("held_fixed", {}))
    if not held.get("model") or not held.get("model_family") or held.get("backend") != "ollama":
        raise ValueError("the trigger-collapse study must pin one local Ollama model family")
    if int(cast(int, held.get("n_tasks", 0))) < minimum_discordant_pairs(confidence):
        raise ValueError("trigger-collapse cells cannot reach the 97.5% exact-sign-test floor")
    if any(
        not 0.0 <= float(cast(float, held.get(floor, -1.0))) <= 1.0
        for floor in ("minimum_compaction_rate", "minimum_cell_completion")
    ):
        raise ValueError("trigger-collapse activation or completion floor is invalid")

    control = cast(dict[str, object], design.get("control_recheck", {}))
    threshold = float(cast(float, control.get("minimum_completion_rate", 0.0)))
    if (
        int(cast(int, control.get("n_tasks", 0))) < 1
        or int(cast(int, control.get("depth", 0))) < 3
        or not 0.0 < threshold <= 1.0
    ):
        raise ValueError("trigger-collapse control recheck contract is invalid")

    families = cast(list[dict[str, object]], design.get("families", []))
    _validate_families(families, held)
    _validate_equivalence(cast(dict[str, object], design.get("equivalence", {})), families)


def collapse_cap_peaks(design: dict[str, object]) -> dict[int, int]:
    """The deterministic cap peak prompt behind every tested depth (no model, no GPU)."""
    held = cast(dict[str, object], design["held_fixed"])
    return {
        depth: _cap_peak(depth, held)
        for depth in sorted(
            {
                int(cast(int, family["depth"]))
                for family in cast(list[dict[str, object]], design["families"])
            }
        )
    }


def collapse_prompt_sequences(design: dict[str, object]) -> dict[int, list[int]]:
    """The deterministic per-step cap prompt sizes behind every tested depth."""
    held = cast(dict[str, object], design["held_fixed"])
    return {
        depth: cap_prompt_sequence(
            depth=depth,
            n_tasks=int(cast(int, held["n_tasks"])),
            pad_chars=int(cast(int, held["pad_chars"])),
            max_steps_margin=int(cast(int, held["max_steps_margin"])),
            observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
            observation_head_share=float(cast(float, held["observation_head_share"])),
        )
        for depth in collapse_cap_peaks(design)
    }


def _cap_peak(depth: int, held: dict[str, object]) -> int:
    return cap_peak_prompt_chars(
        depth=depth,
        n_tasks=int(cast(int, held["n_tasks"])),
        pad_chars=int(cast(int, held["pad_chars"])),
        max_steps_margin=int(cast(int, held["max_steps_margin"])),
        observation_cap_chars=int(cast(int, held["observation_cap_chars"])),
        observation_head_share=float(cast(float, held["observation_head_share"])),
    )


def _validate_families(families: list[dict[str, object]], held: dict[str, object]) -> None:
    ids = [str(family.get("family_id", "")) for family in families]
    kinds = [str(family.get("kind", "")) for family in families]
    if len(families) < 2 or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("the collapse study needs at least two uniquely named families")
    if kinds.count(KIND_EQUAL_TRIGGER) < 1 or kinds.count(KIND_EQUAL_GUARD) != 1:
        raise ValueError("the collapse study needs equal-trigger families and one equal-guard one")
    seen: set[str] = set()
    for family in families:
        cells = cast(list[dict[str, object]], family.get("cells", []))
        cell_ids = [str(cell.get("cell_id", "")) for cell in cells]
        if len(cells) < 2 or not all(cell_ids) or seen & set(cell_ids):
            raise ValueError(f"family {family.get('family_id')} needs two or more unique cells")
        seen |= set(cell_ids)
        _validate_family_axes(family, cells)
        _validate_band(family, held, cells)


def _validate_family_axes(family: dict[str, object], cells: list[dict[str, object]]) -> None:
    shares = [float(cast(float, cell.get("compact_share", 0.0))) for cell in cells]
    guards = [int(cast(int, cell.get("max_prompt_chars", 0))) for cell in cells]
    if int(cast(int, family.get("depth", 0))) < 3 or any(
        not 0.0 < share <= 1.0 or guard <= 0 for share, guard in zip(shares, guards, strict=True)
    ):
        raise ValueError(f"family {family.get('family_id')} geometry is invalid")
    if len(set(shares)) != len(shares):
        raise ValueError(f"family {family.get('family_id')} must move compact_share")
    triggers = {
        compaction_trigger_chars(guard, share) for share, guard in zip(shares, guards, strict=True)
    }
    if family.get("kind") == KIND_EQUAL_TRIGGER:
        if len(triggers) != 1 or len(set(guards)) != len(guards):
            raise ValueError(
                f"family {family.get('family_id')} must hold ONE trigger while moving the guard"
            )
    elif len(set(guards)) != 1 or len(triggers) != len(cells):
        raise ValueError(
            f"family {family.get('family_id')} must hold the guard while moving the trigger"
        )


def _validate_band(
    family: dict[str, object], held: dict[str, object], cells: list[dict[str, object]]
) -> None:
    from llb.optimize.tuning_space import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS

    peak = _cap_peak(int(cast(int, family["depth"])), held)
    pairs = [
        (float(cast(float, cell["compact_share"])), int(cast(int, cell["max_prompt_chars"])))
        for cell in cells
    ]
    outside = [pair for pair in pairs if not guard_is_cap_fitting(pair[1], peak, pair[0])]
    if outside:
        raise ValueError(
            f"family {family['family_id']} pairs {outside} fall outside the usable band around a "
            f"{peak}-char cap peak, where cap fits and compact still activates"
        )
    widest = max(guard for _share, guard in pairs)
    required = int(widest / CHARS_PER_TOKEN) + PROMPT_HEADROOM_TOKENS
    if int(cast(int, held["max_model_len"])) < required:
        raise ValueError(
            f"declared max_model_len cannot carry the widest guard in {family['family_id']} "
            f"({widest} chars needs about {required} tokens)"
        )


def _validate_equivalence(rule: dict[str, object], families: list[dict[str, object]]) -> None:
    fraction = float(cast(float, rule.get("cap_cost_fraction", 0.0)))
    contrasts = {
        str(family["family_id"]) for family in families if family.get("kind") == KIND_EQUAL_GUARD
    }
    if (
        rule.get("rule") != EQUIVALENCE_RULE
        or rule.get("metric") != EQUIVALENCE_METRIC
        or rule.get("requires_same_side") is not True
        or not 0.0 < fraction <= 0.1
        or rule.get("contrast_family") not in contrasts
    ):
        raise ValueError("the collapse study must predeclare the equivalence band and its contrast")
