"""Arm rows for the summarize-input-cap study, plus the across-arm completion pairing.

One arm is one summarize-input bound run over the whole fold-step ladder, so an arm row IS a
fold-step depth ladder -- built by the fold-step study's own row builders so the residual and the
boundary are on the exact scale it publishes. What this module adds is the comparison BETWEEN the
arms: the elided span the reference arm paid, and the paired compact-completion delta that says
whether the span carried anything the summary needed.
"""

from typing import cast

from llb.bench.agentic_memory_fold_step_ladder import measured_cap_peak
from llb.bench.agentic_memory_fold_step_rows import depth_fold_row, step_rows
from llb.bench.agentic_memory_fold_step_reading import READING_CONFIRMED
from llb.bench.agentic_memory_summary_cap_reading import ROLE_REFERENCE, ROLE_STEP_ALIGNED
from llb.rag.fusion_evidence.evidence_gate import READING_FLAT, apply_evidence_gate
from llb.rag.fusion_evidence.paired import PairedComparison, paired_comparison
from llb.rag.fusion_evidence.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_index_sets


def arm_rows(
    cells: list[dict[str, object]],
    *,
    arms: list[dict[str, object]],
    prompt_sequence: list[int],
    depth: int,
    compact_share: float,
    cap_cost_fraction: float,
) -> list[dict[str, object]]:
    """One row per arm: its own fold-step ladder plus what its cap elided from the summarizer."""
    return [
        _arm_row(
            arm,
            [cell for cell in cells if cell["arm_id"] == arm["arm_id"]],
            prompt_sequence=prompt_sequence,
            depth=depth,
            compact_share=compact_share,
            cap_cost_fraction=cap_cost_fraction,
        )
        for arm in arms
    ]


def elision_completion_delta(
    cells: list[dict[str, object]], arms: list[dict[str, object]], *, confidence: float
) -> PairedComparison:
    """Step-aligned MINUS reference compact completion, paired over the shared task set.

    The two arms run the SAME tasks at the SAME guards and differ only in how much of the folded
    transcript the summarizer was shown, so the per-task completions pair cell-by-cell: the delta
    is what removing the elision did, not two independent samples differenced.
    """
    aligned = _by_cell(cells, arms, ROLE_STEP_ALIGNED)
    reference = _by_cell(cells, arms, ROLE_REFERENCE)
    shared = [cell_id for cell_id in aligned if cell_id in reference]
    candidate = [value for cell_id in shared for value in aligned[cell_id]]
    baseline = [value for cell_id in shared for value in reference[cell_id]]
    index_sets = bootstrap_index_sets(len(candidate), DEFAULT_RESAMPLES, DEFAULT_SEED)
    return paired_comparison(candidate, baseline, index_sets, confidence)


def completion_delta_reading(completion: PairedComparison, *, confidence: float) -> str:
    """`separated` / `flat` / `insufficient_evidence` for the across-arm completion delta.

    Gated on the DISCORDANT pairs the same way every other paired reading in the repo is: a delta
    resting on fewer differing tasks than the reporting level can reach states nothing.
    """
    stability = cast(dict[str, object], completion.get("stability", {}))
    reading = str(stability.get("reading", READING_FLAT))
    return apply_evidence_gate(
        reading,
        discordant=int(cast(int, completion["wins"])) + int(cast(int, completion["losses"])),
        confidence=confidence,
    )


def reference_elided_chars(cells: list[dict[str, object]], arms: list[dict[str, object]]) -> float:
    """The widest per-episode summarizer elision the reference arm paid across the ladder."""
    reference = next(arm for arm in arms if arm["role"] == ROLE_REFERENCE)
    return max(
        (
            float(cast(float, cell["compact_mean_summary_input_elided_chars"]))
            for cell in cells
            if cell["arm_id"] == reference["arm_id"]
        ),
        default=0.0,
    )


def _arm_row(
    arm: dict[str, object],
    members: list[dict[str, object]],
    *,
    prompt_sequence: list[int],
    depth: int,
    compact_share: float,
    cap_cost_fraction: float,
) -> dict[str, object]:
    ladder = depth_fold_row(
        depth,
        step_rows(
            members,
            prompt_sequence=prompt_sequence,
            compact_share=compact_share,
            cap_cost_fraction=cap_cost_fraction,
        ),
        prompt_sequence=prompt_sequence,
        compact_share=compact_share,
        cap_peak_prompt_chars=measured_cap_peak(
            prompt_sequence, geometry=f"the depth {depth} ladder"
        ),
        reference_guard=None,
    )
    return {
        **ladder,
        "arm_id": arm["arm_id"],
        "role": arm["role"],
        "summary_input_cap": arm["summary_input_cap"],
        "cell_ids": [cast(str, cell["cell_id"]) for cell in members],
        "ladder_reading": ladder["reading"],
        "ladder_confirms_boundary": ladder["reading"] == READING_CONFIRMED,
        "mean_summary_input_chars": _mean(members, "compact_mean_summary_input_chars"),
        "mean_summary_input_elided_chars": _mean(
            members, "compact_mean_summary_input_elided_chars"
        ),
        "max_summary_input_elided_chars": max(
            float(cast(float, cell["compact_mean_summary_input_elided_chars"])) for cell in members
        ),
        "mean_compact_completion": _mean(members, "compact_completion"),
    }


def _by_cell(
    cells: list[dict[str, object]], arms: list[dict[str, object]], role: str
) -> dict[str, list[float]]:
    """Per-task compact completion keyed by cell id, for one arm."""
    arm = next(item for item in arms if item["role"] == role)
    return {
        cast(str, cell["declared_cell_id"]): [
            float(value) for value in cast(list[float], cell["compact_case_success"])
        ]
        for cell in cells
        if cell["arm_id"] == arm["arm_id"]
    }


def _mean(members: list[dict[str, object]], metric: str) -> float:
    values = [float(cast(float, cell[metric])) for cell in members]
    return sum(values) / len(values)
