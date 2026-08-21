"""The failure message the context-policy pin gate fails a build with.

A gate is only as useful as what it says when it fires. This one has to answer, in the terminal
output of a build somebody did not expect to break: which constant moved, which cell measurements
and derived statements that retires, which exact published figures those cells stand under, where
those numbers are quoted, and what the ways out are. The cell scope itself is rendered by the
audit's own reporter, so the operator reads the same lines whether they asked the question or CI
asked it.
"""

from typing import cast

from llb.bench.policy_change.audit_report import (
    audited_axis,
    format_invalidated_cells,
    partial_note,
)
from llb.bench.policy_change.pin_gate import PINS_PATH, PinCheck, PinClaim, PinDrift, PolicyPins


def format_pin_gate_report(check: PinCheck) -> str:
    """Name moved constants, retired cells and derived values, and the required repair."""
    n_pins = len(check.pins.pins)
    if check.ok:
        return (
            f"verdict: all {n_pins} shipped context-policy constants match the value the "
            f"published agentic evidence was measured under ({PINS_PATH})"
        )
    lines = [
        f"verdict: {len(check.moves)} of {n_pins} shipped context-policy constants no "
        "longer match the value the published agentic evidence was measured under",
        f"pins: {PINS_PATH}",
        f"published numbers: {', '.join(check.pins.published_in)}",
    ]
    if check.drift is not None:
        lines.extend(["", *_drift_lines(check.drift, check.pins)])
    for claim in check.stale_claims:
        lines.extend(["", *_claim_lines(claim)])
    return "\n".join(lines)


def _drift_lines(drift: PinDrift, pins: PolicyPins) -> list[str]:
    summary = drift.summary
    skipped = cast(int, summary["n_not_applicable"])
    applicable = cast(int, summary["n_cells"]) - skipped
    tail = (
        [
            f"  {skipped} further cell(s) pin {audited_axis(summary)} as their own study axis, so "
            "the change does not describe them."
        ]
        if skipped
        else []
    )
    tail.extend(partial_note(summary, indent="  "))
    label = "the shipped values send" if drift.is_compound else "the shipped value sends"
    notes = [
        f"  pinned because{f' ({move.field})' if drift.is_compound else ''}: "
        f"{pins.pins[move.field].note}"
        for move in drift.moves
    ]
    if not drift.n_invalidated:
        cell_reading = (
            "bit-identical prompts; the registered-arithmetic scope follows."
            if drift.affected_published_values
            else "bit-identical prompts, so restating the pin is free."
        )
        lines = [
            *_head_lines(drift),
            f"  no published cell is invalidated ({applicable} applicable): {label} {cell_reading}",
            *tail,
        ]
    else:
        studies = cast(list[str], summary["studies_invalidated"])
        lines = [
            *_head_lines(drift),
            f"  {drift.n_invalidated} of {applicable} applicable published cells are invalidated "
            f"across {len(studies)} study/studies ({', '.join(studies)}):",
            *format_invalidated_cells(summary, indent="  "),
            "  re-measure those cells and restate their published numbers, then move the pin -- "
            "or revert the constant.",
            *tail,
        ]
    if drift.retired_figures:
        lines.extend(
            [
                "  those cells retire these published figures:",
                *(f"  - {figure.named()}" for figure in drift.retired_figures),
                "  restate those figures in the docs when moving the pin -- or revert the "
                "constant.",
            ]
        )
    if drift.affected_published_values:
        lines.extend(
            [
                "  registered arithmetic also moves these published values:",
                *(f"  - {value.named()}" for value in drift.affected_published_values),
                "  re-derive and restate those values when moving the pin -- or revert the "
                "constant.",
            ]
        )
    return [*lines, *notes]


def _head_lines(drift: PinDrift) -> list[str]:
    """Name what moved. Several constants moving in one commit is ONE change, and says so."""
    moved = [
        f"- {move.field}: pinned {move.pinned!r} -> shipped {move.shipped!r}"
        for move in drift.moves
    ]
    if not drift.is_compound:
        return moved
    return [
        f"- {len(drift.moves)} constants moved together and are audited as ONE change:",
        *(f"  {line}" for line in moved),
        "  the baseline arm replays the full pinned policy and the candidate arm the full shipped "
        "policy, so the scope below is what THIS commit retires -- not one scope per constant "
        "against a configuration that never shipped.",
    ]


def _claim_lines(claim: PinClaim) -> list[str]:
    stated = ", ".join(f"{kind}={value!r}" for kind, value in claim.stated) or "no design"
    return [
        f"- {claim.field}: the pin declares designs={claim.declared!r}, but the committed studies "
        f"support {claim.supported!r} ({stated})",
        "  restate the pin's `designs` field (and its note) to what the designs actually say.",
    ]
