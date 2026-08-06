"""The failure message the context-policy pin gate fails a build with.

A gate is only as useful as what it says when it fires. This one has to answer, in the terminal
output of a build somebody did not expect to break: which constant moved, which published numbers
that retires, where those numbers are quoted, and what the two ways out are (re-measure the cells, or
restate the pin because nothing moved). The re-run scope itself is rendered by the audit's own
reporter, so the operator reads the same lines whether they asked the question or CI asked it.
"""

from typing import cast

from llb.bench.agentic_policy_change_audit_report import format_invalidated_cells
from llb.bench.agentic_policy_pin_gate import PINS_PATH, PinCheck, PinClaim, PinDrift, PolicyPin


def format_pin_gate_report(check: PinCheck) -> str:
    """The failure message: which constant moved, which published cells it retires, what to do."""
    n_pins = len(check.pins.pins)
    if check.ok:
        return (
            f"verdict: all {n_pins} shipped context-policy constants match the value the "
            f"published agentic evidence was measured under ({PINS_PATH})"
        )
    lines = [
        f"verdict: {len(check.drifted)} of {n_pins} shipped context-policy constants no "
        "longer match the value the published agentic evidence was measured under",
        f"pins: {PINS_PATH}",
        f"published numbers: {', '.join(check.pins.published_in)}",
    ]
    for drift in check.drifted:
        lines.extend(["", *_drift_lines(drift, check.pins.pins[drift.field])])
    for claim in check.stale_claims:
        lines.extend(["", *_claim_lines(claim)])
    return "\n".join(lines)


def _drift_lines(drift: PinDrift, pin: PolicyPin) -> list[str]:
    summary = drift.summary
    skipped = cast(int, summary["n_not_applicable"])
    applicable = cast(int, summary["n_cells"]) - skipped
    head = f"- {drift.field}: pinned {drift.pinned!r} -> shipped {drift.shipped!r}"
    tail = (
        [
            f"  {skipped} further cell(s) pin {drift.field} as their own study axis, so the change "
            "does not describe them."
        ]
        if skipped
        else []
    )
    if not drift.n_invalidated:
        return [
            head,
            f"  no published cell is invalidated ({applicable} applicable): the shipped value sends "
            "bit-identical prompts, so restating the pin is free.",
            *tail,
            f"  pinned because: {pin.note}",
        ]
    studies = cast(list[str], summary["studies_invalidated"])
    return [
        head,
        f"  {drift.n_invalidated} of {applicable} applicable published cells are invalidated across "
        f"{len(studies)} study/studies ({', '.join(studies)}):",
        *format_invalidated_cells(summary, indent="  "),
        "  re-measure those cells and restate their published numbers, then move the pin -- or "
        "revert the constant.",
        *tail,
        f"  pinned because: {pin.note}",
    ]


def _claim_lines(claim: PinClaim) -> list[str]:
    stated = ", ".join(f"{kind}={value!r}" for kind, value in claim.stated) or "no design"
    return [
        f"- {claim.field}: the pin declares designs={claim.declared!r}, but the committed studies "
        f"support {claim.supported!r} ({stated})",
        "  restate the pin's `designs` field (and its note) to what the designs actually say.",
    ]
