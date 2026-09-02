"""Render an adjudicator's frozen-probe calibration, for both places it is read.

`calibration_lines` is the compact form the audit report prints beside the precision the probe
licenses. `calibration_report` is the standalone page for the other use: comparing model families
against the probe with no corpus, no store, and no adjudication budget in the way -- so what it has
to show is the difficulty LADDER (which tier the model held and which it lost) and the pairs it
actually missed, not one accuracy number.
"""

from llb.core.contracts.common import JsonObject

_TIER_HEADER = (
    "| tier | pairs | parsed | agree | accuracy | Wilson 95% | recall | specificity "
    "| gate | verdict |"
)
_TIER_RULE = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def _measured(value: object) -> str:
    """A measured number, or `--` where the tier held no pair of that half to measure it on."""
    return "--" if value is None else str(value)


def _tier_row(tier: str, block: JsonObject) -> str:
    gate = block["min_accuracy_lcb"]
    verdict = ("cleared" if block["passed"] else "MISSED") if gate is not None else "reports only"
    return (
        f"| {tier} | {block['probe_pairs']} | {block['parsed_pairs']} | {block['agreements']} "
        f"| {block['accuracy']} | {block['accuracy_wilson_95']} "
        f"| {_measured(block['recall_on_actionable'])} "
        f"| {_measured(block['specificity_on_complementary'])} "
        f"| {gate if gate is not None else '--'} | {verdict} |"
    )


def _misses(calibration: JsonObject) -> list[str]:
    """Every pair the adjudicator answered against its frozen label, or an explicit none."""
    missed = [
        verdict
        for verdict in calibration["verdicts"]
        if not verdict["parsed"] or not verdict["agrees"]
    ]
    if not missed:
        return ["The adjudicator agreed with every frozen label on every tier it was shown.", ""]
    lines = [
        "| tier | pair | frozen relation | frozen | answered | answered as |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for verdict in missed:
        expected = "actionable" if verdict["expected_actionable"] else "complementary"
        answered = (
            ("actionable" if verdict["actionable"] else "complementary")
            if verdict["parsed"]
            else "unparsable"
        )
        lines.append(
            f"| {verdict['tier']} | `{verdict['pair_id']}` | {verdict['expected_relation']} "
            f"| {expected} | {answered} | {verdict['relation'] or '--'} |"
        )
    return lines + [""]


def _separation_lines(calibration: JsonObject) -> list[str]:
    separation = calibration.get("tier_separation")
    if not separation:
        return []
    lines = [
        f"Against the `{separation['floor_tier']}` floor at accuracy "
        f"{separation['floor_accuracy']}:",
        "",
    ]
    for name, block in separation["scored_tiers"].items():
        lines.append(
            f"- `{name}`: accuracy {block['accuracy']} ({block['delta_from_floor']:+}), recall "
            f"{_measured(block['recall_on_actionable'])}, specificity "
            f"{_measured(block['specificity_on_complementary'])}"
        )
    return lines + [""]


def _tier_line(tier: str, block: JsonObject) -> str:
    """One probe tier, as one line: what it agreed on and whether it decided anything."""
    gate = block["min_accuracy_lcb"]
    verdict = (
        f"gate {gate}, {'cleared' if block['passed'] else 'missed'}"
        if gate is not None
        else "reported, does not gate"
    )
    return (
        f"  - {tier} tier: {block['agreements']}/{block['parsed_pairs']} frozen probe pairs agree "
        f"(accuracy {block['accuracy']}, Wilson 95% {block['accuracy_wilson_95']}; recall "
        f"{_measured(block['recall_on_actionable'])}, specificity "
        f"{_measured(block['specificity_on_complementary'])}) -- {verdict}"
    )


def calibration_lines(calibration: JsonObject | None) -> list[str]:
    """The probe's reading, tier by tier -- the floor that gates and the tiers that only report."""
    if not calibration:
        return ["- adjudicator calibration: not run"]
    lines = [
        f"- adjudicator calibration, probe `{calibration['probe_id']}`: "
        f"{calibration['agreements']}/{calibration['parsed_pairs']} pairs agree overall"
    ]
    lines += [_tier_line(tier, block) for tier, block in calibration["tiers"].items()]
    separation = calibration.get("tier_separation")
    if separation:
        deltas = ", ".join(
            f"{name} {block['delta_from_floor']:+}"
            for name, block in separation["scored_tiers"].items()
        )
        lines.append(
            f"  - accuracy against the {separation['floor_tier']} floor "
            f"({separation['floor_accuracy']}): {deltas}"
        )
    return lines


def calibration_report(payload: JsonObject) -> str:
    """One model against one probe: the ladder, the separation, and the pairs it missed."""
    calibration = payload["calibration"]
    corpora = ", ".join(
        f"`{tier}` -> `{path}`" for tier, path in calibration["probe_corpora"].items()
    )
    lines = [
        "# Adjudicator calibration",
        "",
        f"- model: `{payload['model']}` on `{payload['backend']}` "
        f"(temperature {payload['temperature']}, seed {payload['seed']})",
        f"- probe: `{calibration['probe_id']}` from `{payload['probe']}`",
        f"- probe corpora: {corpora}",
        f"- pairs adjudicated: {calibration['probe_pairs']} in {payload['seconds']} s",
        f"- overall agreement: {calibration['agreements']}/{calibration['parsed_pairs']} "
        f"(accuracy {calibration['accuracy']}, Wilson 95% {calibration['accuracy_wilson_95']})",
        f"- **calibrated: {'yes' if calibration['calibrated'] else 'no'}**"
        + ("" if calibration["calibrated"] else " -- " + "; ".join(calibration["gate_failures"])),
        "",
        "## Tiers",
        "",
        "Agreement is scored on the actionable/complementary binary. `recall` is agreement on the",
        "actionable half of the tier, `specificity` on the complementary half. Only a tier with a",
        "gate decides anything; the others are measured so two adjudicators can be compared.",
        "",
        _TIER_HEADER,
        _TIER_RULE,
    ]
    lines += [_tier_row(tier, block) for tier, block in calibration["tiers"].items()]
    lines += ["", *_separation_lines(calibration), "## Disagreements", "", *_misses(calibration)]
    return "\n".join(lines)
