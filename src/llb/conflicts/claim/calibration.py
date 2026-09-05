"""Score an adjudicator against the frozen probe before quoting any precision it produced.

A precision figure computed from a model's own verdicts is only as good as the model. The audit
therefore adjudicates the COMMITTED probe (`probe.py`) with the same prompt and the same endpoint
it uses on the operator's corpus, and reports precision only when the model agrees with the frozen
labels at a lower bound that clears the gate for every GATING tier.

Which tiers gate is a deliberate, narrow choice. `TIER_ACCURACY_GATES` holds the floor the audit
refuses to publish below; a tier absent from it is scored, reported, and compared across models
without deciding anything, because a probe tier earns the right to gate only once measurement
shows it separates adjudicators an operator would actually choose between.
"""

import logging

from llb.conflicts.claim.probe import BASE_TIER, CalibrationProbe, ProbePair
from llb.conflicts.claim.prompt import AdjudicationError, adjudication_prompt, parse_adjudication
from llb.conflicts.constants import REL_COMPLEMENTARY
from llb.conflicts.interval_stats import wilson_interval
from llb.core.contracts.common import JsonObject
from llb.prep.frontier.telemetry import LLMComplete

_LOG = logging.getLogger(__name__)

# The adjudicator must agree with the frozen labels at this Wilson 95% lower bound before any
# precision measured with it is printed. Same gate the independent-null research lane applies.
MIN_ADJUDICATOR_ACCURACY_LCB = 0.60

# The gate is a FLOOR, so only the floor tier carries one: the base tier's job is to reject an
# adjudicator that is broken, and no measured hard-tier reading has yet earned the right to reject
# one that is merely weaker than another. See the conflict-detection docs for the evidence.
TIER_ACCURACY_GATES: dict[str, float] = {BASE_TIER: MIN_ADJUDICATOR_ACCURACY_LCB}


def _probe_verdict(pair: ProbePair, complete: LLMComplete) -> JsonObject:
    record: JsonObject = {
        "pair_id": pair.pair_id,
        "tier": pair.tier,
        "expected_relation": pair.relation,
        "expected_actionable": pair.actionable,
    }
    try:
        parsed = parse_adjudication(complete(adjudication_prompt(pair.left_text, pair.right_text)))
    except (AdjudicationError, RuntimeError) as exc:
        return {**record, "parsed": False, "relation": None, "error": str(exc)[:200]}
    actionable = parsed["relation"] != REL_COMPLEMENTARY
    return {
        **record,
        "parsed": True,
        "relation": parsed["relation"],
        "actionable": actionable,
        "agrees": actionable == pair.actionable,
    }


def _agreement(verdicts: list[JsonObject]) -> JsonObject:
    """Agreement over one slice of the probe, on the actionable binary and both sides of it.

    Agreement is measured on the ACTIONABLE binary rather than on the exact relation: a duplicate
    reported as `subsumes` still sends the operator to the same decision, while a conflict reported
    as `complementary` is the error a precision figure would hide.
    """
    parsed = [verdict for verdict in verdicts if verdict["parsed"]]
    agreements = sum(bool(verdict["agrees"]) for verdict in parsed)
    lower, upper = wilson_interval(agreements, len(parsed))
    positives = [verdict for verdict in parsed if verdict["expected_actionable"]]
    negatives = [verdict for verdict in parsed if not verdict["expected_actionable"]]
    return {
        "probe_pairs": len(verdicts),
        "parsed_pairs": len(parsed),
        "unparsed_pairs": len(verdicts) - len(parsed),
        "agreements": agreements,
        "accuracy": round(agreements / len(parsed), 6) if parsed else 0.0,
        "accuracy_wilson_95": [round(lower, 6), round(upper, 6)],
        "labelled_actionable": len(positives),
        "labelled_complementary": len(negatives),
        "recall_on_actionable": round(
            sum(bool(verdict["actionable"]) for verdict in positives) / len(positives), 6
        )
        if positives
        else None,
        "specificity_on_complementary": round(
            sum(not verdict["actionable"] for verdict in negatives) / len(negatives), 6
        )
        if negatives
        else None,
    }


def _tier_block(tier: str, verdicts: list[JsonObject]) -> JsonObject:
    """One tier's agreement, plus the gate it carries and whether it cleared it."""
    stats = _agreement(verdicts)
    gate = TIER_ACCURACY_GATES.get(tier)
    passed = None
    if gate is not None:
        passed = (
            bool(stats["parsed_pairs"])
            and not stats["unparsed_pairs"]
            and float(stats["accuracy_wilson_95"][0]) >= gate
        )
    return {**stats, "min_accuracy_lcb": gate, "gates": gate is not None, "passed": passed}


def _gate_failures(tiers: JsonObject) -> list[str]:
    """Why the probe refuses this adjudicator, one stated reason per gating tier it missed."""
    reasons = []
    for name, block in tiers.items():
        if not block["gates"] or block["passed"]:
            continue
        if block["unparsed_pairs"]:
            reasons.append(
                f"{block['unparsed_pairs']} of {block['probe_pairs']} {name}-tier probe pairs "
                "returned an unparsable verdict"
            )
            continue
        reasons.append(
            f"{name}-tier accuracy {block['accuracy']} over {block['parsed_pairs']} parsed pairs, "
            f"Wilson 95% lower bound {block['accuracy_wilson_95'][0]} against the "
            f"{block['min_accuracy_lcb']} gate"
        )
    return reasons


def _separation(tiers: JsonObject) -> JsonObject | None:
    """How much harder the hard tier proved than the floor, for THIS adjudicator.

    A ranking needs two models, which one run cannot supply; what one run can say is whether the
    tiers read differently at all. A separation of zero on every model is what would license
    retiring the hard tier, and a wide one is what would license promoting it to a gate.
    """
    gating = [name for name, block in tiers.items() if block["gates"]]
    scored = [name for name, block in tiers.items() if not block["gates"]]
    if len(gating) != 1 or not scored:
        return None
    floor, blocks = tiers[gating[0]], {name: tiers[name] for name in scored}
    return {
        "floor_tier": gating[0],
        "floor_accuracy": floor["accuracy"],
        "scored_tiers": {
            name: {
                "accuracy": block["accuracy"],
                "delta_from_floor": round(float(block["accuracy"]) - float(floor["accuracy"]), 6),
                "recall_on_actionable": block["recall_on_actionable"],
                "specificity_on_complementary": block["specificity_on_complementary"],
            }
            for name, block in blocks.items()
        },
    }


def calibrate_adjudicator(probe: CalibrationProbe, complete: LLMComplete) -> JsonObject:
    """Agreement between this adjudicator and the probe's frozen labels, tier by tier."""
    verdicts = [_probe_verdict(pair, complete) for pair in probe.pairs]
    tiers = {
        tier: _tier_block(tier, [v for v in verdicts if v["tier"] == tier]) for tier in probe.tiers
    }
    reasons = _gate_failures(tiers)
    if not any(block["gates"] for block in tiers.values()):
        reasons.append(
            "the probe carried no gating tier, so nothing established that this adjudicator "
            f"clears the floor (gating tiers: {', '.join(sorted(TIER_ACCURACY_GATES))})"
        )
    return {
        "probe_id": probe.probe_id,
        "probe_corpora": probe.corpora,
        **_agreement(verdicts),
        "min_accuracy_lcb": MIN_ADJUDICATOR_ACCURACY_LCB,
        "tiers": tiers,
        "tier_separation": _separation(tiers),
        "calibrated": not reasons,
        "gate_failures": reasons,
        "verdicts": verdicts,
    }


def log_calibration(calibration: JsonObject) -> None:
    """One line per tier, so a long run says what the probe found while it is still running."""
    for tier, block in calibration["tiers"].items():
        gate = block["min_accuracy_lcb"]
        _LOG.info(
            "[conflicts] adjudicator calibration: %s of %s %s-tier probe pairs agree "
            "(accuracy %s, Wilson 95%% lower bound %s, %s)",
            block["agreements"],
            block["parsed_pairs"],
            tier,
            block["accuracy"],
            block["accuracy_wilson_95"][0],
            f"gate {gate}" if gate is not None else "reported, does not gate",
        )
