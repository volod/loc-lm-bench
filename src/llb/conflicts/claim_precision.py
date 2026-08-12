"""Measured precision of the claim-tier candidate list, shared by the audit and the null research.

A semantic false-positive rate needs an independent null the corpus cannot supply
(`docs/impl/current/data-prep/conflict-detection.md`). The precision of the list an operator
actually receives does not: every returned row is adjudicated anyway, so the estimate's sample size
is the adjudication budget rather than the pair space.

Two properties make the number honest, and both live here so the audit and the research harness
report the SAME quantity:

  - rows that share a left or a right chunk are not independent evidence, so the bound is a
    two-way clustered resample over the two unit sets, never a pair-row binomial;
  - an unparsed verdict counts against precision rather than being dropped, because a row the
    adjudicator could not classify is a row the operator still has to look at.

The bound itself is `two_way_proportion_bound`, the estimator the independent-null research
established; importing it rather than reimplementing it is what makes the audit's printed bound
equal the research harness's bound on the same rows.
"""

from dataclasses import dataclass

from llb.conflicts.constants import REL_COMPLEMENTARY
from llb.conflicts.interval_stats import wilson_interval
from llb.conflicts.null_research_clusters import two_way_proportion_bound
from llb.core.contracts.common import JsonObject

# Budgets the precision curve is reported at: small enough to read the head of the list, large
# enough to show where a corpus's real relations run out.
PRECISION_BUDGETS = (6, 12, 25, 50)
# A row whose verdict could not be parsed is counted as not actionable; above this share of the
# list the precision estimate is measuring the parser, not the corpus.
MAX_UNPARSED_FRACTION = 0.05
# ...but an unparsed row biases precision DOWNWARD, so the printed figure stays a lower bound and
# one malformed completion must not erase a whole measurement. At the suggested 12-row candidate
# budget the fraction alone would suppress on a single row, which is why the allowance has a floor.
MIN_UNPARSED_ALLOWANCE = 1


def unparsed_allowance(rows: int) -> int:
    """How many unparsable verdicts a list of `rows` may carry and still be worth reporting."""
    return max(MIN_UNPARSED_ALLOWANCE, int(MAX_UNPARSED_FRACTION * rows))


@dataclass(frozen=True)
class AdjudicatedRow:
    """One adjudicated candidate row, addressed by the two chunks whose sharing makes rows dependent."""

    rank: int
    left_key: str
    right_key: str
    score: float
    relation: str | None
    parsed: bool

    @property
    def actionable(self) -> bool:
        """Whether the row is something an operator must decide about, not a coexisting fact."""
        return self.parsed and self.relation is not None and self.relation != REL_COMPLEMENTARY

    def payload(self) -> JsonObject:
        return {
            "rank": self.rank,
            "left_chunk": self.left_key,
            "right_chunk": self.right_key,
            "cosine": round(float(self.score), 6),
            "relation": self.relation,
            "actionable": self.actionable,
            "parsed": self.parsed,
        }


def precision_point(
    flags: list[bool], left_keys: list[str], right_keys: list[str], *, budget: int, seed: int
) -> JsonObject:
    """Precision over the first `budget` rows with its Wilson and two-way clustered bounds."""
    head, left, right = flags[:budget], left_keys[:budget], right_keys[:budget]
    successes = sum(head)
    lower, upper = wilson_interval(successes, budget)
    return {
        "budget": budget,
        "actionable_rows": successes,
        "precision": round(successes / budget, 6) if budget else 0.0,
        "wilson_95": [round(lower, 6), round(upper, 6)],
        "two_way_clustered_lcb": round(
            two_way_proportion_bound(head, left, right, seed=seed + budget), 6
        ),
        "left_clusters": len(set(left)),
        "right_clusters": len(set(right)),
    }


def precision_curve_points(
    flags: list[bool],
    left_keys: list[str],
    right_keys: list[str],
    *,
    seed: int,
    budgets: tuple[int, ...] = PRECISION_BUDGETS,
) -> list[JsonObject]:
    """The precision curve at every budget the adjudicated list is long enough to report."""
    return [
        precision_point(flags, left_keys, right_keys, budget=budget, seed=seed)
        for budget in budgets
        if budget <= len(flags)
    ]


def rows_precision(rows: list[AdjudicatedRow], *, seed: int) -> JsonObject:
    """Precision at the returned candidate budget plus the curve, from adjudicated rows."""
    flags = [row.actionable for row in rows]
    left = [row.left_key for row in rows]
    right = [row.right_key for row in rows]
    return {
        "cosine_range": [round(rows[-1].score, 6), round(rows[0].score, 6)] if rows else None,
        "returned_budget": precision_point(flags, left, right, budget=len(rows), seed=seed),
        "precision_curve": precision_curve_points(flags, left, right, seed=seed),
    }


def _suppression_reason(
    rows: list[AdjudicatedRow], calibration: JsonObject | None, unparsed: int
) -> str | None:
    """Why a precision figure must not be printed, or None when it is earned."""
    if not rows:
        return "no candidate rows were adjudicated, so there is no returned list to measure"
    if calibration is None:
        return (
            "adjudicator calibration was not run, so a precision figure would rest on an "
            "unmeasured adjudicator"
        )
    if not calibration.get("calibrated"):
        lower = float(calibration.get("accuracy_wilson_95", [0.0, 1.0])[0])
        return (
            f"the adjudicator missed its calibration bound on the frozen probe: accuracy "
            f"{calibration.get('accuracy')} over {calibration.get('parsed_pairs')} parsed pairs, "
            f"Wilson 95% lower bound {round(lower, 4)} against the "
            f"{calibration.get('min_accuracy_lcb')} gate"
        )
    allowance = unparsed_allowance(len(rows))
    if unparsed > allowance:
        return (
            f"{unparsed} of {len(rows)} adjudicated rows returned an unparsable verdict, above "
            f"the {allowance} this list may carry, so precision would measure the adjudicator's "
            "output format rather than the corpus"
        )
    return None


def precision_block(
    rows: list[AdjudicatedRow], calibration: JsonObject | None, *, seed: int
) -> JsonObject:
    """The audit's claim-tier precision block, or the stated reason it is not reported."""
    unparsed = sum(not row.parsed for row in rows)
    block: JsonObject = {
        "method": "claim_tier_precision",
        "adjudicated_rows": len(rows),
        "unparsed_rows": unparsed,
        "unparsed_fraction": round(unparsed / len(rows), 6) if rows else 1.0,
        "seed": seed,
        "adjudicator_calibration": calibration,
        "reported": False,
    }
    reason = _suppression_reason(rows, calibration, unparsed)
    if reason is not None:
        block["reason"] = reason
        return block
    block["reported"] = True
    block.update(rows_precision(rows, seed=seed))
    block["rows"] = [row.payload() for row in rows]
    return block
