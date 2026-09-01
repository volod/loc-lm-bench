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

from llb.conflicts.constants import is_actionable
from llb.conflicts.interval_stats import wilson_interval
from llb.conflicts.null_research.statistics.clusters import two_way_proportion_bound
from llb.core.contracts.common import JsonObject

# Budgets the precision curve is reported at: small enough to read the head of the list, large
# enough to show where a corpus's real relations run out AND large enough for the clustered bound
# to have a chance of clearing zero. Because candidates come out in rank order, the top-N list is a
# PREFIX of the top-M list for N < M, so one run at the largest budget measures every smaller
# budget on exactly the same rows -- the curve IS the budget sweep, with no re-adjudication.
PRECISION_BUDGETS = (6, 12, 25, 50, 100)
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
        """Whether the row is something an operator must decide about, not a coexisting fact.

        The predicate is shared with the report's ordering (`constants.is_actionable`), so the set
        this block MEASURES is the set the report READS first.
        """
        return self.parsed and is_actionable(self.relation)

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
    """Precision over the first `budget` rows with its Wilson and two-way clustered bounds.

    The two cluster censuses answer different questions. `left_clusters` / `right_clusters` size
    the evidence overall; `actionable_left_clusters` / `actionable_right_clusters` are what decide
    whether the clustered bound can clear zero at all -- a resampled draw returns zero whenever it
    misses every chunk that carries an actionable row, so the bound is pinned to 0.0 while the
    conflicts sit on a handful of chunks, however high the point estimate reads.
    """
    head, left, right = flags[:budget], left_keys[:budget], right_keys[:budget]
    successes = sum(head)
    lower, upper = wilson_interval(successes, budget)
    hits = [index for index, flag in enumerate(head) if flag]
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
        "actionable_left_clusters": len({left[index] for index in hits}),
        "actionable_right_clusters": len({right[index] for index in hits}),
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


def budget_resolution(curve: list[JsonObject]) -> JsonObject:
    """The smallest measured budget whose clustered bound clears zero, and what to read from it.

    A point estimate with a 0.0 floor tells an operator nothing about how much of the list is real,
    so the budget that first buys a non-zero floor is the one worth spending adjudication on. When
    no measured budget clears it, that is the answer -- and the census says which of the two causes
    is responsible: too few distinct chunks carrying conflicts, or too few conflicts at all.
    """
    resolving = next(
        (point for point in curve if float(point["two_way_clustered_lcb"]) > 0.0), None
    )
    widest = curve[-1] if curve else None
    payload: JsonObject = {
        "budgets_measured": [int(point["budget"]) for point in curve],
        "resolving_budget": int(resolving["budget"]) if resolving else None,
        "resolving_lcb": float(resolving["two_way_clustered_lcb"]) if resolving else None,
    }
    if resolving is not None:
        payload["reading"] = (
            f"the clustered lower bound first clears zero at budget {resolving['budget']} "
            f"({resolving['two_way_clustered_lcb']}), where the actionable rows rest on "
            f"{resolving['actionable_left_clusters']} left and "
            f"{resolving['actionable_right_clusters']} right chunks; a smaller budget concentrates "
            "them on too few chunks for any floor to survive resampling"
        )
        return payload
    if widest is None:
        payload["reading"] = "no budget was measured, so no floor can be quoted"
        return payload
    payload["reading"] = (
        f"no measured budget up to {widest['budget']} clears zero: at the widest one the "
        f"{widest['actionable_rows']} actionable rows rest on "
        f"{widest['actionable_left_clusters']} left and {widest['actionable_right_clusters']} "
        "right chunks, few enough that a resampled draw can miss all of them, so this corpus "
        "cannot certify a precision floor at an affordable adjudication budget"
    )
    return payload


def rows_precision(rows: list[AdjudicatedRow], *, seed: int) -> JsonObject:
    """Precision at the returned candidate budget plus the curve, from adjudicated rows."""
    flags = [row.actionable for row in rows]
    left = [row.left_key for row in rows]
    right = [row.right_key for row in rows]
    curve = precision_curve_points(flags, left, right, seed=seed)
    return {
        # Cross-encoder ordering is not monotone in cosine, so read the actual extrema rather than
        # assuming the first and last rows bound the range. The cosine-ordered path is unchanged.
        "cosine_range": [
            round(min(row.score for row in rows), 6),
            round(max(row.score for row in rows), 6),
        ]
        if rows
        else None,
        "returned_budget": precision_point(flags, left, right, budget=len(rows), seed=seed),
        "precision_curve": curve,
        "budget_resolution": budget_resolution(curve),
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
