"""Whether every rung of the fold-count ladder carries evidence, on every qualified family.

The per-family reading in `replication_reading` answers "how far does THIS family carry the rule".
This answers the question one level up, which the per-family tables cannot: is there a fold count
some qualified family never reached with enough paired cases to read? A rule stated on the rungs
either side of an empty one is weaker than it looks, and the hole is a property of the ladder
rather than of any one family's table -- so it is named here, with the guard that produced it, and
the aggregate carries it beside the verdict instead of leaving it to a reader to notice.
"""

from typing import cast

ARM_TYPED_MARKER = "typed_marker"


def ladder_coverage(qualified: list[dict[str, object]]) -> dict[str, object]:
    """Whether every rung of every qualified family's ladder carries floor-clearing evidence.

    The fold-count rule is a LADDER, and a rule stated on the rungs either side of an empty one is
    weaker than it looks. This says so in the aggregate rather than leaving it to a reader to
    compare per-family tables: each hole is named by family, fold count, and the guard that
    produced it, so the fix -- a different guard on that family -- is visible from the reading.
    """
    holes = [
        {
            "model_family": family["model_family"],
            "model": family["model"],
            "measured_folds": row["measured_folds"],
            "n_evidence": row["n_evidence"],
            "evidence_floor": family["evidence_floor"],
            "max_prompt_chars": _fold_group_guards(family, int(cast(int, row["measured_folds"]))),
        }
        for family in qualified
        for row in cast(list[dict[str, object]], family["fold_groups"])
        if not row["meets_evidence_floor"]
    ]
    return {
        "ladder_fully_powered": not holes,
        "underpowered_ladder_rungs": holes,
        "ladder_coverage_reason": (
            "every measured fold group on every qualified family clears its evidence floor"
            if not holes
            else "; ".join(
                f"{hole['model_family']} measured {hole['n_evidence']} paired case(s) at "
                f"{hole['measured_folds']} folds under guard(s) {hole['max_prompt_chars']}, "
                f"below the floor of {hole['evidence_floor']}"
                for hole in holes
            )
        ),
    }


def _fold_group_guards(family: dict[str, object], folds: int) -> list[int]:
    """The guards -- declared or fitted -- whose shipped-policy cases landed on one fold count."""
    return sorted(
        {
            int(cast(int, row["max_prompt_chars"]))
            for row in cast(list[dict[str, object]], family["cells"])
            if row["arm"] == ARM_TYPED_MARKER
            and any(
                int(cast(int, case["measured_folds"])) == folds
                for case in cast(list[dict[str, object]], row["cases"])
            )
        }
    )
