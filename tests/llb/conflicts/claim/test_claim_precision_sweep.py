"""The candidate-budget sweep: which budget, if any, buys a non-zero precision floor.

Candidates come out in rank order, so the top-N list is a PREFIX of the top-M list and one
adjudicated run measures every budget on exactly the same rows. What the sweep is really reading is
not the point estimate but the cluster census behind it: the clustered bound is pinned to zero for
as long as the actionable rows sit on few enough chunks that a resampled draw can miss all of them.
"""

from llb.conflicts.claim.precision import precision_block
from tests.llb.conflicts.conflict_helpers import adjudicated_rows, calibrated_stub

SEED = 20260812


def test_the_curve_is_a_prefix_sweep_of_one_adjudicated_list():
    """Every budget reads the same rows, so the sweep costs one run rather than one run per budget."""
    flags = [index % 3 != 2 for index in range(100)]
    block = precision_block(adjudicated_rows(flags), calibrated_stub(), seed=SEED)
    curve = block["precision_curve"]
    assert [point["budget"] for point in curve] == [6, 12, 25, 50, 100]
    for point in curve:
        budget = int(point["budget"])
        assert point["actionable_rows"] == sum(flags[:budget]), "each point is a prefix, not a draw"
    assert curve[-1]["budget"] == block["returned_budget"]["budget"]


def test_the_actionable_cluster_census_counts_only_the_chunks_conflicts_sit_on():
    rows = adjudicated_rows(
        [True, True, False, False, False, False],
        left_keys=["L0", "L0", "L1", "L2", "L3", "L4"],
        right_keys=["R0", "R1", "R2", "R3", "R4", "R5"],
    )
    point = precision_block(rows, calibrated_stub(), seed=SEED)["returned_budget"]
    assert (point["left_clusters"], point["right_clusters"]) == (5, 6)
    assert (point["actionable_left_clusters"], point["actionable_right_clusters"]) == (1, 2)


def test_the_sweep_names_the_smallest_budget_whose_bound_clears_zero():
    """Conflicts concentrated on two chunks cannot hold a floor; spread over many, they can."""
    concentrated = [True] * 2 + [False] * 10
    spread = concentrated + [index % 2 == 0 for index in range(13)]
    rows = adjudicated_rows(
        spread,
        left_keys=["L0", "L0"] + [f"L{index}" for index in range(1, 24)],
        right_keys=["R0", "R0"] + [f"R{index}" for index in range(1, 24)],
    )
    resolution = precision_block(rows, calibrated_stub(), seed=SEED)["budget_resolution"]
    assert resolution["resolving_budget"] == 25
    assert resolution["resolving_lcb"] > 0.0
    assert "first clears zero at budget 25" in resolution["reading"]


def test_the_sweep_records_that_no_budget_clears_zero_when_none_does():
    rows = adjudicated_rows(
        [True] * 2 + [False] * 48,
        left_keys=["L0", "L0"] + [f"L{index}" for index in range(1, 49)],
        right_keys=["R0", "R0"] + [f"R{index}" for index in range(1, 49)],
    )
    resolution = precision_block(rows, calibrated_stub(), seed=SEED)["budget_resolution"]
    assert resolution["resolving_budget"] is None and resolution["resolving_lcb"] is None
    assert resolution["budgets_measured"] == [6, 12, 25, 50]
    assert "no measured budget up to 50 clears zero" in resolution["reading"]
    assert "rest on 1 left and 1 right chunks" in resolution["reading"]
