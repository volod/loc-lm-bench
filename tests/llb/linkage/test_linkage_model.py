"""The result contract: what an operator reads off a linkage run without opening the bundle."""

from llb.linkage.model import (
    AccuracyPoint,
    BlockingCount,
    LinkageCluster,
    LinkagePair,
    LinkageResult,
    MatchParameter,
    pairs_above,
    pairs_co_clustered,
    score_labels,
    sorted_clusters,
    sorted_pairs,
)
from llb.linkage.records import ReviewerLabel
from llb.linkage.spec import BlockingRule, ComparisonSpec, LinkageSpec

SPEC = LinkageSpec(
    comparisons=(ComparisonSpec("name", "exact"), ComparisonSpec("city", "exact")),
    blocking_rules=(BlockingRule(("city",)),),
    match_threshold=0.9,
)


def _result(**overrides) -> LinkageResult:
    base = dict(
        spec=SPEC,
        blocking_counts=(BlockingCount("city", 10, 6),),
        match_parameters=(MatchParameter("name", "Exact match on name", 0.8, 0.01),),
        pairs=(
            LinkagePair("a", "b", 0.99, 6.6, {"name": 2}),
            LinkagePair("a", "c", 0.40, 0.1, {"name": 0}),
        ),
        clusters=(
            LinkageCluster("a", ("a", "b")),
            LinkageCluster("c", ("c",)),
        ),
        trained_model={"comparisons": []},
        n_records=3,
    )
    base.update(overrides)
    return LinkageResult(**base)


def test_only_pairs_at_or_above_the_threshold_count_as_matched():
    result = _result()
    assert [(p.left_id, p.right_id) for p in result.matched_pairs] == [("a", "b")]


def test_the_summary_reports_the_counts_and_the_settings_behind_them():
    assert _result().summary() == {
        "n_records": 3,
        "n_scored_pairs": 2,
        "n_matched_pairs": 1,
        "n_clusters": 2,
        "n_multi_record_clusters": 1,
        "largest_cluster": 2,
        "match_threshold": 0.9,
        "seed": SPEC.seed,
        "trained_from_labels": False,
        "n_accuracy_points": 0,
        "n_untrained_levels": 0,
    }


def test_a_level_with_no_estimate_is_reported_not_hidden():
    result = _result(
        match_parameters=(
            MatchParameter("name", "Exact match on name", 0.8, 0.01),
            MatchParameter("name", "name is NULL", None, None),
            MatchParameter("city", "Exact match on city", None, 0.3),
        )
    )
    assert [p.comparison for p in result.untrained_levels] == ["city"]
    assert result.summary()["n_untrained_levels"] == 1


def test_a_record_reports_the_cluster_it_was_proposed_into():
    result = _result()
    assert result.cluster_of("b") == "a"
    assert result.cluster_of("missing") is None


def test_pairs_sort_by_probability_and_clusters_by_size():
    pairs = sorted_pairs([LinkagePair("x", "y", 0.5, 1.0), LinkagePair("a", "b", 0.9, 2.0)])
    assert [p.left_id for p in pairs] == ["a", "x"]
    clusters = sorted_clusters([LinkageCluster("z", ("z",)), LinkageCluster("a", ("a", "b", "c"))])
    assert [c.cluster_id for c in clusters] == ["a", "z"]


def test_without_labels_there_is_no_operating_point():
    result = _result()
    assert result.best_accuracy() is None
    assert result.pair_operating_point is None
    assert result.cluster_operating_point is None


def test_the_operating_point_is_read_off_the_labelled_curve():
    curve = (
        AccuracyPoint(0.50, 10, 4, 6, 0, 0.714, 1.0, 0.833),
        AccuracyPoint(0.85, 10, 0, 10, 0, 1.0, 1.0, 1.0),
        AccuracyPoint(0.99, 4, 0, 10, 6, 1.0, 0.4, 0.571),
    )
    result = _result(accuracy=curve)
    assert result.best_accuracy() is not None
    assert result.best_accuracy().threshold == 0.85
    assert result.summary()["n_accuracy_points"] == 3


def test_the_cut_is_scored_against_the_labels_exactly():
    labels = [
        ReviewerLabel("a", "b", True),  # scored above the cut
        ReviewerLabel("b", "a", True),  # same pair, reviewer order reversed
        ReviewerLabel("a", "c", False),  # scored below the cut
        ReviewerLabel("a", "z", True),  # never compared: a merge never proposed
    ]
    pairs = _result().pairs
    point = score_labels(pairs_above(pairs, 0.9), labels, 0.9)
    assert (point.true_positives, point.false_positives) == (2, 0)
    assert (point.true_negatives, point.false_negatives) == (1, 1)
    assert point.precision == 1.0 and point.recall == 2 / 3


def test_clustering_merges_pairs_the_pairwise_cut_leaves_apart():
    result = _result(clusters=(LinkageCluster("a", ("a", "b", "c")),))
    labels = [ReviewerLabel("a", "c", True)]
    pairwise = score_labels(pairs_above(result.pairs, 0.9), labels, 0.9)
    clustered = score_labels(pairs_co_clustered(result.clusters), labels, 0.9)
    assert pairwise.recall == 0.0  # a~c scores 0.40, below the cut
    assert clustered.recall == 1.0  # but b links them, so the run did merge a and c


def test_an_empty_label_set_scores_to_zero_rather_than_dividing_by_zero():
    point = score_labels(set(), [], 0.9)
    assert (point.precision, point.recall, point.f1) == (0.0, 0.0, 0.0)
