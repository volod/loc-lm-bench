"""The specification contract: what a linkage run refuses before it reads a table.

No Splink here on purpose -- the spec, its validation, and its JSON round trip must work in the
base install, because every consumer of the seam reads a saved `settings.json`.
"""

import pytest
from llb.linkage.spec import BlockingRule, ComparisonSpec, LinkageSpec, load_spec


def _spec(**overrides) -> LinkageSpec:
    base = {
        "comparisons": (
            ComparisonSpec("name", "jaro_winkler", (0.96, 0.88)),
            ComparisonSpec("city", "exact"),
        ),
        "blocking_rules": (BlockingRule(("city",)),),
    }
    base.update(overrides)
    return LinkageSpec(**base)


def test_spec_round_trips_through_its_payload():
    spec = _spec(retain_columns=("source_doc",), match_threshold=0.85, seed=11)
    spec.validate()
    assert load_spec(spec.payload()) == spec


def test_a_single_column_spec_is_refused():
    spec = _spec(comparisons=(ComparisonSpec("name", "jaro_winkler", (0.9,)),))
    with pytest.raises(ValueError, match="at least 2 comparisons"):
        spec.validate()


def test_a_spec_with_no_blocking_rule_is_refused():
    with pytest.raises(ValueError, match="at least one blocking rule"):
        _spec(blocking_rules=()).validate()


def test_the_identifier_column_cannot_also_be_compared():
    spec = _spec(
        comparisons=(ComparisonSpec("unique_id", "exact"), ComparisonSpec("city", "exact"))
    )
    with pytest.raises(ValueError, match="cannot also be compared"):
        spec.validate()


@pytest.mark.parametrize("column", ["cluster_id", "node_id", "representative"])
def test_a_column_the_clustering_step_reserves_is_refused(column):
    """Splink's clustering SQL introduces these itself, so the collision surfaces after the fit."""
    compared = _spec(comparisons=(ComparisonSpec(column, "exact"), ComparisonSpec("city", "exact")))
    with pytest.raises(ValueError, match="reserved by the clustering step"):
        compared.validate()
    with pytest.raises(ValueError, match="reserved by the clustering step"):
        _spec(retain_columns=(column,)).validate()


def test_one_column_carries_one_comparison():
    spec = _spec(
        comparisons=(
            ComparisonSpec("name", "exact"),
            ComparisonSpec("name", "jaro_winkler", (0.9,)),
        )
    )
    with pytest.raises(ValueError, match="repeated"):
        spec.validate()


@pytest.mark.parametrize(
    "comparison, message",
    [
        (ComparisonSpec("name", "levenshtein"), "needs at least one threshold"),
        (ComparisonSpec("name", "jaro_winkler", (1.4,)), "similarity scores"),
        (ComparisonSpec("name", "levenshtein", (1.5,)), "whole numbers"),
        (ComparisonSpec("name", "levenshtein", (0.0, 2.0)), "cannot be zero"),
        (ComparisonSpec("a", "array_intersect", (1.0, 1.0)), "repeated thresholds"),
        (ComparisonSpec("v", "cosine", (0.9,)), "must declare its embedding dimension"),
        (ComparisonSpec("name", "shingles", (1.0,)), "unknown comparison kind"),
        (ComparisonSpec("d", "date_difference", (30.0,), date_metric="fortnight"), "date metric"),
    ],
)
def test_a_comparison_that_cannot_be_scored_is_refused(comparison, message):
    with pytest.raises(ValueError, match=message):
        _spec(comparisons=(comparison, ComparisonSpec("city", "exact"))).validate()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"match_threshold": 0.0}, "match_threshold"),
        ({"match_threshold": 1.5}, "match_threshold"),
        ({"max_pairs": 0}, "max_pairs"),
        ({"duckdb_threads": 0}, "duckdb_threads"),
    ],
)
def test_out_of_range_run_settings_are_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        _spec(**overrides).validate()


def test_training_rules_default_to_the_prediction_rules():
    spec = _spec()
    assert spec.em_rules == spec.blocking_rules
    trained = _spec(training_rules=(BlockingRule(("name",)),))
    assert trained.em_rules == (BlockingRule(("name",)),)


def test_a_blocking_rule_reads_from_a_bare_string_or_list():
    assert BlockingRule.from_payload("city") == BlockingRule(("city",))
    assert BlockingRule.from_payload(["city", "name"]) == BlockingRule(("city", "name"))
    assert BlockingRule.from_payload({"expressions": ["city"]}).label == "city"
