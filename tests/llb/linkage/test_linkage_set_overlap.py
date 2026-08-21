"""The three things the seam grew for a set-valued identity decision.

A `set_overlap` comparison (two measures of one element pair), an EXPLODING blocking rule (records
sharing one element are compared), and a pseudo-count floor on the fitted parameters. The
specification half runs in the base install; the engine half needs the extra.
"""

import pytest

from llb.linkage.constants import (
    DEFAULT_MIN_LEVEL_PROBABILITY,
    KIND_JACCARD,
    KIND_SET_OVERLAP,
)
from llb.linkage.fitting import smoothed_model
from llb.linkage.records import column_types
from llb.linkage.comparison_spec import ComparisonSpec
from llb.linkage.spec import BlockingRule, LinkageSpec, load_spec

OVERLAP = ComparisonSpec(
    column="shingles", kind=KIND_SET_OVERLAP, thresholds=(0.8, 0.5), containment_thresholds=(0.9,)
)
EXACT = ComparisonSpec(column="system", kind="exact")
EXPLODING = BlockingRule(("shingles",), arrays_to_explode=("shingles",))


def _spec(**overrides) -> LinkageSpec:
    fields = {
        "comparisons": (OVERLAP, EXACT),
        "blocking_rules": (EXPLODING,),
        "training_rules": (BlockingRule(("system",)),),
    }
    return LinkageSpec(**{**fields, **overrides})


def test_a_set_overlap_comparison_carries_two_ladders_of_one_column():
    spec = _spec()
    spec.validate()
    assert spec.compared_columns == ("shingles", "system")
    assert column_types(spec)["shingles"] == "VARCHAR[]"


def test_containment_thresholds_are_refused_on_every_other_kind():
    comparison = ComparisonSpec(
        column="address", kind=KIND_JACCARD, thresholds=(0.8,), containment_thresholds=(0.9,)
    )
    with pytest.raises(ValueError, match="containment thresholds are only defined"):
        comparison.validate()


def test_a_containment_threshold_outside_the_score_range_is_refused():
    comparison = ComparisonSpec(
        column="shingles", kind=KIND_SET_OVERLAP, thresholds=(0.8,), containment_thresholds=(2.0,)
    )
    with pytest.raises(ValueError, match="containment thresholds"):
        comparison.validate()


def test_a_repeated_containment_threshold_is_refused():
    comparison = ComparisonSpec(
        column="shingles",
        kind=KIND_SET_OVERLAP,
        thresholds=(0.8,),
        containment_thresholds=(0.9, 0.9),
    )
    with pytest.raises(ValueError, match="repeated containment thresholds"):
        comparison.validate()


def test_an_exploded_column_is_typed_as_an_array_even_when_nothing_compares_it():
    spec = LinkageSpec(
        comparisons=(ComparisonSpec(column="name", kind="exact"), EXACT),
        blocking_rules=(BlockingRule(("tokens",), arrays_to_explode=("tokens",)),),
    )
    spec.validate()
    assert spec.exploded_columns == ("tokens",)
    assert column_types(spec)["tokens"] == "VARCHAR[]"


def test_an_exploding_rule_cannot_be_a_training_rule():
    """Splink refuses it, and it fixes no comparison for the others to be trained against."""
    with pytest.raises(ValueError, match="exploding blocking rule cannot train"):
        _spec(training_rules=(EXPLODING,)).validate()


def test_a_blocking_rule_labels_itself_as_exploding():
    assert EXPLODING.explodes and "exploded" in EXPLODING.label
    assert not BlockingRule(("system",)).explodes


def test_the_specification_round_trips_through_its_payload():
    spec = _spec(min_level_probability=1e-4, retain_matching_columns=False)
    spec.validate()
    restored = load_spec(spec.payload())
    assert restored == spec


def test_a_payload_written_before_these_fields_existed_still_loads():
    """A bundle on disk predates every field added here; reading it must not need them."""
    spec = load_spec(
        {
            "comparisons": [
                {"column": "name", "kind": "exact"},
                {"column": "city", "kind": "exact"},
            ],
            "blocking_rules": [{"expressions": ["city"]}],
        }
    )
    assert spec.exploded_columns == ()
    assert spec.min_level_probability == DEFAULT_MIN_LEVEL_PROBABILITY
    assert spec.retain_matching_columns is True


MODEL = {
    "comparisons": [
        {
            "output_column_name": "shingles",
            "comparison_levels": [
                {"label_for_charts": "collapsed", "m_probability": 1e-300, "u_probability": 0.9},
                {"label_for_charts": "trained", "m_probability": 0.7, "u_probability": 0.02},
                {"label_for_charts": "untrained", "m_probability": None, "u_probability": None},
            ],
        }
    ]
}


def test_smoothing_lifts_a_collapsed_estimate_and_leaves_a_trained_one_alone():
    levels = smoothed_model(MODEL, 1e-4)["comparisons"][0]["comparison_levels"]
    assert levels[0]["m_probability"] == 1e-4
    assert levels[1]["m_probability"] == 0.7
    assert levels[1]["u_probability"] == 0.02


def test_smoothing_never_invents_an_estimate_nobody_made():
    levels = smoothed_model(MODEL, 1e-4)["comparisons"][0]["comparison_levels"]
    assert levels[2]["m_probability"] is None and levels[2]["u_probability"] is None


def test_a_zero_floor_is_the_model_unchanged():
    assert smoothed_model(MODEL, 0.0) == dict(MODEL)


def test_a_floor_outside_its_range_is_refused():
    with pytest.raises(ValueError, match="pseudo-count floor"):
        _spec(min_level_probability=0.9).validate()


def test_set_overlap_levels_are_emitted_mutual_first_then_containment():
    """The order a caller's own decision rule applies: Jaccard, and containment as the fallback."""
    pytest.importorskip("splink")
    from llb.linkage.comparisons import (
        CONTAINMENT_LEVEL_PREFIX,
        JACCARD_LEVEL_PREFIX,
        build_comparison,
    )

    levels = build_comparison(OVERLAP).get_comparison("duckdb").as_dict()["comparison_levels"]
    labels = [level.get("label_for_charts", "") for level in levels]
    jaccard_at = [index for index, label in enumerate(labels) if JACCARD_LEVEL_PREFIX in label]
    contained_at = [
        index for index, label in enumerate(labels) if CONTAINMENT_LEVEL_PREFIX in label
    ]
    assert jaccard_at and contained_at and max(jaccard_at) < min(contained_at)


def test_an_exploding_rule_compares_records_sharing_one_element():
    """Two records sharing a single element are a candidate pair; disjoint ones are not."""
    pytest.importorskip("splink")
    pytest.importorskip("duckdb")
    from llb.linkage.engine import duckdb_connection
    from llb.linkage.fitting import count_blocking_comparisons
    from llb.linkage.records import register_records

    spec = LinkageSpec(
        comparisons=(OVERLAP, EXACT),
        blocking_rules=(EXPLODING,),
        training_rules=(BlockingRule(("system",)),),
    )
    records = [
        {"unique_id": "a", "shingles": ["1", "2"], "system": "x"},
        {"unique_id": "b", "shingles": ["2", "3"], "system": "y"},
        {"unique_id": "c", "shingles": ["9"], "system": "z"},
    ]
    with duckdb_connection(1) as con:
        table = register_records(con, records, spec)
        counts = count_blocking_comparisons(con, spec, table)
    assert counts[0].post_filter == 1
