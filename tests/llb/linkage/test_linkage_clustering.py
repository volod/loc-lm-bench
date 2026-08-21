"""Re-clustering one fit's pairs at an arbitrary cut, and matching what the engine resolved."""

import pytest
from llb.core.paths import PROJECT_ROOT
from llb.linkage.clustering import cluster_pairs
from llb.linkage.model import LinkagePair
from llb.linkage.records import read_records
from llb.linkage.run import read_spec_file

SAMPLES = PROJECT_ROOT / "samples" / "linkage"


def _pair(left: str, right: str, probability: float) -> LinkagePair:
    return LinkagePair(
        left_id=left, right_id=right, match_probability=probability, match_weight=0.0
    )


def _grouped(clusters):
    return {cluster.cluster_id: cluster.record_ids for cluster in clusters}


def test_every_record_lands_in_exactly_one_cluster():
    clusters = cluster_pairs(["a", "b", "c"], [_pair("a", "b", 0.99)], 0.9)
    assert _grouped(clusters) == {"a": ("a", "b"), "c": ("c",)}


def test_a_cut_above_a_pair_leaves_both_records_singletons():
    clusters = cluster_pairs(["a", "b"], [_pair("a", "b", 0.8)], 0.9)
    assert _grouped(clusters) == {"a": ("a",), "b": ("b",)}


def test_a_third_record_merges_a_pair_that_scored_below_the_cut():
    """The connected-components property a single pairwise threshold cannot express."""
    pairs = [_pair("a", "b", 0.95), _pair("b", "c", 0.95), _pair("a", "c", 0.10)]
    clusters = cluster_pairs(["a", "b", "c"], pairs, 0.9)
    assert _grouped(clusters) == {"a": ("a", "b", "c")}


def test_the_cluster_id_is_the_smallest_member_whatever_order_the_pairs_arrive_in():
    forward = cluster_pairs(["a", "b", "c"], [_pair("b", "c", 0.99), _pair("a", "b", 0.99)], 0.9)
    reverse = cluster_pairs(["c", "b", "a"], [_pair("a", "b", 0.99), _pair("b", "c", 0.99)], 0.9)
    assert _grouped(forward) == _grouped(reverse) == {"a": ("a", "b", "c")}


def test_a_pair_naming_a_record_the_table_does_not_hold_is_an_error():
    with pytest.raises(KeyError, match="not in the record table"):
        cluster_pairs(["a"], [_pair("a", "ghost", 0.99)], 0.9)


def test_lowering_the_cut_never_splits_a_cluster():
    ids = ["a", "b", "c", "d"]
    pairs = [_pair("a", "b", 0.99), _pair("c", "d", 0.5)]
    tight = {frozenset(c.record_ids) for c in cluster_pairs(ids, pairs, 0.9)}
    loose = {frozenset(c.record_ids) for c in cluster_pairs(ids, pairs, 0.4)}
    assert all(any(member <= wider for wider in loose) for member in tight)


@pytest.mark.heavy_env
def test_it_reproduces_what_the_engine_clustered_at_the_specification_threshold():
    pytest.importorskip("splink")
    pytest.importorskip("duckdb")
    from llb.linkage.engine import run_linkage

    spec = read_spec_file(SAMPLES / "entity_spec_uk.json")
    records = read_records(SAMPLES / "entity_records_uk.jsonl")
    result = run_linkage(records, spec)
    replayed = cluster_pairs(
        [str(record["unique_id"]) for record in records], result.pairs, spec.match_threshold
    )
    assert _grouped(replayed) == _grouped(result.clusters)
