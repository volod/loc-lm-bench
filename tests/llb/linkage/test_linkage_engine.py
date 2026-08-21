"""The seam against its committed fixture: does it recover a known identity structure, twice.

`samples/linkage/entity_records_uk.jsonl` carries 36 Ukrainian institution records over 12 real
entities, each record's true entity in a `truth_entity_id` field the specification neither
compares nor blocks on. The structure is genuinely hard for any single feature: inflected and
homoglyph name variants, abbreviated addresses, typo'd registry codes, and distinct institutions
that share a city and most of their name.
"""

import collections
import json

import pytest
from llb.core.paths import PROJECT_ROOT
from llb.linkage.artifacts import (
    linkage_dir,
    read_pairs,
    read_saved_model,
    read_saved_spec,
    write_linkage_artifacts,
)
from llb.linkage.constants import (
    ACCURACY_FILE,
    BLOCKING_COUNTS_FILE,
    CLUSTERS_FILE,
    MATCH_PARAMETERS_FILE,
    MODEL_FILE,
    PAIRS_FILE,
    SETTINGS_FILE,
)
from llb.linkage.records import read_labels, read_records
from llb.linkage.run import read_spec_file

SAMPLES = PROJECT_ROOT / "samples" / "linkage"
RECORDS = SAMPLES / "entity_records_uk.jsonl"
SPEC = SAMPLES / "entity_spec_uk.json"
LABELS = SAMPLES / "entity_labels_uk.jsonl"


def _engine():
    pytest.importorskip("splink")
    pytest.importorskip("duckdb")
    from llb.linkage import engine

    return engine


def _fixture():
    records = read_records(RECORDS)
    return read_spec_file(SPEC), records, {r["unique_id"]: r["truth_entity_id"] for r in records}


def _clusters_by_truth(result, truth):
    grouped = collections.defaultdict(set)
    for cluster in result.clusters:
        for record_id in cluster.record_ids:
            grouped[truth[record_id]].add(cluster.cluster_id)
    return grouped


@pytest.mark.heavy_env
def test_the_fit_recovers_the_fixture_cluster_structure():
    spec, records, truth = _fixture()
    result = _engine().run_linkage(records, spec)
    grouped = _clusters_by_truth(result, truth)
    # Every entity lands in exactly one cluster, and no cluster mixes two entities.
    assert sorted(grouped) == sorted(set(truth.values()))
    assert all(len(ids) == 1 for ids in grouped.values())
    assert len(result.clusters) == len(set(truth.values()))
    for cluster in result.clusters:
        assert len({truth[record_id] for record_id in cluster.record_ids}) == 1


@pytest.mark.heavy_env
def test_no_cross_entity_pair_outscores_a_same_entity_pair():
    spec, records, truth = _fixture()
    result = _engine().run_linkage(records, spec)
    matched = {(p.left_id, p.right_id) for p in result.matched_pairs}
    assert matched, "the fixture must produce matches at its documented threshold"
    assert all(truth[left] == truth[right] for left, right in matched)


@pytest.mark.heavy_env
def test_two_fits_of_the_same_table_are_byte_identical():
    spec, records, _ = _fixture()
    engine = _engine()
    first, second = engine.run_linkage(records, spec), engine.run_linkage(records, spec)
    assert [p.payload() for p in first.pairs] == [p.payload() for p in second.pairs]
    assert [c.payload() for c in first.clusters] == [c.payload() for c in second.clusters]
    assert first.trained_model == second.trained_model


@pytest.mark.heavy_env
def test_a_saved_model_re_scores_the_same_pairs_to_identical_probabilities():
    spec, records, _ = _fixture()
    engine = _engine()
    fitted = engine.run_linkage(records, spec)
    replayed = engine.replay_linkage(records, spec, fitted.trained_model)
    assert [p.payload() for p in replayed.pairs] == [p.payload() for p in fitted.pairs]
    assert [c.payload() for c in replayed.clusters] == [c.payload() for c in fitted.clusters]
    assert not replayed.trained_from_labels


@pytest.mark.heavy_env
def test_blocking_counts_are_recorded_for_every_rule_before_the_fit():
    spec, records, _ = _fixture()
    result = _engine().run_linkage(records, spec)
    assert [c.rule for c in result.blocking_counts] == [r.label for r in spec.blocking_rules]
    for count in result.blocking_counts:
        assert 0 < count.post_filter <= count.pre_filter
    assert len(result.pairs) <= sum(c.post_filter for c in result.blocking_counts)


@pytest.mark.heavy_env
def test_the_unsupervised_fit_trains_every_comparison_level():
    spec, records, _ = _fixture()
    result = _engine().run_linkage(records, spec)
    fitted = [p for p in result.match_parameters if p.m is not None]
    assert {p.comparison for p in fitted} == set(spec.compared_columns)
    assert result.untrained_levels == ()
    assert all(0.0 <= p.m <= 1.0 and 0.0 <= p.u <= 1.0 for p in fitted)


@pytest.mark.heavy_env
def test_reviewer_labels_fit_m_and_produce_an_operating_point():
    spec, records, truth = _fixture()
    labels = read_labels(LABELS, spec)
    result = _engine().run_linkage(records, spec, labels)
    assert result.trained_from_labels
    assert result.accuracy
    best = result.best_accuracy()
    assert best is not None and best.precision == 1.0 and best.recall == 1.0
    # A label set covers only the levels its matches exhibit; the rest are reported as unmeasured
    # rather than silently defaulted.
    assert all(p.m is None or p.u is None for p in result.untrained_levels)
    # The run's own cut is scored exactly, and the clustering recovers what the pairwise cut
    # leaves apart -- the fixture's transitive merges are the whole reason both are reported.
    pairwise, clustered = result.pair_operating_point, result.cluster_operating_point
    assert pairwise is not None and clustered is not None
    assert pairwise.threshold == clustered.threshold == spec.match_threshold
    assert pairwise.precision == clustered.precision == 1.0
    assert clustered.recall == 1.0
    assert pairwise.false_negatives > clustered.false_negatives == 0
    # The labelled fit must still recover the same identity structure.
    assert all(len(ids) == 1 for ids in _clusters_by_truth(result, truth).values())


@pytest.mark.heavy_env
def test_the_bundle_holds_every_documented_artifact_and_reads_back(tmp_path):
    spec, records, _ = _fixture()
    engine = _engine()
    result = engine.run_linkage(records, spec, read_labels(LABELS, spec))
    paths = write_linkage_artifacts(result, tmp_path, {"mode": "fit"})
    out = linkage_dir(tmp_path)
    for name in (
        SETTINGS_FILE,
        BLOCKING_COUNTS_FILE,
        MATCH_PARAMETERS_FILE,
        MODEL_FILE,
        PAIRS_FILE,
        CLUSTERS_FILE,
        ACCURACY_FILE,
    ):
        assert (out / name).is_file(), name
    assert set(paths) == {
        "settings",
        "blocking_counts",
        "match_parameters",
        "model",
        "pairs",
        "clusters",
        "accuracy",
    }
    assert read_saved_spec(tmp_path) == spec
    assert read_saved_model(tmp_path) == result.trained_model
    assert read_pairs(tmp_path) == [p.payload() for p in result.pairs]
    accuracy = json.loads((out / ACCURACY_FILE).read_text(encoding="utf-8"))
    assert accuracy["curve"] and accuracy["pair_operating_point"]["precision"] == 1.0
    assert accuracy["cluster_operating_point"]["recall"] == 1.0
    settings = json.loads((out / SETTINGS_FILE).read_text(encoding="utf-8"))
    assert settings["metadata"] == {"mode": "fit"}
    assert settings["summary"] == result.summary()
    # A replay from the WRITTEN bundle reproduces the probabilities, which is what makes an
    # identity decision reproducible without the original process.
    replayed = engine.replay_linkage(records, read_saved_spec(tmp_path), read_saved_model(tmp_path))
    assert [p.payload() for p in replayed.pairs] == [p.payload() for p in result.pairs]


@pytest.mark.heavy_env
def test_an_unlabelled_run_writes_no_accuracy_curve(tmp_path):
    spec, records, _ = _fixture()
    result = _engine().run_linkage(records, spec)
    paths = write_linkage_artifacts(result, tmp_path)
    assert "accuracy" not in paths
    assert not (linkage_dir(tmp_path) / ACCURACY_FILE).exists()
    assert result.pair_operating_point is None
    assert result.cluster_operating_point is None
