"""Publishing an identity decision: where the bundle lands and what it reports."""

import json

import pytest
from llb.core.paths import PROJECT_ROOT
from llb.linkage.artifacts import linkage_dir
from llb.linkage.constants import METHOD, SETTINGS_FILE
from llb.linkage.model import (
    AccuracyPoint,
    BlockingCount,
    LinkageCluster,
    LinkagePair,
    LinkageResult,
    MatchParameter,
)
from llb.linkage.run import (
    bundle_dir,
    format_accuracy,
    format_pairs,
    format_summary,
    read_spec_file,
)
from llb.linkage.comparison_spec import ComparisonSpec
from llb.linkage.spec import BlockingRule, LinkageSpec

SAMPLES = PROJECT_ROOT / "samples" / "linkage"

SPEC = LinkageSpec(
    comparisons=(ComparisonSpec("name", "exact"), ComparisonSpec("city", "exact")),
    blocking_rules=(BlockingRule(("city",)),),
    match_threshold=0.9,
)

RESULT = LinkageResult(
    spec=SPEC,
    blocking_counts=(BlockingCount("city", 10, 6),),
    match_parameters=(MatchParameter("name", "Exact match on name", 0.8, 0.01),),
    pairs=(LinkagePair("a", "b", 0.99, 6.6, {"name": 2}),),
    clusters=(LinkageCluster("a", ("a", "b")), LinkageCluster("c", ("c",))),
    trained_model={"comparisons": []},
    n_records=3,
)


def test_the_committed_sample_specification_is_valid():
    spec = read_spec_file(SAMPLES / "entity_spec_uk.json")
    assert len(spec.comparisons) == 6
    assert spec.match_threshold == 0.9
    assert spec.duckdb_threads == 1  # the sample must stay byte-reproducible


def test_the_sample_records_carry_the_truth_label_outside_the_specification():
    spec = read_spec_file(SAMPLES / "entity_spec_uk.json")
    records = [
        json.loads(line)
        for line in (SAMPLES / "entity_records_uk.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 36
    assert len({r["truth_entity_id"] for r in records}) == 12
    # The truth field must be invisible to the model: neither compared nor retained.
    assert "truth_entity_id" not in spec.compared_columns
    assert "truth_entity_id" not in spec.retain_columns


def test_a_bundle_lands_under_its_method_and_run(tmp_path):
    named = bundle_dir(tmp_path, run="run-1")
    assert named == tmp_path / METHOD / "run-1"
    assert linkage_dir(named) == named / "linkage"
    stamped = bundle_dir(tmp_path)
    assert stamped.parent == tmp_path / METHOD and stamped.name != "run-1"


def test_the_summary_names_the_proposed_merges_and_refuses_a_conflict_reading():
    report = format_summary(RESULT)
    assert "records=3" in report and "clusters=2" in report
    assert "city: 6 comparisons to score (10 before filters)" in report
    assert "a: a, b" in report
    assert "not contradiction verdicts" in report


def test_an_unmeasured_level_is_named_in_the_report():
    result = LinkageResult(
        **{
            **RESULT.__dict__,
            "match_parameters": (MatchParameter("city", "Exact match on city", None, 0.3),),
        }
    )
    assert "1 comparison level(s) had no estimate" in format_summary(result)
    assert "city/Exact match on city" in format_summary(result)
    assert "had no estimate" not in format_summary(RESULT)


def test_the_pair_report_shows_the_agreement_behind_the_probability():
    report = format_pairs(RESULT)
    assert "a ~ b: p=0.9900" in report and "[name=2]" in report


def test_without_labels_the_report_says_there_is_no_operating_point():
    assert "RANKED CANDIDATE LIST" in format_accuracy(RESULT)


def test_with_labels_the_report_prints_the_curve_and_the_recommendation():
    labelled = LinkageResult(
        **{
            **RESULT.__dict__,
            "accuracy": (
                AccuracyPoint(0.50, 10, 4, 6, 0, 0.714, 1.0, 0.833),
                AccuracyPoint(0.85, 10, 0, 10, 0, 1.0, 1.0, 1.0),
            ),
            "pair_operating_point": AccuracyPoint(0.9, 8, 0, 10, 2, 1.0, 0.8, 0.889),
            "cluster_operating_point": AccuracyPoint(0.9, 10, 0, 10, 0, 1.0, 1.0, 1.0),
        }
    )
    report = format_accuracy(labelled)
    assert "labelled accuracy curve" in report
    assert "best f1 on the curve is at threshold 0.850000" in report
    # The pairwise cut and what the clustering actually merged are reported separately.
    assert "cut at 0.9, pairwise: precision 1.000, recall 0.800" in report
    assert "cut at 0.9, after clustering: precision 1.000, recall 1.000" in report


@pytest.mark.heavy_env
def test_a_blocking_rule_over_a_missing_column_names_the_columns_it_could_use():
    pytest.importorskip("splink")
    from llb.linkage.engine import run_linkage

    spec = LinkageSpec(
        comparisons=(ComparisonSpec("name", "exact"), ComparisonSpec("code", "exact")),
        blocking_rules=(BlockingRule(("city",)),),
    )
    with pytest.raises(ValueError, match="retain_columns"):
        run_linkage([{"unique_id": "r1", "name": "a", "code": "1"}], spec)


@pytest.mark.heavy_env
def test_the_cli_fits_and_then_replays_into_two_bundles(tmp_path):
    pytest.importorskip("splink")
    from llb.linkage.run import link_records, replay_records

    fitted = link_records(
        SAMPLES / "entity_records_uk.jsonl",
        SAMPLES / "entity_spec_uk.json",
        tmp_path,
        labels_path=SAMPLES / "entity_labels_uk.jsonl",
        run="fit",
    )
    assert fitted.out_dir == tmp_path / METHOD / "fit"
    settings = json.loads((linkage_dir(fitted.out_dir) / SETTINGS_FILE).read_text(encoding="utf-8"))
    assert settings["metadata"]["mode"] == "fit"
    assert settings["metadata"]["labels"].endswith("entity_labels_uk.jsonl")

    replayed = replay_records(
        SAMPLES / "entity_records_uk.jsonl", fitted.out_dir, tmp_path, run="replay"
    )
    assert replayed.out_dir == tmp_path / METHOD / "replay"
    assert [p.payload() for p in replayed.result.pairs] == [
        p.payload() for p in fitted.result.pairs
    ]
    replay_settings = json.loads(
        (linkage_dir(replayed.out_dir) / SETTINGS_FILE).read_text(encoding="utf-8")
    )
    assert replay_settings["metadata"]["mode"] == "replay"
    # How the model was fitted is not recoverable from model.json, so the replay carries the
    # source bundle's summary forward rather than reporting an unsupervised fit.
    assert replay_settings["summary"]["trained_from_labels"] is True
