"""The shadow lane against the committed drafting fixture: does it reproduce today's drops?

The gate the lane must clear is not "the model looks reasonable" -- it is that a cut EXISTS which
drops exactly what the shipped exact-question and cosine tiers drop. A model that loses an obvious
duplicate is rejected here, and the pairs where the two policies disagree are the list the reviewer
labelling task consumes.
"""

import json

import pytest
from llb.prep.ontology.linkage.constants import SHADOW_MODE
from llb.prep.ontology.pipeline.deduplication import deduplicate_drafts
from tests.llb.prep.ontology.linkage.gold_item_fixtures import (
    HashingBagEmbedder,
    draft_batch,
    draft_labels,
    write_prior_bundle,
)


def _dedup(tmp_path, *, shadow: bool, bundle_dir=None):
    pytest.importorskip("splink")
    pytest.importorskip("duckdb")
    bundle, prior = write_prior_bundle(tmp_path)
    items = draft_batch(prior)
    return deduplicate_drafts(
        items,
        draft_labels(items),
        dedup_against=[bundle],
        embedder=HashingBagEmbedder(),
        linkage_shadow=shadow,
        bundle_dir=bundle_dir,
    )


def _points(report) -> dict:
    return {point["name"]: point for point in report["linkage_shadow"]["operating_points"]}


@pytest.mark.heavy_env
def test_the_shadow_lane_changes_no_drop(tmp_path):
    kept, labels, report = _dedup(tmp_path / "off", shadow=False)
    shadow_kept, shadow_labels, shadow_report = _dedup(tmp_path / "on", shadow=True)

    assert [item.id for item in kept] == [item.id for item in shadow_kept]
    assert set(labels) == set(shadow_labels)
    assert report["dropped_ids"] == shadow_report["dropped_ids"]
    assert "linkage_shadow" not in report


@pytest.mark.heavy_env
def test_a_cut_exists_that_reproduces_every_shipped_drop(tmp_path):
    _kept, _labels, report = _dedup(tmp_path, shadow=True)
    shadow = report["linkage_shadow"]

    assert shadow["enabled"] and shadow["mode"] == SHADOW_MODE
    assert shadow["n_shipped_drops"] == len(report["dropped_ids"]) > 0
    assert shadow["n_shipped_drops_scored"] == shadow["n_shipped_drops"]
    assert shadow["unreproducible_drop_ids"] == [] and shadow["unscored_drop_ids"] == []
    assert shadow["separates_shipped_decisions"] is True

    provisional = _points(report)["provisional"]
    assert provisional["n_model_drops"] == len(report["dropped_ids"])
    assert provisional["n_disagree"] == 0
    assert provisional["disagreements"] == []


@pytest.mark.heavy_env
def test_every_drop_row_names_the_agreements_behind_it(tmp_path):
    _kept, _labels, report = _dedup(tmp_path, shadow=True)

    for row in report["dropped_detail"]:
        assert 0.0 <= row["match_probability"] <= 1.0
        assert set(row["agreements"]) == {
            "question_vector",
            "answer_vector",
            "source_doc_id",
            "span_blocks",
            "question_type",
        }
        assert "question_vector" in row["agreements"]
        assert row["agreements"]["source_doc_id"]
    # a drop against a prior bundle names the prior ITEM, not only its text
    prior_drops = [row for row in report["dropped_detail"] if "nearest_prior_question" in row]
    assert prior_drops and all(row.get("nearest_prior_id") for row in prior_drops)


@pytest.mark.heavy_env
def test_the_default_cut_disagreements_are_the_labelling_input(tmp_path):
    _kept, _labels, report = _dedup(tmp_path, shadow=True)
    default = _points(report)["linkage-default"]

    assert default["n_disagree"] > 0, "the seam's default cut must be priced against the constant"
    for row in default["disagreements"]:
        assert {row["model"], row["constant"]} == {"drop", "keep"}
        assert row["nearest"]["id"] and row["nearest"]["question"]
        assert row["question"] and row["agreements"]
    # the paraphrases are the interesting half: kept by one cosine, matched on every other field
    assert any(row["id"].startswith("para-") for row in default["disagreements"])


@pytest.mark.heavy_env
def test_the_fitted_model_lands_in_the_bundle(tmp_path):
    out = tmp_path / "bundle"
    out.mkdir()
    _kept, _labels, report = _dedup(tmp_path, shadow=True, bundle_dir=out)

    written = {path.name for path in (out / "linkage").iterdir()}
    assert written == {
        "settings.json",
        "blocking_counts.json",
        "match_parameters.json",
        "model.json",
        "pairs.jsonl",
        "clusters.jsonl",
    }
    settings = json.loads((out / "linkage" / "settings.json").read_text(encoding="utf-8"))
    assert settings["metadata"]["mode"] == SHADOW_MODE
    assert settings["summary"]["trained_from_labels"] is False
    assert report["linkage_shadow"]["artifacts"]["model"].endswith("model.json")


@pytest.mark.heavy_env
def test_an_empty_batch_declines_instead_of_fitting(tmp_path):
    bundle, _prior = write_prior_bundle(tmp_path)
    _kept, _labels, report = deduplicate_drafts(
        [],
        {},
        dedup_against=[bundle],
        embedder=HashingBagEmbedder(),
        linkage_shadow=True,
    )
    assert report["linkage_shadow"]["enabled"] is False
    assert "drafted no items" in report["linkage_shadow"]["reason"]


@pytest.mark.heavy_env
def test_a_table_too_small_to_fit_declines_instead_of_publishing_noise(tmp_path):
    from llb.goldset.schema import dump_goldset

    bundle, prior = write_prior_bundle(tmp_path)
    dump_goldset(prior[:2], bundle / "goldset.jsonl")
    items = draft_batch(prior)[:2]
    _kept, _labels, report = deduplicate_drafts(
        items,
        draft_labels(items),
        dedup_against=[bundle],
        embedder=HashingBagEmbedder(),
        linkage_shadow=True,
    )
    shadow = report["linkage_shadow"]
    assert shadow["enabled"] is False and "floor" in shadow["reason"]


def test_the_drafting_entrypoint_carries_the_shadow_setting(tmp_path):
    """The pipeline wiring: the flag reaches the dedup stage and lands in provenance."""
    from llb.prep.ontology.endpoints.config import (
        EndpointCompleters,
        EndpointConfig,
        EndpointPlan,
    )
    from llb.prep.ontology.pipeline.run import draft_goldset
    from tests.llb.prep.ontology.ontology_yield_helpers import (
        CHAIN_DOC,
        FakeEmbedder,
        _chain_endpoint,
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chain.md").write_text(CHAIN_DOC, encoding="utf-8")
    plan = EndpointPlan.single(EndpointConfig(kind="local", model="fake"))
    completers = EndpointCompleters.single(_chain_endpoint)
    first = tmp_path / "first"
    draft_goldset(corpus, plan, completers=completers, max_items=50, out_dir=first)
    second = tmp_path / "second"
    result = draft_goldset(
        corpus,
        plan,
        completers=completers,
        max_items=50,
        out_dir=second,
        dedup_against=[first],
        dedup_embedder=FakeEmbedder(),
        dedup_linkage_shadow=True,
    )

    assert result.dedup_report is not None
    # this fixture drafts a handful of rows, which is below the fit floor -- the wiring is what is
    # under test, and a declined lane writes no bundle
    assert result.dedup_report["linkage_shadow"]["enabled"] is False
    assert not (second / "linkage").exists()
    provenance = json.loads((second / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["settings"]["dedup_linkage_shadow"] is True
