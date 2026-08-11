"""Deterministic constructors and gates for independent-null research."""

import json

import pytest

from conflict_helpers import FIXTURE_CORPUS, bow_vector, fake_store_view
from llb.conflicts.constants import (
    REL_COMPLEMENTARY,
    REL_CONTRADICTS,
    RESEARCH_GENERATION_NEXT,
    RESEARCH_GENERATION_THIRD,
)
from llb.conflicts.null_research import RESEARCH_METHODS, run_null_research
from llb.conflicts.null_research_evaluation import (
    FIXTURE_POSITIVE_DOC_PAIRS,
    fit_labelled_threshold,
    fixture_metrics,
    null_tail_payload,
    rank_baseline,
    threshold_for_fpr,
)
from llb.conflicts.null_research_geometry import (
    cross_corpus_null,
    held_out_document_null,
    permutation_null,
    prepare_geometry,
)
from llb.conflicts.null_research_nextgen import ADVANCED_RESEARCH_METHODS
from llb.conflicts.null_research_report import write_null_research
from llb.conflicts.null_research_third import THIRD_RESEARCH_METHODS
from llb.conflicts.store_access import StoreView
from llb.conflicts.vectorops import VectorSet


def _reference_store(prefix: str = "reference") -> StoreView:
    store = fake_store_view()
    chunks = [{**chunk, "doc_id": f"{prefix}/{chunk['doc_id']}"} for chunk in store.chunks]
    return StoreView(
        index_dir=store.index_dir,
        chunks=chunks,
        vectors=store.vectors,
        meta={**store.meta, "corpus_fingerprint": "reference"},
    )


def test_vector_set_scores_two_corpora_and_applies_an_external_mean():
    left = VectorSet([[1.0, 0.0], [0.0, 1.0]], use_numpy=False)
    right = VectorSet([[1.0, 1.0], [-1.0, 1.0]], use_numpy=False)

    scores = left.cross_similarities(right, [0, 1], [0, 1])

    assert scores == pytest.approx([2**-0.5, -(2**-0.5), 2**-0.5, 2**-0.5])
    assert left.centered([0.5, 0.5]).similarity(0, 1) == pytest.approx(-1.0)


def test_null_constructors_use_the_filtered_fixture_geometry():
    store = fake_store_view()
    geometry = prepare_geometry("fixture", FIXTURE_CORPUS, store)
    reference = prepare_geometry("reference", FIXTURE_CORPUS, _reference_store())

    cross = cross_corpus_null(geometry, reference)
    held_out = held_out_document_null(geometry)
    permuted = permutation_null(
        geometry,
        lambda texts: [bow_vector(text) for text in texts],
        mode="token",
        permutations=2,
        seed=7,
    )

    assert len(cross) == len(geometry.allowed) ** 2
    assert len(held_out) == len(geometry.document_maxima) == 21
    assert len(permuted) == 2 * len(geometry.allowed)
    assert permuted == pytest.approx([1.0] * len(permuted))
    with pytest.raises(ValueError, match="disjoint document ids"):
        cross_corpus_null(geometry, geometry)


def test_threshold_metrics_separate_a_null_tail_from_supervised_fixture_fit():
    maxima = {
        ("a", "b"): 0.9,
        ("a", "c"): 0.8,
        ("b", "c"): 0.1,
    }
    positives = frozenset({("a", "b")})

    threshold = threshold_for_fpr([0.0, 0.1, 0.2, 0.3], 0.25)
    metrics = fixture_metrics(maxima, 0.85, positives)
    fitted, fitted_metrics = fit_labelled_threshold(maxima, positives)
    baseline = rank_baseline([0.1, 0.8, 0.9], maxima, positives, 2)
    tail = null_tail_payload([0.0] * 100 + [1.0], 1.0, 0.01)

    assert threshold == pytest.approx(0.225)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 1.0
    assert fitted == 0.9
    assert fitted_metrics["f1"] == 1.0
    assert baseline["selected_chunk_pairs"] == 2
    assert tail["tail_exceedances"] == 1


def test_full_matrix_is_deterministic_and_writes_machine_and_human_artifacts(tmp_path):
    store = fake_store_view()
    corpus = (FIXTURE_CORPUS, store)
    reference = (FIXTURE_CORPUS, _reference_store())

    def embed(texts):
        return [bow_vector(text) for text in texts]

    summary = run_null_research(
        fixture=corpus,
        hr=corpus,
        goods=corpus,
        reference=reference,
        embed=embed,
        fpr=0.1,
        max_goods_candidates=0,
        permutations=1,
        seed=3,
    )
    paths = write_null_research(tmp_path / "null-research", summary)

    assert summary["verdict"] == "negative"
    assert [method["method"] for method in summary["methods"]] == list(RESEARCH_METHODS)
    assert summary["rank_baseline"]["labelled_positive_pairs"] == len(FIXTURE_POSITIVE_DOC_PAIRS)
    token_tail = summary["methods"][1]["null_tails"]["fixture"]
    assert token_tail["pair_row_tail_resolved"]
    assert not token_tail["tail_resolved"]
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["verdict"] == "negative"
    report = paths["report"].read_text(encoding="utf-8")
    assert "No candidate satisfies all gates" in report
    assert "cross_corpus" in report


def test_next_generation_matrix_tracks_dependence_and_counterfactual_provenance(tmp_path):
    store = fake_store_view()
    corpus = (FIXTURE_CORPUS, store)

    summary = run_null_research(
        fixture=corpus,
        hr=corpus,
        goods=corpus,
        reference=(FIXTURE_CORPUS, _reference_store("general")),
        domain_reference=(FIXTURE_CORPUS, _reference_store("domain")),
        embed=lambda texts: [bow_vector(text) for text in texts],
        generation=RESEARCH_GENERATION_NEXT,
        fpr=0.1,
        max_goods_candidates=0,
        matches_per_reference=1,
        seed=11,
    )
    paths = write_null_research(tmp_path / "next-generation", summary)

    assert summary["research_generation"] == "next"
    assert summary["verdict"] == "negative"
    assert [method["method"] for method in summary["methods"]] == list(ADVANCED_RESEARCH_METHODS)
    matched = summary["methods"][0]
    assert matched["diagnostics"]["fixture"]["reference_domains"] == ["domain", "general"]
    assert (
        matched["null_tails"]["fixture"]["effective_independent_units"]
        == matched["diagnostics"]["fixture"]["unique_source_texts"]
    )
    assert not matched["null_tails"]["fixture"]["tail_resolved"]
    counterfactual = summary["methods"][-1]
    assert (
        counterfactual["null_tails"]["fixture"]["effective_independent_units"]
        == (counterfactual["diagnostics"]["fixture"]["unique_source_texts"])
    )
    assert not counterfactual["gates"]["eligible_as_independent_null"]
    assert summary["control_traces"]
    assert all(trace["trace_verified"] for trace in summary["control_traces"])
    for trace in summary["control_traces"]:
        if trace["dataset"] != "fixture" or trace["edit_type"] != "quantity_or_date_swap":
            continue
        text = store.chunks[trace["source_ordinal"]]["text"]
        if text.startswith("---"):
            assert trace["char_start"] >= text.find("\n---", 3) + len("\n---")
    persisted = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert "control_traces" not in persisted
    assert persisted["control_trace_rows"] == len(summary["control_traces"])
    assert paths["control_traces"].read_text(encoding="utf-8")


def _scripted_adjudicator(relation_by_position: dict[int, str]):
    """A deterministic stand-in for the local model: nth call -> nth scripted relation."""
    calls = {"n": 0}

    def complete(prompt: str) -> str:
        position = calls["n"]
        calls["n"] += 1
        relation = relation_by_position.get(position % 2, REL_COMPLEMENTARY)
        return json.dumps(
            {
                "relation": relation,
                "confidence": 0.9,
                "claim_a": "",
                "claim_b": "",
                "rationale": "scripted",
            }
        )

    return complete


@pytest.fixture(scope="module")
def third_generation(tmp_path_factory):
    """One deterministic third-generation run, shared by the assertions below."""
    store = fake_store_view()
    corpus = (FIXTURE_CORPUS, store)
    summary = run_null_research(
        fixture=corpus,
        hr=corpus,
        goods=corpus,
        reference=(FIXTURE_CORPUS, _reference_store("general")),
        domain_reference=(FIXTURE_CORPUS, _reference_store("domain")),
        embed=lambda texts: [bow_vector(text) for text in texts],
        complete=_scripted_adjudicator({0: REL_CONTRADICTS, 1: REL_COMPLEMENTARY}),
        generation=RESEARCH_GENERATION_THIRD,
        adjudicator_model="scripted-fake",
        fpr=0.1,
        max_goods_candidates=6,
        adjudication_budget=6,
        role_samples_per_type=1,
        seed=5,
    )
    paths = write_null_research(tmp_path_factory.mktemp("third-generation"), summary)
    return summary, paths


def test_third_generation_bounds_the_control_bank_a_usable_tail_would_need(third_generation):
    summary, _ = third_generation

    assert summary["research_generation"] == "third"
    assert summary["parameters"]["adjudicator_model"] == "scripted-fake"
    assert [method["method"] for method in summary["methods"]] == list(THIRD_RESEARCH_METHODS)

    feasibility = summary["feasibility"]["datasets"]["hr"]
    assert feasibility["required_independent_units"] > feasibility["available_independent_units"]
    assert not summary["feasibility"]["feasible"]

    comparable = len(prepare_geometry("reference", FIXTURE_CORPUS, _reference_store()).allowed)
    balanced = summary["methods"][0]
    diagnostics = balanced["diagnostics"]["hr"]
    assert diagnostics["unique_reference_claims"] == comparable * 2
    assert "held_out_domain_membership_auc" in diagnostics
    assert not balanced["gates"]["operating_point_feasible"]
    tail = balanced["null_tails"]["hr"]
    lower, upper = tail["tail_rate_two_way_95"]
    assert lower <= tail["weighted_tail_rate"] <= upper

    identifiability = summary["identifiability"]["hr"]
    assert identifiability["observationally_equivalent_mixtures"] >= 1
    low, high = identifiability["implied_selected_rows_range"]
    assert low <= high


def test_third_generation_measures_claim_precision_and_verifies_control_roles(third_generation):
    summary, paths = third_generation

    precision = summary["claim_precision"]
    curve = precision["datasets"]["hr"]["precision_curve"]
    assert [point["budget"] for point in curve] == [6]
    assert curve[0]["precision"] == pytest.approx(0.5)
    assert curve[0]["two_way_clustered_lcb"] <= curve[0]["precision"]
    assert precision["gates"]["rows_adjudicated"]

    roles = summary["control_role_verification"]
    assert roles["verified_rows"] > 0
    assert set(roles["by_edit_type"]) <= {
        "capitalized_argument_swap",
        "quantity_or_date_swap",
        "modality_flip",
    }

    report = paths["report"].read_text(encoding="utf-8")
    assert "Operating-point feasibility" in report
    assert "Cosine-only identifiability" in report
    assert "Claim-tier precision" in report
    assert "Verified control roles" in report
