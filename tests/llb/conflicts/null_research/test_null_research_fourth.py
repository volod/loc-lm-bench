"""Fourth-generation lanes: in-support synthesis, cross-encoder scoring, conformal certification.

Every model is injected: the local adjudicator is a scripted completer that answers the generation
prompt and the relation prompt differently, and the cross-encoder is a deterministic fake. The
conformal lane needs no model at all, so its coverage assertions are exact.
"""

import json

import pytest

from tests.llb.conflicts.conflict_helpers import FIXTURE_CORPUS, bow_vector, fake_store_view
from llb.conflicts.constants import (
    REL_COMPLEMENTARY,
    REL_CONTRADICTS,
    RESEARCH_GENERATION_FOURTH,
)
from llb.conflicts.null_research.run import run_null_research
from llb.conflicts.null_research.statistics.conformal import (
    certifiable_units,
    conformal_lane,
    tolerance_rank,
)
from llb.conflicts.null_research.controls.cross_encoder import calibration_curve, score_pairs
from llb.conflicts.null_research.generations.fourth import FOURTH_RESEARCH_METHODS
from llb.conflicts.null_research.geometry import prepare_geometry
from llb.conflicts.null_research.report.render import write_null_research
from llb.conflicts.null_research.controls.synthesis import SynthesisError, parse_synthesis
from llb.conflicts.null_research.controls.synthesis_bank import synthesize_bank

CONTROL_PASSAGE = " ".join(f"контрольне-твердження-{index}" for index in range(40))
CONFLICTING_MARKER = "архівне зберігання"


def _scripted_model(conflicting_marker: str | None = None):
    """Answers the synthesis prompt with a passage and the relation prompt with a verdict."""

    def complete(prompt: str) -> str:
        if "You write control passages" in prompt:
            return json.dumps({"passage": CONTROL_PASSAGE})
        relation = (
            REL_CONTRADICTS
            if conflicting_marker is not None and conflicting_marker in prompt
            else REL_COMPLEMENTARY
        )
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


def _fake_cross_encoder(question: str, texts: list[str]) -> list[float]:
    """Deterministic pair score: token overlap, so it tracks the fake encoder's own geometry."""
    left = set(question.casefold().split())
    return [
        len(left & set(text.casefold().split())) / max(1, len(left | set(text.casefold().split())))
        for text in texts
    ]


@pytest.fixture(scope="module")
def fourth_generation(tmp_path_factory):
    """One deterministic fourth-generation run, shared by the assertions below."""
    corpus = (FIXTURE_CORPUS, fake_store_view())
    summary = run_null_research(
        fixture=corpus,
        hr=corpus,
        goods=corpus,
        reference=(FIXTURE_CORPUS, fake_store_view()),
        embed=lambda texts: [bow_vector(text) for text in texts],
        complete=_scripted_model(),
        scorer=_fake_cross_encoder,
        generation=RESEARCH_GENERATION_FOURTH,
        adjudicator_model="scripted-fake",
        cross_encoder_model="fake-cross-encoder",
        fpr=0.1,
        max_goods_candidates=6,
        synthesis_per_document=1,
        cross_encoder_rows=8,
        seed=5,
    )
    paths = write_null_research(tmp_path_factory.mktemp("fourth-generation"), summary)
    return summary, paths


def test_fourth_generation_runs_every_lane_the_third_left_open(fourth_generation):
    summary, _ = fourth_generation

    assert summary["research_generation"] == "fourth"
    assert summary["parameters"]["cross_encoder_model"] == "fake-cross-encoder"
    assert [method["method"] for method in summary["methods"]] == list(FOURTH_RESEARCH_METHODS)

    synthesis = summary["control_synthesis"]["hr"]
    assert synthesis["retained_claims"] > 0
    assert synthesis["conflicting_claims"] == 0
    assert synthesis["scale"]["required_independent_units"] > synthesis["retained_claims"]
    assert synthesis["scale"]["years_to_required_units"] is not None

    in_support = summary["methods"][0]
    assert set(in_support["diagnostics"]["hr"]) >= {"membership_auc", "verified_yield"}
    assert (
        in_support["null_tails"]["hr"]["effective_independent_units"]
        <= (synthesis["retained_claims"])
    )
    assert not in_support["gates"]["operating_point_feasible"]


def test_fourth_generation_certifies_nothing_the_verified_bank_cannot_support(fourth_generation):
    summary, paths = fourth_generation

    certification = summary["tail_certification"]["datasets"]["goods"]
    assert certification["certifiable_units_required"] > certification["verified_units_available"]
    assert not summary["tail_certification"]["certifiable"]

    cross_encoder = summary["methods"][1]
    assert "relation_recall" in cross_encoder["diagnostics"]["fixture"]
    assert cross_encoder["hr"]["baseline_recall"] <= 1.0
    assert summary["verdict"] == "negative"

    report = paths["report"].read_text(encoding="utf-8")
    assert "In-support control synthesis" in report
    assert "Distribution-free tail certification" in report
    assert "Group-split conformal tail inference" in report


def test_a_generated_claim_the_verifier_calls_conflicting_never_enters_the_bank():
    corpus = prepare_geometry("fixture", FIXTURE_CORPUS, fake_store_view())

    clean = synthesize_bank(
        corpus,
        _scripted_model(),
        lambda texts: [bow_vector(text) for text in texts],
        per_document=1,
        required_units=1000,
    )
    rejected = synthesize_bank(
        corpus,
        _scripted_model(CONFLICTING_MARKER),
        lambda texts: [bow_vector(text) for text in texts],
        per_document=1,
        required_units=1000,
    )

    assert clean.payload["conflicting_claims"] == 0
    assert rejected.payload["conflicting_claims"] > 0
    assert len(rejected.retained) == len(clean.retained) - rejected.payload["conflicting_claims"]
    assert all(control.text == CONTROL_PASSAGE for control in clean.retained)
    with pytest.raises(SynthesisError):
        parse_synthesis('{"passage": "too short"}')


def test_conformal_certification_refuses_a_tail_its_unit_count_cannot_carry():
    lane = conformal_lane(unit_grid=(25, 100), replications=20, draws=25, seed=3)
    scenarios = {payload["scenario"]: payload for payload in lane["scenarios"]}

    assert certifiable_units(0.05, 0.95) == 59
    assert tolerance_rank(25, 0.05, 0.95) is None
    assert tolerance_rank(100, 0.05, 0.95) == 2

    duplicate = scenarios["duplicate_reference_reuse"]["grid"]
    assert duplicate[0]["conformal_claim_rate"] == 0.0
    assert duplicate[1]["conformal_bound_coverage"] >= 0.9
    shifted = scenarios["domain_shift"]["grid"][1]
    assert shifted["conformal_bound_coverage"] < duplicate[1]["conformal_bound_coverage"]
    assert not lane["gates"]["survives_domain_shift"]
    assert not lane["gates"]["accepted"]


def test_cross_encoder_helpers_batch_by_left_passage_and_bin_by_score():
    seen: list[tuple[str, int]] = []

    def counting_scorer(question: str, texts: list[str]) -> list[float]:
        seen.append((question, len(texts)))
        return _fake_cross_encoder(question, texts)

    scores = score_pairs(counting_scorer, [("a b", "a"), ("a b", "b c"), ("z", "z")])

    assert len(scores) == 3
    assert seen == [("a b", 2), ("z", 1)]

    class _Scored:
        def labelled(self):
            return [(0.1, False), (0.2, False), (0.3, False), (0.4, True)] * 2

    curve = calibration_curve(_Scored())
    assert curve["resolved"] and curve["monotone"]
    assert curve["bins"][-1]["actionable_fraction"] > curve["bins"][0]["actionable_fraction"]
