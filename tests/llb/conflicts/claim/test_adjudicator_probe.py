"""The frozen adjudicator-calibration probe: its two tiers, what they gate, and what they report.

The probe answers two different questions with two different difficulties. The `base` tier is a
FLOOR -- it rejects an adjudicator that is broken -- and it is the only tier that gates, so the
tests here pin exactly that: the floor decides whether precision may be printed, the `hard` tier is
measured and reported beside it without deciding anything, and dropping the floor from a run leaves
nothing to gate rather than silently passing.

The other half is the probe FILE. Its labels are frozen, so every one of them must resolve to real
corpus bytes and every tier must stay balanced; a tier that drifted to mostly-actionable would let
an adjudicator that says "actionable" to everything read as calibrated.
"""

import json

import pytest
from typer.testing import CliRunner

from llb.cli.app import app
from llb.conflicts.claim.calibration import (
    MIN_ADJUDICATOR_ACCURACY_LCB,
    TIER_ACCURACY_GATES,
    calibrate_adjudicator,
)
from llb.conflicts.claim.probe import (
    BASE_TIER,
    HARD_TIER,
    PROBE_TIERS,
    load_calibration_probe,
)
from llb.conflicts.report.calibration import calibration_report
from tests.llb.conflicts.conflict_helpers import FIXTURE_CORPUS, probe_aware
from tests.llb.conflicts.test_audit import scripted

PROBE_FILE = FIXTURE_CORPUS.parent / "adjudicator_probe.json"
TIER_SIZES = {BASE_TIER: 24, HARD_TIER: 16}


# --- the probe file ---------------------------------------------------------------------------


def test_every_tier_resolves_to_corpus_passages_and_is_half_actionable():
    probe = load_calibration_probe()
    assert probe.tiers == PROBE_TIERS
    for tier, size in TIER_SIZES.items():
        pairs = probe.pairs_of(tier)
        assert len(pairs) == size
        assert sum(pair.actionable for pair in pairs) == size // 2, f"{tier} must be half and half"
        for pair in pairs:
            assert pair.left_text.startswith("#") and pair.right_text.startswith("#")
            assert len(pair.left_text.split()) > 5 and len(pair.right_text.split()) > 5


def test_no_two_probe_pairs_show_the_adjudicator_the_same_prompt():
    """Two identical prompts are one observation counted twice, whatever tier they sit in."""
    probe = load_calibration_probe()
    prompts = [(pair.left_text, pair.right_text) for pair in probe.pairs]
    assert len(set(prompts)) == len(prompts)
    assert len({pair.pair_id for pair in probe.pairs}) == len(probe.pairs)


def test_the_hard_tier_repeats_no_passage_pair_in_either_direction():
    """The base tier carries two order-swapped pairs, which read the same two passages twice with
    A and B exchanged. That is an order-symmetry check rather than a second observation, so the
    separator -- the tier a gate decision would rest on -- deliberately carries none of them."""
    probe = load_calibration_probe()
    unordered = {
        tier: [tuple(sorted([pair.left_text, pair.right_text])) for pair in probe.pairs_of(tier)]
        for tier in probe.tiers
    }
    assert len(set(unordered[HARD_TIER])) == len(unordered[HARD_TIER])
    assert not set(unordered[HARD_TIER]) & set(unordered[BASE_TIER])


def test_each_tier_reads_its_own_corpus():
    probe = load_calibration_probe()
    assert probe.corpora[BASE_TIER].endswith("conflicts_uk_v1/corpus")
    assert probe.corpora[HARD_TIER].endswith("conflicts_uk_v1/probe_hard")
    base_text = {pair.left_text for pair in probe.pairs_of(BASE_TIER)}
    hard_text = {pair.left_text for pair in probe.pairs_of(HARD_TIER)}
    assert not base_text & hard_text, "the separator must not be drawn from the detector fixture"


def test_a_single_tier_can_be_adjudicated_on_its_own():
    probe = load_calibration_probe(tiers=(HARD_TIER,))
    assert probe.tiers == (HARD_TIER,)
    assert set(probe.corpora) == {HARD_TIER}
    assert len(probe.pairs) == TIER_SIZES[HARD_TIER]


def test_probe_refuses_a_tier_it_does_not_carry():
    with pytest.raises(SystemExit, match="has no tier trivial"):
        load_calibration_probe(tiers=("trivial",))


def test_probe_refuses_a_heading_the_fixture_does_not_have(tmp_path):
    payload = json.loads(PROBE_FILE.read_text("utf-8"))
    payload["tiers"][0]["pairs"][0]["a"]["heading"] = "## Розділ 9. Немає такого"
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="has no heading"):
        load_calibration_probe(path)


def test_a_single_tier_probe_file_still_loads_as_the_base_tier(tmp_path):
    """The pre-tier file shape is one `base` tier, so an operator's own probe keeps working."""
    payload = json.loads(PROBE_FILE.read_text("utf-8"))
    base = payload["tiers"][0]
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {"probe_id": "flat", "corpus": base["corpus"], "pairs": base["pairs"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    probe = load_calibration_probe(path)
    assert probe.tiers == (BASE_TIER,) and len(probe.pairs) == TIER_SIZES[BASE_TIER]


# --- what each tier decides -------------------------------------------------------------------


def test_only_the_floor_tier_gates():
    assert set(TIER_ACCURACY_GATES) == {BASE_TIER}
    assert TIER_ACCURACY_GATES[BASE_TIER] == MIN_ADJUDICATOR_ACCURACY_LCB


def test_a_frozen_label_adjudicator_calibrates_on_every_tier():
    calibration = calibrate_adjudicator(load_calibration_probe(), probe_aware(scripted))
    assert calibration["accuracy"] == 1.0
    assert calibration["calibrated"] and not calibration["gate_failures"]
    for tier, size in TIER_SIZES.items():
        block = calibration["tiers"][tier]
        assert (block["probe_pairs"], block["accuracy"]) == (size, 1.0)
        assert block["recall_on_actionable"] == block["specificity_on_complementary"] == 1.0
    assert calibration["tier_separation"]["scored_tiers"][HARD_TIER]["delta_from_floor"] == 0.0


def test_an_adjudicator_that_calls_everything_actionable_does_not_calibrate():
    calibration = calibrate_adjudicator(
        load_calibration_probe(), probe_aware(scripted, correct=False)
    )
    assert calibration["accuracy"] == 0.5
    assert not calibration["calibrated"]
    assert calibration["tiers"][BASE_TIER]["specificity_on_complementary"] == 0.0
    assert calibration["tiers"][BASE_TIER]["accuracy_wilson_95"][0] < MIN_ADJUDICATOR_ACCURACY_LCB
    assert "base-tier accuracy 0.5" in calibration["gate_failures"][0]


def test_losing_only_the_hard_tier_is_reported_but_does_not_suppress_precision():
    """The gate is a floor, so a weaker adjudicator is measured rather than refused."""
    calibration = calibrate_adjudicator(
        load_calibration_probe(), probe_aware(scripted, wrong_tiers=(HARD_TIER,))
    )
    assert calibration["calibrated"], "the floor is what gates, and this adjudicator holds it"
    assert calibration["tiers"][BASE_TIER]["passed"] is True
    assert calibration["tiers"][HARD_TIER]["passed"] is None
    assert calibration["tiers"][HARD_TIER]["specificity_on_complementary"] == 0.0
    separation = calibration["tier_separation"]["scored_tiers"][HARD_TIER]
    assert separation["delta_from_floor"] == -0.5, "the hard tier is what tells them apart"


def test_dropping_the_floor_tier_leaves_nothing_to_gate():
    calibration = calibrate_adjudicator(
        load_calibration_probe(tiers=(HARD_TIER,)), probe_aware(scripted)
    )
    assert calibration["accuracy"] == 1.0
    assert not calibration["calibrated"], "a perfect hard tier still establishes no floor"
    assert "no gating tier" in calibration["gate_failures"][0]
    assert calibration["tier_separation"] is None


def test_an_unparsable_hard_pair_does_not_cost_the_floor():
    """Only a gating tier's parse failures can suppress a precision figure."""
    labelled = probe_aware(scripted)
    hard = load_calibration_probe(tiers=(HARD_TIER,)).pairs[0]

    def broken_on_one_hard_pair(prompt: str) -> str:
        return "not json at all" if hard.left_text in prompt else labelled(prompt)

    calibration = calibrate_adjudicator(load_calibration_probe(), broken_on_one_hard_pair)
    assert calibration["unparsed_pairs"] == 1
    assert calibration["tiers"][HARD_TIER]["unparsed_pairs"] == 1
    assert calibration["tiers"][BASE_TIER]["unparsed_pairs"] == 0
    assert calibration["calibrated"]


# --- the standalone report and command --------------------------------------------------------


def test_the_report_shows_the_ladder_and_the_pairs_the_model_missed():
    calibration = calibrate_adjudicator(
        load_calibration_probe(), probe_aware(scripted, wrong_tiers=(HARD_TIER,))
    )
    report = calibration_report(
        {
            "model": "fake",
            "backend": "none",
            "temperature": 0.0,
            "seed": 0,
            "probe": str(PROBE_FILE),
            "seconds": 1.0,
            "calibration": calibration,
        }
    )
    assert "| base | 24 | 24 | 24 | 1.0 |" in report
    assert "reports only" in report and "cleared" in report
    assert "hard-days-against-years" in report, "a missed pair must be nameable"
    assert "calibrated: yes" in report


def test_the_calibration_command_writes_both_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "llb.conflicts.claim.adjudicator.build_adjudicator",
        lambda *args, **kwargs: probe_aware(scripted),
    )
    result = CliRunner().invoke(
        app,
        [
            "calibrate-conflict-adjudicator",
            "--conflict-model",
            "fake-model",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "calibration.json").read_text("utf-8"))
    assert payload["calibration"]["calibrated"]
    assert payload["requested_tiers"] == list(PROBE_TIERS)
    assert set(payload["calibration"]["tiers"]) == set(TIER_SIZES)
    assert "# Adjudicator calibration" in (tmp_path / "calibration.md").read_text("utf-8")
    assert "calibrated=yes" in result.output


def test_a_reasoning_model_that_never_emits_json_is_refused_and_renders_no_rates():
    """The measured gemma-4 failure: unparsable verdicts leave a half with nothing to score."""
    labelled = probe_aware(scripted)
    complementary = {
        pair.left_text
        for pair in load_calibration_probe(tiers=(BASE_TIER,)).pairs
        if not pair.actionable
    }

    def never_json(prompt: str) -> str:
        return "" if any(text in prompt for text in complementary) else labelled(prompt)

    calibration = calibrate_adjudicator(load_calibration_probe(), never_json)
    base = calibration["tiers"][BASE_TIER]
    assert base["specificity_on_complementary"] is None, "no negative pair parsed"
    assert not calibration["calibrated"]
    assert "unparsable verdict" in calibration["gate_failures"][0]
    report = calibration_report(
        {
            "model": "fake",
            "backend": "none",
            "temperature": 0.0,
            "seed": 0,
            "probe": str(PROBE_FILE),
            "seconds": 1.0,
            "calibration": calibration,
        }
    )
    assert "| 1.0 | -- | 0.6 | MISSED |" in report, "an unmeasurable rate must not print as None"
