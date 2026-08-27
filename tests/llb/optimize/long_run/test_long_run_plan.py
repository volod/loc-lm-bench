"""The predeclaration: the screen size is derived from paired power, never chosen."""

from pathlib import Path

import pytest

from llb.optimize.joint_search.long_run.plan import (
    BINDING_AVAILABLE,
    BINDING_POWER,
    declare_plan,
    screen_sizing,
)
from llb.optimize.joint_search.long_run.reference import paired_reference_deltas
from llb.rag.fusion_evidence.power import required_sample_size

# A noisy paired reference: the same items, half won by the candidate and half by the baseline.
NOISY = [0.6, -0.5, 0.4, -0.3, 0.7, -0.6, 0.2, -0.4, 0.5, -0.2]
QUIET = [0.05, 0.04, 0.06, 0.05, 0.04, 0.06, 0.05, 0.05, 0.04, 0.06]


def _selector() -> dict[str, str]:
    return {"lane": "joint-search-long-run", "candidate": "a", "baseline": "b", "metric": "q"}


def test_screen_size_scales_with_the_reference_variance():
    """A noisier reference asks for MORE screen items at the same declared gain."""
    quiet = screen_sizing(
        QUIET, minimum_detectable_gain=0.05, target_power=0.8, confidence=0.95, available_n=10_000
    )
    noisy = screen_sizing(
        NOISY, minimum_detectable_gain=0.05, target_power=0.8, confidence=0.95, available_n=10_000
    )
    assert noisy.required_n > quiet.required_n
    assert noisy.binding == BINDING_POWER
    assert noisy.satisfied


def test_screen_size_matches_the_shared_power_contract():
    """The derived count is the shared paired-power arithmetic, not a second implementation."""
    sizing = screen_sizing(
        NOISY, minimum_detectable_gain=0.10, target_power=0.8, confidence=0.95, available_n=10_000
    )
    expected = required_sample_size(_sd(NOISY), 0.10, alpha=round(1.0 - 0.95, 12), target_power=0.8)
    assert sizing.required_n >= expected  # the discordance floor may bind above the variance one


def test_a_split_too_small_reports_the_shortfall_instead_of_meeting_the_target():
    """The screen cannot exceed the tuning split, and a run that fell short says so."""
    sizing = screen_sizing(
        NOISY, minimum_detectable_gain=0.01, target_power=0.8, confidence=0.95, available_n=12
    )
    assert sizing.applied_n == 12
    assert sizing.required_n > 12
    assert sizing.binding == BINDING_AVAILABLE
    assert not sizing.satisfied


def test_declared_plan_persists_the_stopping_rule_and_the_power_block(tmp_path: Path):
    plan = declare_plan(
        tmp_path / "reference.json",
        NOISY,
        minimum_detectable_gain=0.05,
        available_n=82,
        trial_budget=20,
        trial_block=5,
        stability_blocks=2,
        stability_agreement=1.0,
        selector=_selector(),
    )
    payload = plan.to_dict()
    assert payload["minimum_detectable_gain"] == 0.05
    assert payload["screen"]["applied_n"] == plan.screen.applied_n
    assert payload["power"]["planned_n"] == plan.screen.applied_n
    assert "2 consecutive block transitions" in payload["stopping_rule"]
    assert "20 trials per finalist are spent" in payload["stopping_rule"]


@pytest.mark.parametrize(
    ("budget", "block", "blocks", "agreement"),
    [(4, 5, 2, 1.0), (20, 5, 0, 1.0), (20, 5, 2, 1.5)],
)
def test_an_incoherent_declaration_is_refused(
    tmp_path: Path, budget: int, block: int, blocks: int, agreement: float
):
    """A budget under one block, a zero-block rule, and an out-of-range agreement all raise."""
    with pytest.raises(ValueError):
        declare_plan(
            tmp_path / "reference.json",
            NOISY,
            available_n=82,
            trial_budget=budget,
            trial_block=block,
            stability_blocks=blocks,
            stability_agreement=agreement,
            selector=_selector(),
        )


def test_reference_deltas_come_from_the_items_both_bundles_scored(tmp_path: Path):
    candidate = tmp_path / "cand"
    baseline = tmp_path / "base"
    _write_bundle(candidate, {"a": 1.0, "b": 0.5, "c": 0.25})
    _write_bundle(baseline, {"a": 0.5, "b": 0.5, "d": 0.9})
    assert paired_reference_deltas(candidate, baseline) == [0.5, 0.0]


def test_a_reference_pair_with_no_shared_items_is_refused(tmp_path: Path):
    candidate = tmp_path / "cand"
    baseline = tmp_path / "base"
    _write_bundle(candidate, {"a": 1.0, "b": 0.5})
    _write_bundle(baseline, {"c": 0.5, "d": 0.5})
    with pytest.raises(ValueError, match="at least two items"):
        paired_reference_deltas(candidate, baseline)


def _write_bundle(run_dir: Path, scores: dict[str, float]) -> None:
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scores.jsonl").write_text(
        "\n".join(
            json.dumps({"item_id": item, "objective_score": value})
            for item, value in scores.items()
        )
        + "\n",
        encoding="utf-8",
    )


def _sd(values: list[float]) -> float:
    from llb.rag.fusion_evidence.power import sample_sd

    return sample_sd(values)
