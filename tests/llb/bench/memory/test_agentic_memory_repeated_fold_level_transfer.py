"""What the per-case fold length the fit carries across is worth, and the count it implies.

The guard fit replays one measured fold length per CASE. That is only a per-case prediction if
verbosity is a property of the case, and the replication's own rows say it is not, so the count is
a family rate with a width rather than twelve predictions. This file covers the three things that
follow: the correlation that decides the level rule, the interval the count is stated as, and the
refusal when a guard's flip window sits inside the family's own case-to-case spread.
"""

import json
import re

from llb.bench.memory.repeated_fold.design import completion_cells
from llb.bench.memory.repeated_fold.fold_span import fold_length_span_model
from llb.bench.memory.repeated_fold.guard_fit import fit_fold_guard, guard_fit_spec
from llb.bench.memory.repeated_fold.level_transfer import (
    COUNT_RANK_ONLY,
    COUNT_READABLE,
    COUNT_UNMEASURED,
    LEVEL_CONSTANT_PER_FAMILY,
    LEVEL_PER_CASE,
    LEVEL_TRANSFER_ABSENT,
    LEVEL_TRANSFER_MIN_PAIRS,
    LEVEL_TRANSFER_PRESENT,
    LEVEL_TRANSFER_UNMEASURED,
    LEVEL_TRANSFER_UNREAD,
    count_interval,
    count_reading,
    family_level_transfer_reading,
    fold_length_spread_chars,
    level_correlation,
    level_transfer_reading,
    paired_case_levels,
)
from llb.bench.memory.repeated_fold.replication import (
    analyze_replication_runs,
    run_replication_family,
)
from llb.bench.memory.repeated_fold.replication_design import (
    load_repeated_fold_replication_design,
    replication_roster,
)
from llb.bench.memory.repeated_fold.replication_report import format_replication_table

CONTROL_CELL = "onefold-d10-g14000"
FITTED_CELL = "twofold-d10-g7000"

# The two qualified families of the committed replication, as the run measured them: each family's
# twelve control fold lengths, the (offered span, written length) point its control folded at, and
# the second point the never-fitted 6500 cell gave the replay. CUDA host run
# `agent-context-policy-repeated-fold-completion-replication`, 2026-08-29. The shared-guard run
# before it measured 11 of `qwen3:14b`'s 12 cases and 1 of `gemma4:e4b`'s on the two-fold rung at
# the DECLARED 7000 guard, which is the count the declared interval below has to contain.
MEASURED_FAMILIES = {
    "qwen3:14b": {
        "fold_lengths": [278, 274, 268, 266, 268, 263, 266, 274, 266, 304, 302, 274],
        "anchor": (11802, 274),
        "second": (4185, 212),
        "shared_guard_two_fold_cases": 11,
    },
    "gemma4:e4b": {
        "fold_lengths": [255, 255, 255, 240, 255, 318, 314, 241, 237, 369, 320, 260],
        "anchor": (11802, 255),
        "second": (4253, 284),
        "shared_guard_two_fold_cases": 1,
    },
}

# The same run's paired levels: each case's control fold length against its OWN first fold at the
# fitted cell. This is the measurement the level rule is decided from.
PAIRED_LEVELS = {
    "qwen3:14b": list(
        zip(
            [278, 274, 268, 266, 268, 263, 266, 274, 266, 304, 302, 274],
            [253, 178, 273, 273, 194, 273, 178, 178, 178, 178, 273, 273],
            strict=True,
        )
    ),
    "gemma4:e4b": list(
        zip(
            [255, 255, 255, 240, 255, 318, 314, 241, 237, 369, 320, 260],
            [285, 294, 265, 282, 282, 262, 272, 246, 281, 288, 246, 246],
            strict=True,
        )
    ),
}


def _fitted_cell(design: dict[str, object]) -> dict[str, object]:
    cell_id = guard_fit_spec(design)["cell_id"]
    return next(cell for cell in completion_cells(design) if cell["cell_id"] == cell_id)


def _measured_fit(family: str) -> dict[str, object]:
    """The fit the committed run made, replayed from its persisted measurements alone."""
    design = load_repeated_fold_replication_design()
    measured = MEASURED_FAMILIES[family]
    return fit_fold_guard(
        design,
        _fitted_cell(design),
        design["held_fixed"],
        list(measured["fold_lengths"]),
        evidence_floor=4,
        step_entry_chars=[36] * 120,
        span_model=fold_length_span_model([measured["anchor"]], [measured["second"]]),
    )


def _cells(
    control: list[tuple[str, int]], fitted: list[tuple[str, int]]
) -> list[dict[str, object]]:
    """Persisted cell rows carrying one first-fold length per case, in both cells."""
    return [
        {
            "cell_id": cell_id,
            "arm": "typed_marker",
            "cases": [{"item_id": item, "summary_output_chars": [chars]} for item, chars in rows],
        }
        for cell_id, rows in ((CONTROL_CELL, control), (FITTED_CELL, fitted))
    ]


def test_a_count_interval_is_the_rate_its_lengths_estimate_rounded_outward():
    """The count is n times a rate measured on n lengths, so its width is that rate's width."""
    assert count_interval(10, 12) == [6, 12]
    assert count_interval(0, 12) == [0, 3]
    assert count_interval(12, 12) == [9, 12]
    for successes in range(13):
        low, high = count_interval(successes, 12)
        assert low <= successes <= high


def test_an_unmeasured_control_carries_no_count_and_no_interval():
    assert count_interval(0, 0) == []
    design = load_repeated_fold_replication_design()
    record = fit_fold_guard(
        design, _fitted_cell(design), design["held_fixed"], [], evidence_floor=4
    )
    assert record["predicted_target_cases_interval"] == []
    assert record["declared_target_cases_interval"] == []
    assert record["count_reading"] == COUNT_UNMEASURED
    assert record["declared_count_reading"] == COUNT_UNMEASURED


def test_a_family_that_writes_one_length_has_no_spread_to_width_a_count_with():
    """A count with no measured spread behind it is not widened; it is named unmeasured."""
    design = load_repeated_fold_replication_design()
    record = fit_fold_guard(
        design,
        _fitted_cell(design),
        design["held_fixed"],
        [274] * 12,
        evidence_floor=4,
        step_entry_chars=[36] * 120,
    )
    assert fold_length_spread_chars([274] * 12) == 0
    assert record["fold_length_spread_chars"] == 0
    assert record["count_reading"] == COUNT_UNMEASURED
    assert record["declared_count_reading"] == COUNT_UNMEASURED


def test_a_guard_whose_flip_window_sits_inside_the_spread_is_refused_as_a_count():
    """The honest boundary: a guard the fit can RANK but cannot hand an operator as a number."""
    assert count_reading(200, 41) == COUNT_READABLE
    assert count_reading(10, 41) == COUNT_RANK_ONLY
    assert count_reading(10, 0) == COUNT_UNMEASURED
    for family in MEASURED_FAMILIES:
        record = _measured_fit(family)
        spread = record["fold_length_spread_chars"]
        assert (
            record["declared_fold_count_margin_chars"] < spread <= record["fold_count_margin_chars"]
        )
        assert record["declared_count_reading"] == COUNT_RANK_ONLY
        assert record["count_reading"] == COUNT_READABLE


def test_the_declared_guards_interval_covers_the_count_the_shared_guard_run_measured():
    """The gate this work exists for: a point estimate missed both families, the interval does not."""
    for family, measured in MEASURED_FAMILIES.items():
        record = _measured_fit(family)
        low, high = record["declared_target_cases_interval"]
        assert low <= measured["shared_guard_two_fold_cases"] <= high
        assert record["declared_target_cases"] != measured["shared_guard_two_fold_cases"]


def test_widening_the_count_does_not_move_the_guard_the_fit_picks():
    """The interval changes what a count is worth to read, never which rung the ladder runs."""
    for family in MEASURED_FAMILIES:
        record = _measured_fit(family)
        assert record["fitted_max_prompt_chars"] == 7900
        assert record["predicted_target_cases"] == 12
        low, high = record["predicted_target_cases_interval"]
        assert low <= record["predicted_target_cases"] <= high


def test_the_measured_run_says_the_case_level_does_not_transfer_between_cells():
    """The run's own answer to whether a third calibration point would help: it would not."""
    for family, pairs in PAIRED_LEVELS.items():
        correlation = level_correlation(pairs)
        assert abs(correlation) < 0.1, family
        assert level_transfer_reading(correlation, len(pairs)) == LEVEL_TRANSFER_ABSENT


def test_a_case_level_that_repeats_in_the_next_cell_is_named_as_transferring():
    """The other branch must be reachable, or the reading is a constant dressed as a measurement."""
    cells = _cells(
        [(f"case-{index}", 200 + index * 10) for index in range(6)],
        [(f"case-{index}", 150 + index * 8) for index in range(6)],
    )
    pairs = paired_case_levels(cells, CONTROL_CELL, FITTED_CELL)
    assert len(pairs) == 6
    assert level_correlation(pairs) > 0.99
    assert level_transfer_reading(level_correlation(pairs), len(pairs)) == LEVEL_TRANSFER_PRESENT


def test_too_few_paired_cases_are_refused_a_correlation_rather_than_given_one():
    """Two points define a line, so a correlation over two pairs is +/-1 by construction."""
    pairs = [(200, 150), (260, 190)]
    assert len(pairs) < LEVEL_TRANSFER_MIN_PAIRS
    assert level_correlation(pairs) == 0.0
    assert level_transfer_reading(1.0, len(pairs)) == LEVEL_TRANSFER_UNMEASURED


def test_only_the_shipped_arm_pairs_a_level():
    """The ablation runs a different summarizer, so its lengths belong to a policy nobody ships."""
    cells = _cells(
        [("case-0", 200), ("case-1", 260)],
        [("case-0", 150), ("case-1", 190)],
    )
    cells.append(
        {
            "cell_id": FITTED_CELL,
            "arm": "model_summary_only",
            "cases": [{"item_id": "case-0", "summary_output_chars": [9999]}],
        }
    )
    assert paired_case_levels(cells, CONTROL_CELL, FITTED_CELL) == [(200, 150), (260, 190)]


def test_a_case_that_ran_only_one_cell_is_not_paired():
    cells = _cells([("case-0", 200), ("case-1", 260)], [("case-0", 150)])
    assert paired_case_levels(cells, CONTROL_CELL, FITTED_CELL) == [(200, 150)]


def test_the_cross_family_reading_names_every_correlation_it_stands_on():
    def family(name: str, reading: str, correlation: float) -> dict[str, object]:
        return {
            "model_family": name,
            "control_eligible": True,
            "guard_fits": [
                {
                    "level_transfer_reading": reading,
                    "level_transfer_correlation": correlation,
                    "level_transfer_pairs": 12,
                }
            ],
        }

    constant, reason = family_level_transfer_reading(
        [
            family("qwen", LEVEL_TRANSFER_ABSENT, -0.031),
            family("gemma4", LEVEL_TRANSFER_ABSENT, -0.031),
        ]
    )
    assert constant == LEVEL_CONSTANT_PER_FAMILY
    assert "qwen r=-0.03" in reason and "gemma4 r=-0.03" in reason
    carried, reason = family_level_transfer_reading(
        [
            family("qwen", LEVEL_TRANSFER_ABSENT, -0.031),
            family("gemma4", LEVEL_TRANSFER_PRESENT, 0.910),
        ]
    )
    assert carried == LEVEL_PER_CASE
    assert "gemma4" in reason
    unread, reason = family_level_transfer_reading([family("qwen", LEVEL_TRANSFER_UNMEASURED, 0.0)])
    assert unread == LEVEL_TRANSFER_UNREAD
    assert "paired" in reason


class WritesPerCaseLengths:
    """Perfect play, with a summarizer whose verbosity IS a property of the case it is running."""

    def __init__(self, summary_chars: int):
        self.summary_chars = summary_chars

    def __call__(self, prompt: str) -> str:
        if "Стисло підсумуй" in prompt:
            case = re.search(r"wf-(\d{3})", prompt)
            index = int(case.group(1)) if case else 0
            return ("підсумок попередніх кроків; " * 400)[: self.summary_chars + index * 12]
        if "[workflow complete]" not in prompt:
            tokens = re.findall(r'(?:токеном "|next token: )(wf-\d{3}-\d+)', prompt)
            assert tokens
            return json.dumps({"name": "advance", "arguments": {"token": tokens[-1]}})
        code = re.search(r"MEM-\d{3}-\d{3}", prompt)
        return json.dumps(
            {"name": "finish", "arguments": {"answer": code.group(0) if code else "LOST"}}
        )


def test_a_run_reports_its_level_rule_the_count_interval_and_the_measured_count_inside_it():
    design = load_repeated_fold_replication_design()
    held = {**design["held_fixed"], "n_tasks": 6, "minimum_paired_cases_per_fold": 2}
    design = {**design, "held_fixed": held}
    roster = replication_roster(design)[:2]
    runs = [
        run_replication_family(design, roster[0], complete=WritesPerCaseLengths(120)),
        run_replication_family(design, roster[1], complete=WritesPerCaseLengths(400)),
    ]
    analysis = analyze_replication_runs(design, runs)
    assert analysis["level_transfer_reading"] == LEVEL_PER_CASE
    for family in analysis["families"]:
        fit = family["guard_fits"][0]
        assert fit["level_transfer_pairs"] == 6
        assert fit["level_transfer_reading"] == LEVEL_TRANSFER_PRESENT
        assert fit["fold_length_spread_chars"] > 0
        low, high = fit["predicted_target_cases_interval"]
        assert low <= fit["predicted_target_cases"] <= high
        assert fit["measured_within_predicted_interval"] is (
            low <= fit["measured_target_cases"] <= high
        )
    table = format_replication_table(analysis)
    assert "count interval:" in table
    assert "level transfer:" in table
