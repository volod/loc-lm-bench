"""Two-family repeated-fold replication: design contract, paired uncertainty, and readings."""

import json
import re
from pathlib import Path

from llb.bench.memory.repeated_fold.replication import (
    ReplicationFamilyRun,
    analyze_replication_runs,
    family_fold_analysis,
    run_replication_family,
)
from llb.bench.memory.repeated_fold.replication_design import (
    load_repeated_fold_replication_design,
    replication_roster,
    validate_replication_design,
)
from llb.bench.memory.repeated_fold.replication_reading import (
    REPLICATION_EXTENDS,
    REPLICATION_FAILS,
    REPLICATION_INELIGIBLE,
    fold_group_rows,
    powered_fold_limit,
    replication_reading,
)
from llb.bench.memory.repeated_fold.replication_report import (
    format_replication_table,
    persist_replication_run,
)


class WalksTheChain:
    """Play the workflow perfectly; a named case loses its code once the transcript folds twice."""

    def __init__(self, loses_above_folds: int | None = None, lossy_item: str = ""):
        self.loses_above_folds = loses_above_folds
        self.lossy_item = lossy_item

    def __call__(self, prompt: str) -> str:
        if "Стисло підсумуй" in prompt:
            # The running summary carries its own fold ordinal, so a later fold still knows how
            # many came before it even when it REPLACES the earlier summary entry.
            return f"fold={self._folds(prompt) + 1} retained"
        if "[workflow complete]" not in prompt:
            tokens = re.findall(r'(?:токеном "|next token: )(wf-\d{3}-\d+)', prompt)
            assert tokens
            return json.dumps({"name": "advance", "arguments": {"token": tokens[-1]}})
        code = re.search(r"MEM-\d{3}-\d{3}", prompt)
        answer = code.group(0) if code else "LOST"
        if (
            self.loses_above_folds is not None
            and self.lossy_item in prompt
            and self._folds(prompt) > self.loses_above_folds
        ):
            answer = "LOST"
        return json.dumps({"name": "finish", "arguments": {"answer": answer}})

    @staticmethod
    def _folds(prompt: str) -> int:
        seen = [int(value) for value in re.findall(r"fold=(\d+) retained", prompt)]
        return max(seen, default=0)


def _fake_design(tmp_tasks: int = 8) -> dict[str, object]:
    design = load_repeated_fold_replication_design()
    held = dict(design["held_fixed"])
    held["n_tasks"] = tmp_tasks
    held["minimum_paired_cases_per_fold"] = 2
    return {**design, "held_fixed": held}


def test_committed_replication_design_declares_a_roster_and_a_larger_case_set():
    design = load_repeated_fold_replication_design()
    validate_replication_design(design)
    roster = replication_roster(design)
    assert len(roster) >= design["required_qualified_families"] == 2
    assert len({row["model_family"] for row in roster}) == len(roster)
    completion = json.loads(
        Path("samples/benchmarks/agentic_compact_repeated_fold_completion_design.json").read_text()
    )
    held = design["held_fixed"]
    assert held["n_tasks"] > completion["held_fixed"]["n_tasks"]
    assert design["seed"] == completion["seed"]
    assert [cell["cell_id"] for cell in design["cells"]] == [
        cell["cell_id"] for cell in completion["cells"]
    ]


def test_replication_refuses_a_case_set_no_larger_than_the_study_it_replicates():
    design = load_repeated_fold_replication_design()
    held = {**design["held_fixed"], "n_tasks": 2}
    try:
        validate_replication_design({**design, "held_fixed": held})
    except ValueError as exc:
        assert "predeclared" in str(exc)
    else:
        raise AssertionError("a two-case replication must be refused")


def test_replication_refuses_a_roster_that_repeats_one_family():
    design = load_repeated_fold_replication_design()
    roster = list(design["candidate_roster"])
    duplicated = [roster[0], {**roster[1], "model_family": roster[0]["model_family"]}]
    try:
        validate_replication_design({**design, "candidate_roster": duplicated})
    except ValueError as exc:
        assert "distinct model family" in str(exc)
    else:
        raise AssertionError("a single-family roster must be refused")


def test_two_families_extend_the_fold_rule_and_persist_one_aggregate(tmp_path: Path):
    design = _fake_design()
    runs = [
        run_replication_family(
            design,
            candidate,
            complete=WalksTheChain(),
        )
        for candidate in replication_roster(design)[:2]
    ]
    analysis = analyze_replication_runs(design, runs)
    assert analysis["replication_reading"] == REPLICATION_EXTENDS
    assert analysis["shared_powered_fold_limit"] == 3
    assert analysis["task_set_digest"] is not None
    assert len(analysis["task_set_digests"]) == 1
    assert analysis["family_digest"] and analysis["roster_digest"]

    table = format_replication_table(analysis)
    assert "powered fold limit" in table
    paths = persist_replication_run(
        design,
        runs,
        analysis,
        data_dir=tmp_path,
        table=table,
        mirror=lambda *_: None,
    )
    assert Path(paths["manifest"]).exists()
    bundles = list((tmp_path / "agentic-compact-vs-cap").glob("*/manifest.json"))
    assert len(bundles) == 2 * 6 + 1


def test_a_family_that_loses_a_paired_case_names_the_family_and_the_fold():
    design = _fake_design()
    candidates = replication_roster(design)[:2]
    healthy = run_replication_family(design, candidates[0], complete=WalksTheChain())
    lossy = run_replication_family(
        design,
        candidates[1],
        complete=WalksTheChain(loses_above_folds=1, lossy_item="Case 003"),
    )
    analysis = analyze_replication_runs(design, [healthy, lossy])
    assert analysis["replication_reading"] == REPLICATION_FAILS
    assert candidates[1]["model_family"] in analysis["replication_reason"]
    assert lossy.analysis["powered_fold_limit"] == 1


def test_one_qualified_family_states_no_cross_family_rule():
    design = _fake_design()
    candidate = replication_roster(design)[0]
    run = run_replication_family(design, candidate, complete=WalksTheChain())
    ineligible = ReplicationFamilyRun(
        model_family="stalled",
        model="stalled-model",
        backend="fake",
        base=run.base,
        analysis={
            **family_fold_analysis(design, run.base.analysis, candidate),
            "model_family": "stalled",
            "model": "stalled-model",
            "control_eligible": False,
        },
    )
    analysis = analyze_replication_runs(design, [run, ineligible])
    assert analysis["replication_reading"] == REPLICATION_INELIGIBLE
    assert "1 of 2" in analysis["replication_reason"]


def test_an_underfloor_fold_group_is_reported_and_never_cuts_the_verdict():
    cells = [
        _cell("onefold", True, "typed_marker", [("a", 1, True), ("b", 1, True)]),
        _cell("twofold", False, "typed_marker", [("a", 2, True), ("b", 2, True), ("c", 4, False)]),
        _cell("onefold", True, "model_summary_only", [("a", 1, True), ("b", 1, True)]),
        _cell("twofold", False, "model_summary_only", [("a", 2, True), ("b", 2, True)]),
    ]
    rows = fold_group_rows(cells, evidence_floor=2)
    by_fold = {row["measured_folds"]: row for row in rows}
    assert by_fold[4]["meets_evidence_floor"] is False
    assert by_fold[2]["meets_evidence_floor"] is True
    limit, reason = powered_fold_limit(rows)
    assert limit == 2
    assert "[4]" in reason


def test_an_unmeasured_fold_count_is_a_named_gap_not_a_loss():
    cells = [
        _cell("onefold", True, "typed_marker", [(f"t{i}", 1, True) for i in range(4)]),
        _cell("threefold", False, "typed_marker", [(f"t{i}", 3, True) for i in range(4)]),
    ]
    rows = fold_group_rows(cells, evidence_floor=2)
    limit, reason = powered_fold_limit(rows)
    assert limit == 3
    assert "no case measured [2] folds" in reason


def test_a_loss_inside_an_underfloor_group_is_named_without_cutting_the_limit():
    cells = [
        _cell("onefold", True, "typed_marker", [(f"t{i}", 1, True) for i in range(4)]),
        _cell(
            "twofold",
            False,
            "typed_marker",
            [("t0", 2, True), ("t1", 2, True), ("t2", 3, False)],
        ),
    ]
    rows = fold_group_rows(cells, evidence_floor=2)
    limit, reason = powered_fold_limit(rows)
    assert limit == 2
    assert "under-floor groups that DID lose a paired case: [3]" in reason


def test_wilson_interval_widens_a_small_fold_group():
    cells = [
        _cell("onefold", True, "typed_marker", [(f"t{i}", 1, True) for i in range(8)]),
        _cell("twofold", False, "typed_marker", [("t0", 3, True), ("t1", 3, True)]),
    ]
    rows = fold_group_rows(cells, evidence_floor=2)
    by_fold = {row["measured_folds"]: row for row in rows}
    assert by_fold[3]["completion"] == 1.0
    assert by_fold[3]["completion_lo"] < by_fold[1]["completion_lo"]
    assert by_fold[3]["paired"]["n_pairs"] == 2
    assert by_fold[3]["paired"]["control_wins"] == 0


def test_required_families_are_counted_before_any_rule_is_stated():
    reading, reason, qualified = replication_reading(
        [{"control_eligible": False, "model_family": "a", "model": "a"}], required_families=2
    )
    assert reading == REPLICATION_INELIGIBLE
    assert qualified == []
    assert "no cross-family fold-count rule" in reason


def _cell(
    cell_id: str, control: bool, arm: str, cases: list[tuple[str, int, bool]]
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "arm": arm,
        "cap_fitting_control": control,
        "cases": [
            {"item_id": item, "measured_folds": folds, "success": success, "status": "completed"}
            for item, folds, success in cases
        ],
    }
