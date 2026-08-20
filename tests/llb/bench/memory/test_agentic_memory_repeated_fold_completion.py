"""Repeated-fold completion design, reading, marker ablation, and persistence."""

import json
import re
from pathlib import Path

from llb.bench.memory.repeated_fold.completion import run_repeated_fold_completion
from llb.bench.memory.repeated_fold.reading import (
    MECHANISM_MARKER,
    READING_DECAYS,
    READING_INSUFFICIENT,
    READING_STABLE,
    completion_cost_reading,
)
from llb.bench.memory.repeated_fold.design import (
    completion_cells,
    load_repeated_fold_design,
    probe_completion_cell,
    validate_repeated_fold_design,
)
from llb.bench.memory.repeated_fold.report import (
    format_repeated_fold_table,
    persist_repeated_fold_run,
)


class SummaryDropsMemory:
    """Walk the workflow while every model-written summary omits the early code."""

    def __call__(self, prompt: str) -> str:
        if "Стисло підсумуй" in prompt:
            return "checkpoint retained"
        if "[workflow complete]" not in prompt:
            tokens = re.findall(r'(?:токеном "|next token: )(wf-\d{3}-\d+)', prompt)
            assert tokens
            return json.dumps({"name": "advance", "arguments": {"token": tokens[-1]}})
        code = re.search(r"MEM-\d{3}-\d{3}", prompt)
        return json.dumps(
            {"name": "finish", "arguments": {"answer": code.group(0) if code else "LOST"}}
        )


def test_design_predeclares_one_two_and_three_fold_cells():
    design = load_repeated_fold_design()
    validate_repeated_fold_design(design)
    held = design["held_fixed"]
    probes = [probe_completion_cell(cell, held) for cell in completion_cells(design)]
    assert [probe["oracle_folds"] for probe in probes] == [1, 2, 3]
    assert [probe["cap_fitting"] for probe in probes] == [True, False, False]


def test_compact_only_run_groups_measured_folds_and_attributes_the_marker(tmp_path: Path):
    design = load_repeated_fold_design()
    run = run_repeated_fold_completion(
        design,
        model="fake",
        backend="fake",
        complete=SummaryDropsMemory(),
    )
    analysis = run.analysis
    assert analysis["completion_reading"] == READING_STABLE
    assert analysis["recommended_fold_count_limit"] == 3
    assert analysis["mechanism_reading"] == MECHANISM_MARKER
    assert [row["measured_folds"] for row in analysis["completion_by_measured_fold_count"]] == [
        1,
        2,
        3,
    ]
    digests = {row["task_set_digest"] for row in analysis["cells"]}
    assert len(digests) == 1

    table = format_repeated_fold_table(analysis)
    paths = persist_repeated_fold_run(
        design,
        run,
        data_dir=tmp_path,
        table=table,
        tokens_per_s=12.5,
        mirror=lambda *_: None,
    )
    assert "completion by measured fold count" in table
    assert Path(paths["manifest"]).exists()
    assert len(list((tmp_path / "agentic-compact-vs-cap").glob("*/manifest.json"))) == 7


def test_completion_reading_recommends_the_last_nondecayed_fold():
    reading, reason, limit = completion_cost_reading(
        [
            {"measured_folds": 1, "completion": 1.0},
            {"measured_folds": 2, "completion": 1.0},
            {"measured_folds": 3, "completion": 0.5},
        ]
    )
    assert reading == READING_DECAYS
    assert "3 measured folds" in reason
    assert limit == 2


def test_zero_one_fold_completion_cannot_claim_stability():
    reading, reason, limit = completion_cost_reading(
        [
            {"measured_folds": 1, "completion": 0.0},
            {"measured_folds": 2, "completion": 0.5},
        ]
    )
    assert reading == READING_INSUFFICIENT
    assert "completed nothing" in reason
    assert limit is None
