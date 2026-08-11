"""Stratum placement, two-family reading, conditional prototype, and persistence."""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from llb.bench.agentic_design_fields import as_mapping
from llb.bench.agentic_memory_window_elision_reading import READING_COSTS, READING_FREE
from llb.bench.agentic_memory_window_elision_transfer import (
    analyze_transfer_runs,
    run_entry_aware_prototype,
    run_transfer_family,
)
from llb.bench.agentic_memory_window_elision_transfer_design import (
    load_window_elision_transfer_design,
    transfer_placements,
    validate_window_elision_transfer_design,
)
from llb.bench.agentic_memory_window_elision_transfer_reading import (
    PROTOTYPE_RECOVERS,
    TRANSFER_MIDDLE_COSTS,
)
from llb.bench.agentic_memory_window_elision_transfer_report import (
    format_window_elision_transfer_table,
    persist_window_elision_transfer,
)


class VisibleAnswerSummarizer:
    """Follow the workflow and retain an answer code only when the summary input exposes it."""

    def __call__(self, prompt: str) -> str:
        if "Стисло підсумуй" in prompt:
            code = re.search(r"ELI-[HMT]-\d{3}-\d{3}", prompt)
            return f"answer_code={code.group(0)}" if code else "no answer code was visible"
        if "[workflow complete]" not in prompt:
            tokens = re.findall(r'(?:токеном "|next token: )(el-[hmt]-\d{3}-\d{2})', prompt)
            assert tokens
            return json.dumps({"name": "advance", "arguments": {"token": tokens[-1]}})
        code = re.search(r"ELI-[HMT]-\d{3}-\d{3}", prompt)
        return json.dumps(
            {"name": "finish", "arguments": {"answer": code.group(0) if code else "LOST"}}
        )


def _candidate(family: str, model: str) -> dict[str, object]:
    return {"model_family": family, "model": model, "backend": "fake"}


def test_design_places_answer_facts_strictly_in_head_middle_and_tail():
    design = load_window_elision_transfer_design()
    validate_window_elision_transfer_design(design)
    placements = transfer_placements(design)
    assert {row["measured_stratum"] for row in placements} == {"head", "middle", "tail"}
    assert {row["folded_entries"] for row in placements} == {9}
    assert {row["offered_chars"] for row in placements} == {15776}
    for row in placements:
        if row["measured_stratum"] == "middle":
            assert row["head_end"] < row["fact_start"] < row["fact_end"] < row["tail_start"]


def test_design_rejects_a_fact_stage_that_leaves_its_declared_stratum():
    design = load_window_elision_transfer_design()
    moved = deepcopy(design)
    as_mapping(moved, "fact_stages")["middle"] = 3
    with pytest.raises(ValueError, match="does not occupy its declared elision stratum"):
        validate_window_elision_transfer_design(moved)


def test_two_family_middle_loss_gates_and_entry_aware_prototype_recovers(tmp_path: Path):
    design = load_window_elision_transfer_design()
    complete = VisibleAnswerSummarizer()
    runs = [
        run_transfer_family(design, _candidate("qwen", "fake-qwen"), complete=complete),
        run_transfer_family(design, _candidate("gemma", "fake-gemma"), complete=complete),
    ]
    first = analyze_transfer_runs(design, runs)
    assert first["transfer_reading"] == TRANSFER_MIDDLE_COSTS
    assert first["prototype_required"] is True
    for run in runs:
        strata = run.analysis["strata"]
        assert strata["head"]["reading"] == READING_FREE
        assert strata["middle"]["reading"] == READING_COSTS
        assert strata["tail"]["reading"] == READING_FREE
        run_entry_aware_prototype(design, run, complete=complete)

    final = analyze_transfer_runs(design, runs)
    assert final["prototype_reading"] == PROTOTYPE_RECOVERS
    detail = final["prototype_detail"]["families"]
    assert all(row["same_prompt_chars"] for row in detail)
    assert all(row["middle_recovers"] for row in detail)

    table = format_window_elision_transfer_table(final)
    paths = persist_window_elision_transfer(
        design,
        runs,
        final,
        data_dir=tmp_path,
        table=table,
        mirror=lambda *_: None,
    )
    assert "entry_aware_fold_recovers_middle_completion" in table
    assert Path(paths["manifest"]).exists()
    method = tmp_path / "agentic-compact-window-elision-transfer"
    assert len(list(method.glob("*/manifest.json"))) == 7
