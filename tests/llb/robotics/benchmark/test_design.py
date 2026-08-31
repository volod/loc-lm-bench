import json

import pytest

from llb.core.paths import PROJECT_ROOT
from llb.robotics.benchmark.design import load_design
from llb.robotics.benchmark.run import validate_benchmark_design

DESIGN = PROJECT_ROOT / "samples" / "robotics" / "benchmark" / "design.json"


def test_frozen_design_predeclares_evidence_and_fault_coverage():
    report = validate_benchmark_design(DESIGN)

    assert report["task_count"] == 16
    assert report["minimum_detectable_gain"] == 0.125
    assert report["minimum_evidence_count"] == 16
    assert len(report["fault_classes"]) == 8


def test_changed_task_ledger_is_refused(tmp_path):
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    design["task_ledger"] = "tasks.jsonl"
    (tmp_path / "design.json").write_text(json.dumps(design), encoding="utf-8")
    (tmp_path / "tasks.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        load_design(tmp_path / "design.json")
