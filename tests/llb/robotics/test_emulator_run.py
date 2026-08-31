import json

from llb.core.paths import PROJECT_ROOT
from llb.robotics.emulator_fixture import load_emulator_fixture
from llb.robotics.emulator_run import evaluate_emulator_fixture, run_emulator_check

FIXTURE_PATH = PROJECT_ROOT / "samples" / "robotics" / "emulator" / "scenarios.json"


def test_full_committed_scenario_matrix_blocks_every_forbidden_invocation() -> None:
    report = evaluate_emulator_fixture(load_emulator_fixture(FIXTURE_PATH))
    assert report["verdict"] == "pass"
    assert report["scenario_count"] == 13
    assert report["process_step_count"] == 15
    assert report["forbidden_adapter_invocations"] == 0
    assert report["reason_counts"]["non_idempotent_retry_forbidden"] == 2
    assert report["reason_counts"]["policy_checks_passed"] == 3


def test_runner_writes_json_and_markdown_under_the_data_root(tmp_path) -> None:
    output_dir, report = run_emulator_check(FIXTURE_PATH, tmp_path)
    saved = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    assert saved["run_id"] == report["run_id"]
    assert saved["forbidden_adapter_invocations"] == 0
    assert "- Scenarios: 13" in markdown
    assert "`emergency_stop_active`" in markdown
