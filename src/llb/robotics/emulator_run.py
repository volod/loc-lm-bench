"""Replay the committed emulator matrix and persist its deterministic report."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from llb.bench.common import new_run_timestamp
from llb.core.fsutil import atomic_write_text
from llb.core.paths import resolve_data_dir
from llb.robotics.action_executor import ActionExecutor
from llb.robotics.device_emulator import DeviceEmulator
from llb.robotics.emulator_fixture import load_emulator_fixture
from llb.robotics.emulator_models import (
    EmulatorFixture,
    EmulatorScenario,
    OperatorApproval,
    ScenarioStep,
)

LOGGER = logging.getLogger(__name__)
METHOD_NAME = "robotics-emulator"
MANDATORY_REASONS = frozenset(
    {
        "operation_not_allowed",
        "state_revision_stale",
        "driver_contract_invalid",
        "approval_required",
        "tainted_evidence",
        "emergency_stop_active",
        "device_lock_conflict",
        "non_idempotent_retry_forbidden",
    }
)


def _approvals(fixture: EmulatorFixture) -> dict[str, OperatorApproval]:
    return {approval.approval_id: approval for approval in fixture.approvals}


def _configure(emulator: DeviceEmulator, scenario: EmulatorScenario) -> None:
    for device_id in scenario.setup.emergency_stop_devices:
        emulator.set_emergency_stop(device_id)
    for device_id in scenario.setup.externally_locked_devices:
        emulator.set_external_lock(device_id)
    for fault in scenario.setup.faults:
        emulator.inject_fault(fault)


def _process_step(
    scenario_id: str,
    step: ScenarioStep,
    executor: ActionExecutor,
    emulator: DeviceEmulator,
    approvals: dict[str, OperatorApproval],
) -> tuple[dict[str, Any], str, int]:
    if step.proposal is None or step.expected is None:
        raise ValueError("process step lacks a proposal or expectation")
    before = emulator.invocation_count
    execution = executor.process(step.proposal, approvals.get(step.approval_id or ""))
    reason = execution.decision.reasons[0]
    observed = (
        execution.decision.decision,
        execution.receipt.status,
        reason,
        emulator.invocation_count,
    )
    wanted = (
        step.expected.decision,
        step.expected.receipt_status,
        step.expected.reason,
        step.expected.adapter_invocations,
    )
    if observed != wanted:
        raise ValueError(f"scenario {scenario_id} mismatch: expected {wanted}, got {observed}")
    invocation_delta = emulator.invocation_count - before
    forbidden = invocation_delta if step.expected.decision != "approve" else 0
    row = {
        "action": "process",
        "decision": execution.decision.model_dump(mode="json"),
        "receipt": execution.receipt.model_dump(mode="json"),
        "adapter_invocations": emulator.invocation_count,
    }
    return row, reason, forbidden


def _evaluate_scenario(
    fixture: EmulatorFixture,
    scenario: EmulatorScenario,
    approvals: dict[str, OperatorApproval],
) -> tuple[dict[str, Any], list[str], int, int]:
    emulator = DeviceEmulator(fixture.devices)
    executor = ActionExecutor(emulator, fixture.policy)
    _configure(emulator, scenario)
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    forbidden = 0
    processes = 0
    for step in scenario.steps:
        if step.action == "reconcile":
            snapshot = executor.reconcile(step.device_id or "")
            rows.append({"action": "reconcile", "state_revision": snapshot.state_revision})
            continue
        row, reason, step_forbidden = _process_step(
            scenario.scenario_id, step, executor, emulator, approvals
        )
        rows.append(row)
        reasons.append(reason)
        forbidden += step_forbidden
        processes += 1
    scenario_row = {
        "scenario_id": scenario.scenario_id,
        "safety_case": scenario.safety_case,
        "steps": rows,
    }
    return scenario_row, reasons, forbidden, processes


def evaluate_emulator_fixture(fixture: EmulatorFixture) -> dict[str, Any]:
    approvals = _approvals(fixture)
    scenario_rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    forbidden_invocations = 0
    process_steps = 0
    for scenario in fixture.scenarios:
        row, observed_reasons, forbidden, processes = _evaluate_scenario(
            fixture, scenario, approvals
        )
        scenario_rows.append(row)
        reasons.extend(observed_reasons)
        forbidden_invocations += forbidden
        process_steps += processes
    reason_counts = Counter(reasons)
    missing = sorted(MANDATORY_REASONS - set(reason_counts))
    if missing:
        raise ValueError(f"emulator fixture lacks mandatory gate cases: {missing}")
    return {
        "schema_version": 1,
        "verdict": "pass",
        "scenario_count": len(scenario_rows),
        "process_step_count": process_steps,
        "forbidden_adapter_invocations": forbidden_invocations,
        "reason_counts": dict(sorted(reason_counts.items())),
        "scenarios": scenario_rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Robotics Action Gate and Device Emulator",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Scenarios: {report['scenario_count']}",
        f"- Process steps: {report['process_step_count']}",
        f"- Forbidden adapter invocations: {report['forbidden_adapter_invocations']}",
        "",
        "## Gate outcomes",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{reason}` | {count} |" for reason, count in report["reason_counts"].items())
    return "\n".join(lines) + "\n"


def run_emulator_check(
    fixture_path: Path, data_dir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    fixture = load_emulator_fixture(fixture_path)
    report = evaluate_emulator_fixture(fixture)
    _run_id, timestamp = new_run_timestamp()
    output_dir = resolve_data_dir(data_dir) / METHOD_NAME / timestamp
    report = {**report, "run_id": timestamp}
    atomic_write_text(
        output_dir / "report.json",
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(output_dir / "report.md", _markdown(report))
    LOGGER.info("robotics emulator report written to %s", output_dir)
    return output_dir, report
