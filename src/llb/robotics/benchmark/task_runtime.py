"""Prepare one task emulator and execute a parsed proposal through its action gate."""

from dataclasses import dataclass
from typing import Literal

from llb.core.contracts.robotics import ActionReceipt, DeviceSnapshot
from llb.robotics.action_executor import ActionExecutor
from llb.robotics.benchmark.models import BenchmarkTask, ModelDecision
from llb.robotics.benchmark.parser import action_proposal
from llb.robotics.device_emulator import DeviceEmulator
from llb.robotics.digests import value_digest
from llb.robotics.emulator_models import (
    ActionExecution,
    EmulatorDevice,
    EmulatorFixture,
    InjectedFault,
    OperatorApproval,
)


@dataclass(frozen=True)
class PreparedTask:
    observed: DeviceSnapshot
    emulator: DeviceEmulator
    executor: ActionExecutor


@dataclass(frozen=True)
class ExecutedDecision:
    execution: ActionExecution | None
    setup_receipt: ActionReceipt | None
    adapter_invocations: int
    recovery_success: bool


def _device_state(
    fixture: EmulatorFixture, task: BenchmarkTask
) -> tuple[tuple[EmulatorDevice, ...], DeviceSnapshot]:
    devices: list[EmulatorDevice] = []
    observed: DeviceSnapshot | None = None
    for device in fixture.devices:
        snapshot = device.snapshot
        if device.reference.device_id == task.device_id:
            live_revision = task.setup.live_state_revision
            updates: dict[str, object] = {}
            if live_revision is not None:
                updates.update(
                    state_revision=live_revision,
                    snapshot_id=f"{task.device_id}:state:{live_revision}",
                )
            if task.setup.state:
                updates["state"] = task.setup.state
            snapshot = snapshot.model_copy(update=updates)
            observed_revision = task.setup.observed_state_revision
            observed = snapshot.model_copy(
                update={
                    "state_revision": observed_revision
                    if observed_revision is not None
                    else snapshot.state_revision,
                    "snapshot_id": (
                        f"{task.device_id}:observed:{observed_revision}"
                        if observed_revision is not None
                        else snapshot.snapshot_id
                    ),
                }
            )
        devices.append(device.model_copy(update={"snapshot": snapshot}))
    if observed is None:
        raise ValueError(f"task {task.task_id} names unknown device {task.device_id}")
    return tuple(devices), observed


def _configure(emulator: DeviceEmulator, task: BenchmarkTask) -> None:
    if task.setup.emergency_stop:
        emulator.set_emergency_stop(task.device_id)
    if task.setup.external_lock:
        locked = "arm-cell" if task.device_id == "clamp-cell" else task.device_id
        emulator.set_external_lock(locked)
    if task.setup.fault:
        emulator.inject_fault(
            InjectedFault(
                device_id=task.device_id,
                operation=None if task.setup.fault == "unreachable_read" else task.operation,
                mode=task.setup.fault,
            )
        )


def prepare_task(fixture: EmulatorFixture, task: BenchmarkTask) -> PreparedTask:
    devices, observed = _device_state(fixture, task)
    emulator = DeviceEmulator(devices)
    _configure(emulator, task)
    return PreparedTask(observed, emulator, ActionExecutor(emulator, fixture.policy))


def _approval(
    proposal_digest: str, risk_class: Literal["low", "medium", "high"]
) -> OperatorApproval:
    approval = OperatorApproval(
        schema_version=1,
        approval_id=f"approval:{proposal_digest.removeprefix('sha256:')[:16]}",
        approval_digest=f"sha256:{'0' * 64}",
        proposal_digest=proposal_digest,
        risk_class=risk_class,
    )
    digest = value_digest(approval.model_dump(mode="json", exclude={"approval_digest"}))
    return approval.model_copy(update={"approval_digest": digest})


def _selected_approval(task: BenchmarkTask, proposal_digest: str, risk_class: str):  # type: ignore[no-untyped-def]
    if not task.approval_available or risk_class == "read_only":
        return None
    if risk_class not in {"low", "medium", "high"}:
        raise ValueError(f"unsupported approval risk class: {risk_class}")
    return _approval(proposal_digest, risk_class)  # type: ignore[arg-type]


def _seed_ambiguous(
    task: BenchmarkTask,
    prepared: PreparedTask,
    proposal: object,
    approval: OperatorApproval | None,
) -> ActionReceipt | None:
    if not task.setup.prior_ambiguous_write:
        return None
    from llb.core.contracts.robotics import ActionProposal

    if not isinstance(proposal, ActionProposal):
        raise TypeError("ambiguous setup requires an action proposal")
    prepared.emulator.inject_fault(
        InjectedFault(
            device_id=proposal.device_id,
            operation=proposal.operation,
            mode="outcome_unknown_after_apply",
        )
    )
    return prepared.executor.process(proposal, approval).receipt


def execute_decision(
    task: BenchmarkTask,
    lane: str,
    decision: ModelDecision,
    prepared: PreparedTask,
) -> ExecutedDecision:
    if decision.proposal is None:
        return ExecutedDecision(None, None, 0, False)
    proposal = action_proposal(decision, task, lane, prepared.observed.state_revision)
    approval = _selected_approval(task, proposal.proposal_digest, proposal.risk_class)
    setup_receipt = _seed_ambiguous(task, prepared, proposal, approval)
    before = prepared.emulator.invocation_count
    execution = prepared.executor.process(proposal, approval)
    recovery_success = False
    if task.recovery_expected and execution.receipt.status == "failed":
        prepared.executor.reconcile(task.device_id)
        execution = prepared.executor.process(proposal, approval)
        recovery_success = execution.receipt.status == "succeeded"
    return ExecutedDecision(
        execution,
        setup_receipt,
        prepared.emulator.invocation_count - before,
        recovery_success,
    )
