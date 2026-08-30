"""Strict policy, approval, fault, and scenario records for the device emulator."""

from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.robotics import (
    ActionProposal,
    ActionReceipt,
    DeviceReference,
    DeviceSnapshot,
    GateDecision,
    RoboticsContract,
    ScalarValue,
)


class ArgumentLimit(RoboticsContract):
    name: str = Field(min_length=1)
    minimum: int | float | None = None
    maximum: int | float | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "ArgumentLimit":
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("minimum must not exceed maximum")
        return self


class StateCondition(RoboticsContract):
    statement: str = Field(min_length=1)
    state_name: str = Field(min_length=1)
    operator: Literal["equals"]
    value: ScalarValue


class OperationPolicy(RoboticsContract):
    device_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    risk_class: Literal["read_only", "low", "medium", "high"]
    idempotency: Literal["read_only", "idempotent", "non_idempotent"]
    argument_limits: tuple[ArgumentLimit, ...] = ()
    preconditions: tuple[StateCondition, ...] = ()
    dependencies: tuple[str, ...] = ()


class ActionPolicy(RoboticsContract):
    schema_version: Literal[1]
    policy_id: str = Field(min_length=1)
    policy_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    approval_required: tuple[Literal["low", "medium", "high"], ...]
    tainted_evidence_ids: tuple[str, ...] = ()
    operations: tuple[OperationPolicy, ...] = Field(min_length=1)


class OperatorApproval(RoboticsContract):
    schema_version: Literal[1]
    approval_id: str = Field(min_length=1)
    approval_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    proposal_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    risk_class: Literal["low", "medium", "high"]


class StateEffect(RoboticsContract):
    operation: str = Field(min_length=1)
    argument_name: str = Field(min_length=1)
    state_name: str = Field(min_length=1)
    mode: Literal["assign", "add"]


class EmulatorDevice(RoboticsContract):
    reference: DeviceReference
    snapshot: DeviceSnapshot
    effects: tuple[StateEffect, ...] = ()


class InjectedFault(RoboticsContract):
    device_id: str = Field(min_length=1)
    operation: str | None = None
    mode: Literal[
        "unreachable_read",
        "write_failed",
        "outcome_unknown_before_apply",
        "outcome_unknown_after_apply",
    ]


class ScenarioSetup(RoboticsContract):
    emergency_stop_devices: tuple[str, ...] = ()
    externally_locked_devices: tuple[str, ...] = ()
    faults: tuple[InjectedFault, ...] = ()


class ProcessExpectation(RoboticsContract):
    decision: Literal["approve", "deny", "escalate"]
    receipt_status: Literal["succeeded", "failed", "outcome_unknown", "not_invoked"]
    reason: str = Field(min_length=1)
    adapter_invocations: int = Field(ge=0)


class ScenarioStep(RoboticsContract):
    action: Literal["process", "reconcile"]
    proposal: ActionProposal | None = None
    approval_id: str | None = None
    device_id: str | None = None
    expected: ProcessExpectation | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "ScenarioStep":
        if self.action == "process":
            if self.proposal is None or self.expected is None or self.device_id is not None:
                raise ValueError("process steps require proposal and expected only")
        elif self.device_id is None or self.proposal is not None or self.expected is not None:
            raise ValueError("reconcile steps require device_id only")
        return self


class EmulatorScenario(RoboticsContract):
    scenario_id: str = Field(min_length=1)
    safety_case: bool
    setup: ScenarioSetup = ScenarioSetup()
    steps: tuple[ScenarioStep, ...] = Field(min_length=1)


class EmulatorFixture(RoboticsContract):
    schema_version: Literal[1]
    devices: tuple[EmulatorDevice, ...] = Field(min_length=1)
    policy: ActionPolicy
    approvals: tuple[OperatorApproval, ...] = ()
    scenarios: tuple[EmulatorScenario, ...] = Field(min_length=1)


class ActionExecution(RoboticsContract):
    decision: GateDecision
    receipt: ActionReceipt
