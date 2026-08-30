"""Immutable exchange records at the robotics evidence and operation boundary."""

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest: TypeAlias = str
ScalarValue: TypeAlias = str | int | float | bool | None


class RoboticsContract(BaseModel):
    """Strict base for records that may authorize or describe physical effects."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NamedValue(RoboticsContract):
    name: str = Field(min_length=1)
    value: ScalarValue


class ProducerVersion(RoboticsContract):
    producer: str = Field(min_length=1)
    version: str = Field(min_length=1)


class RoboticsEvidence(RoboticsContract):
    schema_version: Literal[1]
    evidence_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    mcap_uri: str = Field(min_length=1)
    mcap_sha256: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    channels: tuple[str, ...] = Field(min_length=1)
    start_ns: int = Field(ge=0)
    end_ns: int = Field(gt=0)
    producer_versions: tuple[ProducerVersion, ...] = Field(min_length=1)
    quality_state: Literal["accepted", "quarantined", "unverified"]
    projection_uri: str = Field(min_length=1)
    projection_start: int = Field(ge=0)
    projection_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_intervals(self) -> "RoboticsEvidence":
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")
        if self.projection_end <= self.projection_start:
            raise ValueError("projection_end must be greater than projection_start")
        return self


class DeviceParameter(RoboticsContract):
    name: str = Field(min_length=1)
    value_type: Literal["string", "integer", "number", "boolean"]
    required: bool
    minimum: int | float | None = None
    maximum: int | float | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "DeviceParameter":
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("minimum must not exceed maximum")
        return self


class DeviceOperation(RoboticsContract):
    name: str = Field(min_length=1)
    access: Literal["read", "write"]
    description: str = Field(min_length=1)
    parameters: tuple[DeviceParameter, ...] = ()


class DeviceReference(RoboticsContract):
    schema_version: Literal[1]
    driver_id: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    reference_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    operations: tuple[DeviceOperation, ...] = Field(min_length=1)


class DeviceSnapshot(RoboticsContract):
    schema_version: Literal[1]
    snapshot_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    driver_id: str = Field(min_length=1)
    driver_version: str = Field(min_length=1)
    state_revision: int = Field(ge=0)
    observed_at: str = Field(min_length=1)
    reference_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    operations: tuple[DeviceOperation, ...] = Field(min_length=1)
    state: tuple[NamedValue, ...]


class ActionProposal(RoboticsContract):
    schema_version: Literal[1]
    proposal_id: str = Field(min_length=1)
    proposal_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    device_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    arguments: tuple[NamedValue, ...]
    expected_state_revision: int = Field(ge=0)
    evidence_ids: tuple[str, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    risk_class: Literal["read_only", "low", "medium", "high"]
    idempotency: Literal["read_only", "idempotent", "non_idempotent"]


class GateDecision(RoboticsContract):
    schema_version: Literal[1]
    decision_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    proposal_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    snapshot_id: str = Field(min_length=1)
    policy_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    decision: Literal["approve", "deny", "escalate"]
    reasons: tuple[str, ...]
    approval_id: str | None = None
    approval_digest: Digest | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_approval_binding(self) -> "GateDecision":
        if (self.approval_id is None) != (self.approval_digest is None):
            raise ValueError("approval_id and approval_digest must be supplied together")
        return self


class ActionReceipt(RoboticsContract):
    schema_version: Literal[1]
    receipt_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    proposal_digest: Digest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    device_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    status: Literal["succeeded", "failed", "outcome_unknown", "not_invoked"]
    state_revision_before: int = Field(ge=0)
    state_revision_after: int | None = Field(default=None, ge=0)
    result: tuple[NamedValue, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "ActionReceipt":
        if self.status == "succeeded" and self.state_revision_after is None:
            raise ValueError("a successful receipt must name the resulting state revision")
        if self.status == "not_invoked" and self.state_revision_after is not None:
            raise ValueError("a not-invoked receipt cannot claim a resulting state revision")
        return self
