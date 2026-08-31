"""Strict fixture, upstream-pin, and compatibility-report models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llb.core.contracts.robotics import (
    ActionProposal,
    ActionReceipt,
    DeviceReference,
    DeviceSnapshot,
    GateDecision,
    NamedValue,
    RoboticsEvidence,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReferencePin(StrictModel):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ConformanceInput(StrictModel):
    name: str = Field(min_length=1)
    schema_revision: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    verdict: Literal["pass", "fail"]


class UpstreamPin(StrictModel):
    id: Literal["hflow", "mhs"]
    source_url: str = Field(min_length=1)
    release: str = Field(min_length=1)
    revision: str | None
    contract_status: Literal["contract-inspectable", "public-semantics-only"]
    license: str | None
    license_url: str | None
    transports: tuple[str, ...] = Field(min_length=1)
    semantics: tuple[str, ...] = Field(min_length=1)
    normative_reference: str | None
    references: tuple[ReferencePin, ...] = Field(min_length=1)
    conformance_input: ConformanceInput | None = None


class UpstreamPins(StrictModel):
    schema_version: Literal[1]
    sources: tuple[UpstreamPin, ...] = Field(min_length=2, max_length=2)


class PublicSemantics(StrictModel):
    discover: str = Field(min_length=1)
    read: str = Field(min_length=1)
    write: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    limits: str = Field(min_length=1)


class MhsPublicRecord(StrictModel):
    schema_version: Literal[1]
    publication: str = Field(min_length=1)
    announcement_url: str = Field(min_length=1)
    preview_url: str = Field(min_length=1)
    access: Literal["limited-application-preview"]
    normative_schema_public: Literal[False]
    public_license: None
    public_semantics: PublicSemantics


class BoundaryRecords(StrictModel):
    schema_version: Literal[1]
    evidence: RoboticsEvidence
    device_reference: DeviceReference
    device_snapshot: DeviceSnapshot
    action_proposal: ActionProposal
    gate_decision: GateDecision
    action_receipt: ActionReceipt


class FakeExercise(StrictModel):
    schema_version: Literal[1]
    read_operation: str = Field(min_length=1)
    write_proposal_id: str = Field(min_length=1)
    limit_operation: str = Field(min_length=1)
    limit_arguments: tuple[NamedValue, ...] = Field(min_length=1)


class FilePin(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class UpstreamExpectation(StrictModel):
    id: Literal["hflow", "mhs"]
    pin_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class FixtureManifest(StrictModel):
    schema_version: Literal[1]
    contract_schema_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    files: tuple[FilePin, ...] = Field(min_length=1)
    upstream_pins: tuple[UpstreamExpectation, ...] = Field(min_length=2, max_length=2)


class CompatibilityResult(StrictModel):
    outcome: Literal["protocol-neutral", "contract-inspectable", "MHS-compatible"]
    reason: str = Field(min_length=1)
