"""Strict prospective-design, task, model-decision, and run records."""

from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.robotics import NamedValue, RoboticsContract


class BenchmarkDesign(RoboticsContract):
    schema_version: Literal[1]
    benchmark_id: str = Field(min_length=1)
    split: Literal["final"]
    frozen: Literal[True]
    task_ledger: str = Field(min_length=1)
    task_ledger_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    minimum_detectable_gain: float = Field(gt=0, le=1)
    minimum_evidence_count: int = Field(gt=0)
    confidence: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=1000)
    seed: int = Field(ge=0)
    mandatory_fault_classes: tuple[str, ...] = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    chunk_strategy: str = Field(min_length=1)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    generation_max_tokens: int = Field(gt=0)
    request_timeout_s: float = Field(gt=0)


class BenchmarkSetup(RoboticsContract):
    observed_state_revision: int | None = Field(default=None, ge=0)
    live_state_revision: int | None = Field(default=None, ge=0)
    state: tuple[NamedValue, ...] = ()
    emergency_stop: bool = False
    external_lock: bool = False
    fault: Literal["write_failed", "unreachable_read"] | None = None
    prior_ambiguous_write: bool = False
    runtime_notice: str | None = None


class BenchmarkTask(RoboticsContract):
    schema_version: Literal[1]
    task_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    expected_behavior: Literal["complete", "refuse"]
    safety_class: str | None = None
    device_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    arguments: tuple[NamedValue, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    risk_class: Literal["read_only", "low", "medium", "high"]
    idempotency: Literal["read_only", "idempotent", "non_idempotent"]
    approval_available: bool
    expected_evidence_ids: tuple[str, ...] = ()
    recovery_expected: bool = False
    setup: BenchmarkSetup = BenchmarkSetup()


class ModelProposal(RoboticsContract):
    device_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    arguments: tuple[NamedValue, ...]
    evidence_ids: tuple[str, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    risk_class: Literal["read_only", "low", "medium", "high"]
    idempotency: Literal["read_only", "idempotent", "non_idempotent"]


class ModelDecision(RoboticsContract):
    decision: Literal["propose", "refuse", "escalate"]
    reason: str = Field(min_length=1)
    proposal: ModelProposal | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "ModelDecision":
        if (self.decision == "propose") != (self.proposal is not None):
            raise ValueError("only a propose decision carries a proposal")
        return self
