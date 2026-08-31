"""Strict models for the HFlow projection manifest and evidence ledger."""

from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.robotics import ProducerVersion, RoboticsContract, RoboticsEvidence

QualityState = Literal["accepted", "quarantined", "unverified"]
AdmissionState = Literal["accepted", "draft", "quarantined", "unverified"]


class HflowGeneration(RoboticsContract):
    hflow_release: str = Field(min_length=1)
    hflow_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    hflow_schema_version: str = Field(min_length=1)
    pipeline_version: str = Field(min_length=1)
    curation_query_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    check_versions: tuple[ProducerVersion, ...] = Field(min_length=1)
    enrichment_versions: tuple[ProducerVersion, ...] = Field(min_length=1)


class HflowProjectionRow(RoboticsContract):
    bridge_schema_version: Literal[1]
    hflow_release: str = Field(min_length=1)
    hflow_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    hflow_schema_version: str = Field(min_length=1)
    pipeline_version: str = Field(min_length=1)
    curation_query_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    check_versions: tuple[ProducerVersion, ...] = Field(min_length=1)
    enrichment_versions: tuple[ProducerVersion, ...] = Field(min_length=1)
    episode_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    mcap_uri: str = Field(min_length=1)
    mcap_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    channels: tuple[str, ...] = Field(min_length=1)
    start_ns: int = Field(ge=0)
    end_ns: int = Field(gt=0)
    quality_state: QualityState
    quarantine_tags: tuple[str, ...] = ()
    projection_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    projection_kind: Literal["caption", "label", "procedure_summary", "other"]
    authored_by: Literal["human", "model", "pipeline"]
    verified: bool
    verification_ref: str | None = None
    language: str = Field(min_length=2)
    projection_uri: str = Field(min_length=1)
    projection_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    projection_start: int = Field(ge=0)
    projection_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "HflowProjectionRow":
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")
        if self.projection_end <= self.projection_start:
            raise ValueError("projection_end must be greater than projection_start")
        if self.quality_state == "quarantined" and not self.quarantine_tags:
            raise ValueError("a quarantined projection must preserve its quarantine tags")
        return self

    def generation(self) -> HflowGeneration:
        return HflowGeneration(
            hflow_release=self.hflow_release,
            hflow_revision=self.hflow_revision,
            hflow_schema_version=self.hflow_schema_version,
            pipeline_version=self.pipeline_version,
            curation_query_digest=self.curation_query_digest,
            check_versions=self.check_versions,
            enrichment_versions=self.enrichment_versions,
        )


class HflowFixtureFile(RoboticsContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class HflowFixtureManifest(RoboticsContract):
    schema_version: Literal[1]
    hflow_release: str = Field(min_length=1)
    hflow_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    manifest: str = Field(min_length=1)
    files: tuple[HflowFixtureFile, ...] = Field(min_length=1)


class CorpusSpan(RoboticsContract):
    doc_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_length(self) -> "CorpusSpan":
        if len(self.text) != self.char_end - self.char_start:
            raise ValueError("source-span text length does not match its offsets")
        return self


class EvidenceLedgerEntry(RoboticsContract):
    schema_version: Literal[1]
    projection_id: str = Field(min_length=1)
    admission: AdmissionState
    admission_reason: str = Field(min_length=1)
    projection_kind: str = Field(min_length=1)
    authored_by: str = Field(min_length=1)
    language: str = Field(min_length=2)
    verification_ref: str | None
    quarantine_tags: tuple[str, ...]
    generation: HflowGeneration
    source_span: CorpusSpan | None
    evidence: RoboticsEvidence
