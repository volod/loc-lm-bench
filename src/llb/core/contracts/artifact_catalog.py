"""Portable machine-readable catalog models for registered artifact contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llb.core.contracts.artifacts import (
    ArtifactContract,
    ArtifactFormat,
    RecordGranularity,
    SCHEMA_ID_PATTERN,
    SEMANTIC_VERSION_PATTERN,
    validate_artifact_binding,
)

SemanticVersionString = Annotated[str, Field(pattern=SEMANTIC_VERSION_PATTERN)]


class FormatBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    format: ArtifactFormat
    media_type: str = Field(min_length=1)
    granularity: RecordGranularity

    @model_validator(mode="after")
    def validate_physical_binding(self) -> "FormatBinding":
        validate_artifact_binding(self.format, self.media_type, self.granularity)
        return self


class CompatibilityDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    from_version: SemanticVersionString
    to_version: SemanticVersionString
    kind: Literal["migration", "refusal"]
    description: str = Field(min_length=1)


class ContractCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: str = Field(pattern=SCHEMA_ID_PATTERN)
    description: str = Field(min_length=1)
    current_version: SemanticVersionString
    supported_read_versions: list[SemanticVersionString]
    deprecation_policy: str = Field(min_length=1)
    schema_paths: dict[str, str]
    bindings: list[FormatBinding]
    compatibility: list[CompatibilityDeclaration]
    extension_point: str | None = None


class ArtifactCatalog(ArtifactContract):
    schema_id: Literal["llb.artifact-catalog"]
    schema_version: Literal["1.0.0"]
    odcs_api_version: Literal["v3.1.0"]
    contracts: list[ContractCatalogEntry]
