"""Shared strict models for versioned artifact and dataset contracts."""

from pathlib import PurePosixPath
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExtensionValue: TypeAlias = str | int | float | bool | None
ArtifactFormat: TypeAlias = Literal["json", "jsonl", "yaml", "csv", "parquet", "opaque"]
RecordGranularity: TypeAlias = Literal["document", "row", "table", "opaque"]

SCHEMA_ID_PATTERN = r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$"
SEMANTIC_VERSION_PATTERN = r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
ARTIFACT_MEDIA_TYPES = {
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "yaml": "application/yaml",
    "csv": "text/csv",
    "parquet": "application/vnd.apache.parquet",
}
ARTIFACT_GRANULARITIES = {
    "json": "document",
    "jsonl": "row",
    "yaml": "document",
    "csv": "row",
    "parquet": "row",
}


def validate_artifact_binding(
    artifact_format: ArtifactFormat, media_type: str, granularity: RecordGranularity
) -> None:
    """Reject physical binding metadata that contradicts its declared format."""
    if artifact_format == "opaque":
        if granularity != "opaque":
            raise ValueError("opaque formats require opaque granularity")
        return
    expected_media_type = ARTIFACT_MEDIA_TYPES[artifact_format]
    if media_type != expected_media_type:
        raise ValueError(f"{artifact_format} formats require media_type={expected_media_type!r}")
    expected_granularity = ARTIFACT_GRANULARITIES[artifact_format]
    if granularity != expected_granularity:
        raise ValueError(f"{artifact_format} formats require granularity={expected_granularity!r}")


class ArtifactContract(BaseModel):
    """Strict immutable base for one version of a durable record contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: str = Field(pattern=SCHEMA_ID_PATTERN)
    schema_version: str = Field(pattern=SEMANTIC_VERSION_PATTERN)


class ExtensibleArtifactContract(ArtifactContract):
    """Artifact contract with the only supported open-ended extension point."""

    extensions: dict[str, ExtensionValue] = Field(default_factory=dict)


class ContractReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: str = Field(pattern=SCHEMA_ID_PATTERN)
    schema_version: str = Field(pattern=SEMANTIC_VERSION_PATTERN)


class OpaqueBinding(BaseModel):
    """Who owns an opaque member's bytes, and which of their formats it is written in.

    This project binds a FAISS index, a lexical postings file, or a DuckDB database into a
    dataset without modelling their contents: the format belongs to its own owner and is theirs
    to evolve. What a reader still needs is the two facts a digest cannot supply -- whose format
    this is and which version of it -- so an incompatible member is named rather than guessed at.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    owner: str = Field(min_length=1)
    format_version: str = Field(min_length=1)
    description: str = Field(min_length=1)


class DatasetMember(BaseModel):
    """One member of a dataset, bound either to a record contract or to the owner of its format."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    member_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    path: str = Field(min_length=1)
    format: ArtifactFormat
    media_type: str = Field(min_length=1)
    granularity: RecordGranularity
    required: bool = True
    digest: str = Field(pattern=DIGEST_PATTERN)
    record_contract: ContractReference | None = None
    opaque_binding: OpaqueBinding | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "DatasetMember":
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("dataset member path must be relative and may not contain '..'")
        validate_artifact_binding(self.format, self.media_type, self.granularity)
        if self.format == "opaque":
            if self.granularity != "opaque" or self.record_contract is not None:
                raise ValueError("opaque members require opaque granularity and no record contract")
            if self.opaque_binding is None:
                raise ValueError("opaque members must declare their owner and format version")
            return self
        if self.granularity == "opaque" or self.record_contract is None:
            raise ValueError(
                "structured members require a record contract and non-opaque granularity"
            )
        if self.opaque_binding is not None:
            raise ValueError("only opaque members carry an opaque binding")
        return self


class DatasetRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    from_member: str = Field(min_length=1)
    to_member: str = Field(min_length=1)
    description: str = Field(min_length=1)


class DatasetQualityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    check_id: str = Field(min_length=1)
    kind: Literal["structural", "domain"]
    description: str = Field(min_length=1)


class DatasetManifest(ExtensibleArtifactContract):
    """A physical dataset: its members, how they relate, and what is checked about them."""

    schema_id: Literal["llb.dataset-manifest"]
    schema_version: Literal["1.0.0"]
    dataset_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    members: list[DatasetMember] = Field(min_length=1)
    relationships: list[DatasetRelationship] = Field(default_factory=list)
    quality_checks: list[DatasetQualityCheck] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> "DatasetManifest":
        member_ids = [member.member_id for member in self.members]
        paths = [member.path for member in self.members]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("dataset member ids must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("dataset member paths must be unique")
        known = set(member_ids)
        for relationship in self.relationships:
            if relationship.from_member not in known or relationship.to_member not in known:
                raise ValueError("dataset relationships must reference declared member ids")
        return self


class CompatibilityProbeV1(ArtifactContract):
    """Committed old form used to exercise the compatibility mechanism itself."""

    schema_id: Literal["llb.artifact-contract.compatibility-probe"]
    schema_version: Literal["1.0.0"]
    name: str = Field(min_length=1)


class CompatibilityProbeV2(ExtensibleArtifactContract):
    """Current compatibility probe; ``label`` retains the v1 ``name`` meaning."""

    schema_id: Literal["llb.artifact-contract.compatibility-probe"]
    schema_version: Literal["2.0.0"]
    label: str = Field(min_length=1)
