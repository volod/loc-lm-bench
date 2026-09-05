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
    """Who owns a member this project stores but does not define the bytes of.

    A FAISS index, a DuckDB database, and a persisted BM25 posting list are all members a store
    cannot be read without, and none of them is a record this project may model: their layout is
    decided by their own library and moves with its releases. Naming the owner and the format
    version it writes is what a reader needs -- enough to say "this member is a FAISS index at
    format 1, and I do not have FAISS" rather than guessing from a file extension.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    owner: str = Field(min_length=1)  # the library or component whose format this is
    format: str = Field(min_length=1)  # the owner's own name for the format, not a media type
    format_version: str = Field(min_length=1)  # the owner's version of that format
    description: str = Field(min_length=1)


def validate_member_binding(member: "DatasetMemberV1 | DatasetMember") -> None:
    """The binding rules every manifest version shares, stated once."""
    path = PurePosixPath(member.path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("dataset member path must be relative and may not contain '..'")
    validate_artifact_binding(member.format, member.media_type, member.granularity)
    if member.format == "opaque":
        if member.granularity != "opaque" or member.record_contract is not None:
            raise ValueError("opaque members require opaque granularity and no record contract")
    elif member.granularity == "opaque" or member.record_contract is None:
        raise ValueError("structured members require a record contract and non-opaque granularity")


class DatasetMemberV1(BaseModel):
    """A member at manifest version 1, which could not say whose format an opaque one is."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    member_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    path: str = Field(min_length=1)
    format: ArtifactFormat
    media_type: str = Field(min_length=1)
    granularity: RecordGranularity
    required: bool = True
    digest: str = Field(pattern=DIGEST_PATTERN)
    record_contract: ContractReference | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "DatasetMemberV1":
        validate_member_binding(self)
        return self


class DatasetMember(BaseModel):
    """A member at manifest version 1.1: an opaque one must say whose format it is."""

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
        validate_member_binding(self)
        if self.format == "opaque" and self.opaque_binding is None:
            raise ValueError("an opaque member must declare its opaque_binding owner and format")
        if self.format != "opaque" and self.opaque_binding is not None:
            raise ValueError("only an opaque member may declare an opaque_binding")
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


def validate_dataset_members(
    members: "list[DatasetMemberV1] | list[DatasetMember]",
    relationships: list[DatasetRelationship],
) -> None:
    """The dataset rules every manifest version shares, stated once."""
    member_ids = [member.member_id for member in members]
    paths = [member.path for member in members]
    if len(member_ids) != len(set(member_ids)):
        raise ValueError("dataset member ids must be unique")
    if len(paths) != len(set(paths)):
        raise ValueError("dataset member paths must be unique")
    known = set(member_ids)
    for relationship in relationships:
        if relationship.from_member not in known or relationship.to_member not in known:
            raise ValueError("dataset relationships must reference declared member ids")


class DatasetManifestV1(ExtensibleArtifactContract):
    """A dataset at manifest version 1: every member bound, an opaque one unattributed."""

    schema_id: Literal["llb.dataset-manifest"]
    schema_version: Literal["1.0.0"]
    dataset_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    members: list[DatasetMemberV1] = Field(min_length=1)
    relationships: list[DatasetRelationship] = Field(default_factory=list)
    quality_checks: list[DatasetQualityCheck] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> "DatasetManifestV1":
        validate_dataset_members(self.members, self.relationships)
        return self


class DatasetManifest(ExtensibleArtifactContract):
    """The current manifest: an opaque member names its owner and that owner's format version."""

    schema_id: Literal["llb.dataset-manifest"]
    schema_version: Literal["1.1.0"]
    dataset_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    members: list[DatasetMember] = Field(min_length=1)
    relationships: list[DatasetRelationship] = Field(default_factory=list)
    quality_checks: list[DatasetQualityCheck] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> "DatasetManifest":
        validate_dataset_members(self.members, self.relationships)
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
