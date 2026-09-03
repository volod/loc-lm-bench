"""Describe a directory of artifacts as a dataset and read it member by member.

A corpus, a draft bundle, a vector store, a knowledge graph, and a prompt-system package are each
a SET of files that only mean something together. Reading one member and trusting the rest is how
a directory gets half-migrated, so this module names every member of a described directory, binds
it either to its registered record contract or to the owner of its opaque format, and reads all of
them through those bindings -- "this directory is readable" is one answer rather than a series of
separate lucky reads.

Members are DISCOVERED, not assumed: a bundle drafted without chains has no chains file and says
so by omission, a flat store has no parents file, while a member that is present and unreadable is
a refusal.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast


from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.definitions import ContractDefinition
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.artifacts import (
    ARTIFACT_GRANULARITIES,
    ARTIFACT_MEDIA_TYPES,
    ArtifactFormat,
    ContractReference,
    DatasetManifest,
    DatasetMember,
    DatasetQualityCheck,
    OpaqueBinding,
    RecordGranularity,
)

DATASET_MANIFEST_FILE = "dataset_manifest.json"
DATASET_MANIFEST_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
DATASET_OWNER = "loc-lm-bench maintainers"


@dataclass(frozen=True)
class MemberSpec:
    """One project-owned member: where it lives and what contract it must satisfy."""

    member_id: str
    relative_path: str
    schema_id: str
    artifact_format: ArtifactFormat = "json"


@dataclass(frozen=True)
class OpaqueMemberSpec:
    """One member whose bytes belong to somebody else's format.

    A FAISS index, a BM25 postings file, and a DuckDB database are all real members of a store --
    losing one makes it unusable -- but their formats are not this project's to model. The spec
    binds them by digest and names the owner and format version instead.
    """

    member_id: str
    relative_path: str
    owner: str
    format_version: str
    description: str


STRUCTURAL_CHECKS = (
    DatasetQualityCheck(
        check_id="member-contract-dispatch",
        kind="structural",
        description="Every present member resolves to its registered contract's current version.",
    ),
    DatasetQualityCheck(
        check_id="member-digest",
        kind="structural",
        description="Every member's content matches the digest recorded when it was described.",
    ),
)


def describe_dataset(
    root: Path | str,
    dataset_id: str,
    description: str,
    specs: tuple[MemberSpec, ...],
    registry: ContractRegistry = DEFAULT_REGISTRY,
    opaque_specs: tuple[OpaqueMemberSpec, ...] = (),
) -> DatasetManifest:
    """Bind every present member of `root` by identity, digest, and physical format."""
    base = Path(root)
    members = [
        _member(base, spec, registry) for spec in specs if _present(base, spec.relative_path)
    ]
    members.extend(
        _opaque_member(base, spec) for spec in opaque_specs if _present(base, spec.relative_path)
    )
    if not members:
        raise FileNotFoundError(f"{base}: no registered member is present")
    return DatasetManifest(
        schema_id="llb.dataset-manifest",
        schema_version=DATASET_MANIFEST_VERSION,
        dataset_id=dataset_id,
        description=description,
        owner=DATASET_OWNER,
        members=members,
        quality_checks=list(STRUCTURAL_CHECKS),
    )


def publish_dataset_manifest(root: Path | str, manifest: DatasetManifest) -> Path:
    """Write a described dataset's manifest into the directory it describes.

    The manifest is what makes an opaque member checkable at all: a vector index and a postings
    file have no identity of their own, so the digest recorded here is the only thing that says
    the index beside these chunk rows is the index that was built from them. It describes the
    members present when the directory was published; a sidecar written later by another tool is
    simply not among them.
    """
    target = Path(root) / DATASET_MANIFEST_FILE
    target.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def load_dataset_manifest(
    root: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest | None:
    """The published manifest of a directory, or None when it was published without one."""
    path = Path(root) / DATASET_MANIFEST_FILE
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    model = registry.read_current(record, source=str(path))
    return cast(DatasetManifest, model)


def member_digest_problems(root: Path | str, manifest: DatasetManifest) -> tuple[str, ...]:
    """Every member whose bytes are not the ones the manifest recorded, opaque members included."""
    base = Path(root)
    problems: list[str] = []
    for member in manifest.members:
        path = base / member.path
        if not path.is_file():
            if member.required:
                problems.append(f"{path}: described member is missing")
            continue
        observed = member_digest(path)
        if observed != member.digest:
            problems.append(
                f"{path}: digest mismatch; manifest={member.digest}, observed={observed}"
            )
    return tuple(problems)


def _present(root: Path, relative_path: str) -> bool:
    return (root / relative_path).is_file()


def _member(root: Path, spec: MemberSpec, registry: ContractRegistry) -> DatasetMember:
    path = root / spec.relative_path
    definition = registry.definition(spec.schema_id)
    return DatasetMember(
        member_id=spec.member_id,
        path=spec.relative_path,
        format=spec.artifact_format,
        media_type=ARTIFACT_MEDIA_TYPES[spec.artifact_format],
        granularity=cast(RecordGranularity, ARTIFACT_GRANULARITIES[spec.artifact_format]),
        digest=member_digest(path),
        record_contract=ContractReference(
            schema_id=spec.schema_id,
            schema_version=bound_version(path, definition),
        ),
    )


def _opaque_member(root: Path, spec: OpaqueMemberSpec) -> DatasetMember:
    return DatasetMember(
        member_id=spec.member_id,
        path=spec.relative_path,
        format="opaque",
        media_type="application/octet-stream",
        granularity="opaque",
        digest=member_digest(root / spec.relative_path),
        opaque_binding=OpaqueBinding(
            owner=spec.owner,
            format_version=spec.format_version,
            description=spec.description,
        ),
    )


def member_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def bound_version(path: Path, definition: ContractDefinition) -> str:
    """The version a member is bound at: the one it declares, or the family's legacy version.

    A member written by this build carries its own identity and is bound at exactly that version;
    a member written before the registry existed is bound at the version the family declares its
    history to be, which is what the migration then carries forward.
    """
    declared = _declared_version(path)
    if declared is not None and declared in definition.models:
        return declared
    if definition.legacy_version is None:
        raise ValueError(f"{path}: no declared version and no legacy read version")
    return definition.legacy_version


def _declared_version(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    first = next((line for line in text.splitlines() if line.strip()), "")
    try:
        record = json.loads(first) if path.suffix == ".jsonl" else json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("schema_id") is None:
        return None
    version = record.get("schema_version")
    return version if isinstance(version, str) else None
