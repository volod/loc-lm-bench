"""Read every member of a described run bundle and report what each one turned out to be.

A survey, unlike a gate, does not stop at the first refusal: an operator deciding whether a run
can be handed to a board wants the whole list, not its first line. Each member is read by the
reader that owns its family, because three of them accept a shape a generic JSON reader cannot: a
pre-contract manifest, a score table whose columns the run itself declared, and a study record
written in the study's own local form.
"""

from pathlib import Path

from llb.artifacts.bundles import MemberReading
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import ArtifactContractError, DatasetReadError
from llb.artifacts.io import read_bound_member
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.retrieval_graph.opaque import content_digest
from llb.artifacts.run_bundle.manifests import (
    MANIFEST_FILE,
    SCORES_FILE,
    read_run_manifest,
    read_score_rows,
)
from llb.artifacts.run_bundle.studies import read_study_record
from llb.core.contracts.artifacts import DatasetManifest, DatasetMember
from llb.core.contracts.run_bundle.manifest import (
    RUN_MANIFEST_SCHEMA_ID,
    RunArtifactDeclaration,
    RunManifestDocument,
)
from llb.core.contracts.run_bundle.studies import STUDY_ANALYSIS_SCHEMA_ID, STUDY_DESIGN_SCHEMA_ID

STUDY_SCHEMA_IDS = frozenset({STUDY_DESIGN_SCHEMA_ID, STUDY_ANALYSIS_SCHEMA_ID})


def survey_run_bundle(
    root: Path | str, manifest: DatasetManifest, registry: ContractRegistry = DEFAULT_REGISTRY
) -> tuple[MemberReading, ...]:
    """Read every described member and report what each is, including the ones that refuse."""
    root = Path(root)
    head = _head(root, registry)
    return tuple(_reading(root, member, head, registry) for member in manifest.members)


def _head(root: Path, registry: ContractRegistry) -> RunManifestDocument | None:
    try:
        return read_run_manifest(root / MANIFEST_FILE, registry)
    except ArtifactContractError:
        return None


def _reading(
    root: Path,
    member: DatasetMember,
    head: RunManifestDocument | None,
    registry: ContractRegistry,
) -> MemberReading:
    contract = member.record_contract
    schema_id = contract.schema_id if contract is not None else ""
    current = registry.definition(schema_id).current_version if schema_id else ""
    source = contract.schema_version if contract is not None else ""
    try:
        records = _records(root, member, schema_id, head, registry)
    except ArtifactContractError as exc:
        return MemberReading(member.member_id, member.path, schema_id, source, current, 0, str(exc))
    return MemberReading(member.member_id, member.path, schema_id, source, current, records)


def _records(
    root: Path,
    member: DatasetMember,
    schema_id: str,
    head: RunManifestDocument | None,
    registry: ContractRegistry,
) -> int:
    if schema_id == RUN_MANIFEST_SCHEMA_ID:
        read_run_manifest(root / member.path, registry)
        return 1
    if schema_id in STUDY_SCHEMA_IDS:
        declaration = _declaration(head, member.path)
        _check_digest(root / member.path, member.digest)
        read_study_record(
            root / member.path,
            schema_id,
            study_id=declaration.study_id if declaration is not None else None,
            registry=registry,
        )
        return 1
    if member.path == SCORES_FILE:
        _check_digest(root / member.path, member.digest)
        return read_score_rows(
            root / member.path, head.score_rows if head is not None else None, registry
        )
    if member.format == "opaque":
        _check_digest(root / member.path, member.digest)
        return 0
    return len(read_bound_member(root, member, registry))


def _declaration(head: RunManifestDocument | None, name: str) -> RunArtifactDeclaration | None:
    if head is None:
        return None
    return next((artifact for artifact in head.artifacts if artifact.name == name), None)


def _check_digest(path: Path, declared: str) -> None:
    """The one thing every member owes the bundle: the bytes it was published at."""
    if not path.exists():
        raise DatasetReadError(f"{path}: declared bundle member is missing")
    observed = content_digest(path)
    if observed != declared:
        raise DatasetReadError(f"{path}: digest mismatch; manifest={declared}, observed={observed}")
