"""Read every member of a described generation and report what each one turned out to be.

A survey, unlike a gate, does not stop at the first refusal: an operator deciding whether a store
can be handed on wants the whole list, not its first line. Each member is read by the reader that
owns its family, because three of them accept a pre-contract shape a generic JSON reader cannot:
a bare array, a bare mapping, or a metadata document from before the registry existed.
"""

from collections.abc import Callable
from pathlib import Path

from llb.artifacts.bundles import MemberReading
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import ArtifactContractError, DatasetReadError
from llb.artifacts.io import read_bound_member
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.retrieval_graph.graphs import read_community_summaries, read_graph_meta
from llb.artifacts.retrieval_graph.opaque import content_digest
from llb.artifacts.retrieval_graph.prompt_systems import (
    PACKAGE_FIELDS,
    read_member,
    read_prompt_system_manifest,
)
from llb.artifacts.retrieval_graph.stores import read_store_meta
from llb.core.contracts.artifacts import DatasetManifest, DatasetMember
from llb.core.contracts.retrieval_graph.graph import (
    COMMUNITY_SUMMARIES_SCHEMA_ID,
    GRAPH_META_SCHEMA_ID,
)
from llb.core.contracts.retrieval_graph.prompt_system import PROMPT_SYSTEM_MANIFEST_SCHEMA_ID
from llb.core.contracts.retrieval_graph.stores import STORE_META_SCHEMA_ID

DocumentReader = Callable[[Path, ContractRegistry], int]


def _one(read: Callable[[Path, ContractRegistry], object]) -> DocumentReader:
    """A reader of a single-document member: it resolves, so the member holds one record."""

    def reader(path: Path, registry: ContractRegistry) -> int:
        read(path, registry)
        return 1

    return reader


def _wrapped(schema_id: str) -> DocumentReader:
    """A reader of a member whose one field holds the list or mapping the file used to be."""

    def reader(path: Path, registry: ContractRegistry) -> int:
        return len(read_member(path, schema_id, registry))

    return reader


DOCUMENT_READERS: dict[str, DocumentReader] = {
    STORE_META_SCHEMA_ID: _one(read_store_meta),
    GRAPH_META_SCHEMA_ID: _one(read_graph_meta),
    PROMPT_SYSTEM_MANIFEST_SCHEMA_ID: _one(read_prompt_system_manifest),
    COMMUNITY_SUMMARIES_SCHEMA_ID: lambda path, registry: len(
        read_community_summaries(path, registry)
    ),
    **{schema_id: _wrapped(schema_id) for schema_id in PACKAGE_FIELDS},
}


def survey_generation(
    root: Path | str, manifest: DatasetManifest, registry: ContractRegistry = DEFAULT_REGISTRY
) -> tuple[MemberReading, ...]:
    """Read every described member and report what each is, including the ones that refuse."""
    return tuple(_reading(Path(root), member, registry) for member in manifest.members)


def _reading(root: Path, member: DatasetMember, registry: ContractRegistry) -> MemberReading:
    contract = member.record_contract
    schema_id = contract.schema_id if contract is not None else ""
    current = registry.definition(schema_id).current_version if schema_id else ""
    source = contract.schema_version if contract is not None else ""
    try:
        records = _records(root, member, schema_id, registry)
    except ArtifactContractError as exc:
        return MemberReading(member.member_id, member.path, schema_id, source, current, 0, str(exc))
    return MemberReading(member.member_id, member.path, schema_id, source, current, records)


def _records(root: Path, member: DatasetMember, schema_id: str, registry: ContractRegistry) -> int:
    if member.format == "opaque":
        return _opaque_records(root, member)
    reader = DOCUMENT_READERS.get(schema_id)
    if reader is not None and member.format == "json":
        return reader(root / member.path, registry)
    return len(read_bound_member(root, member, registry))


def _opaque_records(root: Path, member: DatasetMember) -> int:
    """An opaque member is checked for the one thing this project owns about it: its bytes."""
    path = root / member.path
    observed = content_digest(path)
    if observed != member.digest:
        raise DatasetReadError(
            f"{path}: digest mismatch; manifest={member.digest}, observed={observed}"
        )
    return 0
