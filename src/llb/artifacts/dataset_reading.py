"""Read a described dataset member by member, survey it, and upgrade what an old writer left.

`llb.artifacts.datasets` says what a directory IS -- which members it has and what each is bound
to. This module acts on that description: it reads every member through its binding, reports what
each turned out to be without stopping at the first refusal, and rewrites the members an older
writer produced at the current contract.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import ArtifactContractError
from llb.artifacts.io import read_bound_member
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.artifacts import DatasetManifest, DatasetMember


def read_dataset(
    root: Path | str, manifest: DatasetManifest, registry: ContractRegistry = DEFAULT_REGISTRY
) -> dict[str, tuple[BaseModel, ...]]:
    """Read every described member through its binding, keyed by member id.

    The first member that cannot be read raises; a caller wanting a survey rather than a gate
    uses `survey_dataset`, which reports all of them.
    """
    return {
        member.member_id: read_bound_member(Path(root), member, registry)
        for member in manifest.members
    }


@dataclass(frozen=True)
class MemberReading:
    """What one member of a described dataset turned out to be."""

    member_id: str
    path: str
    schema_id: str
    source_version: str
    current_version: str
    records: int
    refusal: str = ""
    # Set only for an opaque member: what the manifest names instead of a record contract.
    owner: str = ""
    format_version: str = ""

    @property
    def needs_upgrade(self) -> bool:
        return not self.refusal and self.source_version != self.current_version

    @property
    def is_opaque(self) -> bool:
        return bool(self.owner)


def survey_dataset(
    root: Path | str, manifest: DatasetManifest, registry: ContractRegistry = DEFAULT_REGISTRY
) -> tuple[MemberReading, ...]:
    """Read every member and report what each is, including the ones that refuse.

    A survey, unlike `read_dataset`, does not stop at the first refusal: an operator deciding
    whether a directory can be handed on wants the whole list, not its first line.
    """
    return tuple(_reading(Path(root), member, registry) for member in manifest.members)


def upgrade_dataset(
    root: Path | str, manifest: DatasetManifest, registry: ContractRegistry = DEFAULT_REGISTRY
) -> tuple[str, ...]:
    """Rewrite every member written at an older version at the current one, in place.

    Only members the registry can migrate are touched, and each is rewritten from the migrated
    record rather than edited, so a member that was already current is left byte-for-byte alone.
    Returns the ids that were rewritten.
    """
    upgraded: list[str] = []
    for member in manifest.members:
        reading = _reading(Path(root), member, registry)
        if not reading.needs_upgrade:
            continue
        records = read_bound_member(Path(root), member, registry)
        _rewrite(Path(root) / member.path, member.format, records)
        upgraded.append(member.member_id)
    return tuple(upgraded)


def _rewrite(path: Path, artifact_format: str, records: tuple[BaseModel, ...]) -> None:
    if artifact_format == "jsonl":
        body = "".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for record in records
        )
    else:
        body = json.dumps(records[0].model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    path.write_text(body, encoding="utf-8")


def _reading(root: Path, member: DatasetMember, registry: ContractRegistry) -> MemberReading:
    contract = member.record_contract
    binding = member.opaque_binding
    schema_id = contract.schema_id if contract is not None else ""
    current = registry.definition(schema_id).current_version if schema_id else ""
    source = contract.schema_version if contract is not None else ""
    try:
        records = read_bound_member(root, member, registry)
    except ArtifactContractError as exc:
        return MemberReading(member.member_id, member.path, schema_id, source, current, 0, str(exc))
    return MemberReading(
        member.member_id,
        member.path,
        schema_id,
        source,
        current,
        len(records),
        owner=binding.owner if binding is not None else "",
        format_version=binding.format_version if binding is not None else "",
    )
