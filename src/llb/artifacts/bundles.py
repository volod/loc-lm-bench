"""Describe a data-prep bundle as a dataset and validate it member by member.

A corpus directory and a draft bundle are each a SET of files that only mean something together:
the gold rows index into the corpus copy, the provenance names the corpus version, the citation
sidecars locate a span on a page. Reading one member and trusting the rest is how a bundle gets
half-migrated. This module names every project-owned member of each, binds it to its registered
contract with a content digest, and reads all of them through that binding, so "this bundle is
readable" is one answer rather than a series of separate lucky reads.

Members are DISCOVERED, not assumed: a bundle drafted without chains has no chains file and says
so by omission, while a member that is present and unreadable is a refusal.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import ArtifactContractError
from llb.artifacts.definitions import ContractDefinition
from llb.artifacts.io import read_bound_member
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.artifacts import (
    ARTIFACT_GRANULARITIES,
    ARTIFACT_MEDIA_TYPES,
    ArtifactFormat,
    ContractReference,
    DatasetManifest,
    DatasetMember,
    DatasetQualityCheck,
    RecordGranularity,
)

CORPUS_DATASET_ID = "llb-staged-corpus"
DRAFT_DATASET_ID = "llb-draft-bundle"
PDF_CITATION_SUFFIX = ".citations.json"
OVERLAY_RELATIVE_PATH = ".llb/conflict_overlay.json"


@dataclass(frozen=True)
class MemberSpec:
    """One project-owned member of a bundle: where it lives and what contract it must satisfy."""

    member_id: str
    relative_path: str
    schema_id: str
    artifact_format: ArtifactFormat = "json"


CORPUS_MEMBERS: tuple[MemberSpec, ...] = (
    MemberSpec("corpus-manifest", "corpus_manifest.json", "llb.corpus-manifest"),
    MemberSpec("pdf-manifest", "pdf_corpus_manifest.json", "llb.pdf-corpus-manifest"),
    MemberSpec("pdf-quality", "pdf_corpus_quality.json", "llb.pdf-corpus-manifest"),
    MemberSpec("conflict-overlay", OVERLAY_RELATIVE_PATH, "llb.conflict-overlay"),
)

DRAFT_MEMBERS: tuple[MemberSpec, ...] = (
    MemberSpec("gold-items", "goldset.jsonl", "llb.gold-item", "jsonl"),
    MemberSpec("gold-chains", "chains.jsonl", "llb.gold-chain", "jsonl"),
    MemberSpec("ontology", "ontology.json", "llb.ontology"),
    MemberSpec("extraction", "extraction.jsonl", "llb.ontology-extraction", "jsonl"),
    MemberSpec("provenance", "provenance.json", "llb.ontology-provenance"),
    MemberSpec("import-report", "import_report.json", "llb.external-draft-provenance"),
    MemberSpec("item-provenance", "item_provenance.jsonl", "llb.external-draft-item", "jsonl"),
)

_QUALITY_CHECKS = (
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


def corpus_bundle_manifest(
    corpus_root: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe a staged corpus, including one citation sidecar member per converted PDF."""
    root = Path(corpus_root)
    citations = tuple(
        MemberSpec(f"citations-{path.name}", path.name, "llb.pdf-citations")
        for path in sorted(root.glob(f"*{PDF_CITATION_SUFFIX}"))
    )
    return _manifest(
        root,
        CORPUS_DATASET_ID,
        "One staged corpus: its ingestion manifests, citation sidecars, and applied overlay.",
        (*CORPUS_MEMBERS, *citations),
        registry,
    )


def draft_bundle_manifest(
    bundle_dir: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe a draft bundle: its gold rows, chains, ontology, extraction, and provenance."""
    return _manifest(
        Path(bundle_dir),
        DRAFT_DATASET_ID,
        "One draft bundle: the drafted items and everything that says what produced them.",
        DRAFT_MEMBERS,
        registry,
    )


def read_bundle(
    root: Path | str, manifest: DatasetManifest, registry: ContractRegistry = DEFAULT_REGISTRY
) -> dict[str, tuple[BaseModel, ...]]:
    """Read every described member through its binding, keyed by member id.

    The first member that cannot be read raises; a caller wanting a survey rather than a gate
    catches `ArtifactContractError` around one member at a time.
    """
    return {
        member.member_id: read_bound_member(Path(root), member, registry)
        for member in manifest.members
    }


def _manifest(
    root: Path,
    dataset_id: str,
    description: str,
    specs: tuple[MemberSpec, ...],
    registry: ContractRegistry,
) -> DatasetManifest:
    members = [
        _member(root, spec, registry) for spec in specs if (root / spec.relative_path).is_file()
    ]
    if not members:
        raise FileNotFoundError(f"{root}: no registered data-prep member is present")
    return DatasetManifest(
        schema_id="llb.dataset-manifest",
        schema_version="1.0.0",
        dataset_id=dataset_id,
        description=description,
        owner="loc-lm-bench maintainers",
        members=members,
        quality_checks=list(_QUALITY_CHECKS),
    )


def _member(root: Path, spec: MemberSpec, registry: ContractRegistry) -> DatasetMember:
    path = root / spec.relative_path
    definition = registry.definition(spec.schema_id)
    return DatasetMember(
        member_id=spec.member_id,
        path=spec.relative_path,
        format=spec.artifact_format,
        media_type=ARTIFACT_MEDIA_TYPES[spec.artifact_format],
        granularity=cast(RecordGranularity, ARTIFACT_GRANULARITIES[spec.artifact_format]),
        digest=f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
        record_contract=ContractReference(
            schema_id=spec.schema_id,
            schema_version=_bound_version(path, definition),
        ),
    )


def _bound_version(path: Path, definition: ContractDefinition) -> str:
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


@dataclass(frozen=True)
class MemberReading:
    """What one member of a described bundle turned out to be."""

    member_id: str
    path: str
    schema_id: str
    source_version: str
    current_version: str
    records: int
    refusal: str = ""

    @property
    def needs_upgrade(self) -> bool:
        return not self.refusal and self.source_version != self.current_version


def survey_bundle(
    root: Path | str, manifest: DatasetManifest, registry: ContractRegistry = DEFAULT_REGISTRY
) -> tuple[MemberReading, ...]:
    """Read every member and report what each is, including the ones that refuse.

    A survey, unlike `read_bundle`, does not stop at the first refusal: an operator deciding
    whether a bundle can be handed on wants the whole list, not its first line.
    """
    return tuple(_reading(Path(root), member, registry) for member in manifest.members)


def upgrade_bundle(
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
    schema_id = contract.schema_id if contract is not None else ""
    current = registry.definition(schema_id).current_version if schema_id else ""
    source = contract.schema_version if contract is not None else ""
    try:
        records = read_bound_member(root, member, registry)
    except ArtifactContractError as exc:
        return MemberReading(member.member_id, member.path, schema_id, source, current, 0, str(exc))
    return MemberReading(member.member_id, member.path, schema_id, source, current, len(records))
