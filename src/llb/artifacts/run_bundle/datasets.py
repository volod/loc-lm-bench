"""Describe one published run bundle as a dataset, and read every member of it.

A run bundle is a SET of files that only mean something together: the manifest says what ran, the
score rows say what each case did, the retrieval sidecar says what each case's context held, and
the additional records say what the method concluded. The manifest is what makes the set
self-describing -- it declares the score-row contract and every additional artifact with the
digest it was published at -- so the description below is READ off the bundle rather than assumed
from filenames, and binding at the declared digest is what makes the survey a tamper check rather
than a restatement of what the bytes hash today.

Members are DISCOVERED: a run that retrieved nothing has no retrieval sidecar and says so by
omission, while a member that is present and unreadable is a refusal.
"""

from pathlib import Path
from typing import cast

from llb.artifacts.bundles import bound_version
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.retrieval_graph.opaque import content_digest
from llb.artifacts.run_bundle.manifests import (
    MANIFEST_FILE,
    RETRIEVAL_FILE,
    SCORES_FILE,
    read_run_manifest,
)
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
from llb.core.contracts.run_bundle.manifest import (
    RUN_MANIFEST_SCHEMA_ID,
    RunManifestDocument,
    ScoreRowsDeclaration,
)
from llb.core.contracts.run_bundle.rows import RETRIEVAL_CASE_SCHEMA_ID

RUN_BUNDLE_DATASET_ID = "llb-run-bundle"
OPAQUE_MEDIA_TYPE = "application/octet-stream"
STUDY_ROWS_FORMAT = "llb-study-cell-rows"

_QUALITY_CHECKS = (
    DatasetQualityCheck(
        check_id="member-contract-dispatch",
        kind="structural",
        description="Every present member resolves to its registered contract's current version.",
    ),
    DatasetQualityCheck(
        check_id="member-digest",
        kind="structural",
        description="Every member's content matches the digest the bundle published it at.",
    ),
    DatasetQualityCheck(
        check_id="declared-artifact",
        kind="structural",
        description="Every additional artifact names a record contract or a human-report reason.",
    ),
)


def run_bundle_manifest(
    run_dir: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe one published run bundle from what its own manifest declares."""
    root = Path(run_dir)
    head = read_run_manifest(root / MANIFEST_FILE, registry)
    members = [
        _structured(MANIFEST_FILE, "manifest", RUN_MANIFEST_SCHEMA_ID, "json", root, registry),
        _score_member(root, head.score_rows),
    ]
    if (root / RETRIEVAL_FILE).is_file():
        members.append(
            _structured(
                RETRIEVAL_FILE, "retrieval", RETRIEVAL_CASE_SCHEMA_ID, "jsonl", root, registry
            )
        )
    members.extend(_artifact_member(root, head))
    return DatasetManifest(
        schema_id="llb.dataset-manifest",
        schema_version="1.1.0",
        dataset_id=RUN_BUNDLE_DATASET_ID,
        description="One run bundle: what ran, what each case did, and what the method concluded.",
        owner="loc-lm-bench maintainers",
        members=members,
        quality_checks=list(_QUALITY_CHECKS),
    )


def _structured(
    relative: str,
    member_id: str,
    schema_id: str,
    artifact_format: ArtifactFormat,
    root: Path,
    registry: ContractRegistry,
) -> DatasetMember:
    definition = registry.definition(schema_id)
    return DatasetMember(
        member_id=member_id,
        path=relative,
        format=artifact_format,
        media_type=ARTIFACT_MEDIA_TYPES[artifact_format],
        granularity=cast(RecordGranularity, ARTIFACT_GRANULARITIES[artifact_format]),
        digest=content_digest(root / relative),
        record_contract=ContractReference(
            schema_id=schema_id, schema_version=bound_version(root / relative, definition)
        ),
    )


def _score_member(root: Path, declaration: ScoreRowsDeclaration | None) -> DatasetMember:
    """The `scores.jsonl` member, bound to whatever the bundle says its rows answer to.

    A study's cell table and a pre-contract bundle's rows are both bound as OPAQUE members here,
    for the same reason the FAISS index is: the dataset manifest can only bind a structured member
    to a record contract, and neither of these has one -- a study owns its columns, and an older
    bundle never said. Naming the owner is the honest binding, and the survey then checks the
    study's rows against the column set the run published.
    """
    contract = declaration.record_contract if declaration is not None else None
    if contract is not None:
        return DatasetMember(
            member_id="scores",
            path=SCORES_FILE,
            format="jsonl",
            media_type=ARTIFACT_MEDIA_TYPES["jsonl"],
            granularity="row",
            digest=content_digest(root / SCORES_FILE),
            record_contract=contract,
        )
    owner = declaration.owner if declaration is not None else None
    return DatasetMember(
        member_id="scores",
        path=SCORES_FILE,
        format="opaque",
        media_type=OPAQUE_MEDIA_TYPE,
        granularity="opaque",
        digest=content_digest(root / SCORES_FILE),
        opaque_binding=OpaqueBinding(
            owner=owner or "unstated",
            format=STUDY_ROWS_FORMAT,
            format_version="1",
            description=(
                f"Cell rows whose columns {owner} chose."
                if owner
                else "Rows published before a bundle declared what they answer to."
            ),
        ),
    )


def _artifact_member(root: Path, head: RunManifestDocument) -> list[DatasetMember]:
    """Each additional artifact, bound at the digest the bundle published it at."""
    members: list[DatasetMember] = []
    for index, artifact in enumerate(head.artifacts, start=1):
        contract = artifact.record_contract
        if contract is not None:
            members.append(
                DatasetMember(
                    member_id=f"artifact-{index}",
                    path=artifact.name,
                    format="json",
                    media_type=ARTIFACT_MEDIA_TYPES["json"],
                    granularity="document",
                    digest=artifact.digest,
                    record_contract=contract,
                )
            )
            continue
        members.append(
            DatasetMember(
                member_id=f"artifact-{index}",
                path=artifact.name,
                format="opaque",
                media_type=OPAQUE_MEDIA_TYPE,
                granularity="opaque",
                digest=artifact.digest,
                opaque_binding=OpaqueBinding(
                    owner="loc-lm-bench maintainers",
                    format="human-report",
                    format_version="1",
                    description=artifact.human_report or "Rendered human report.",
                ),
            )
        )
    return members
