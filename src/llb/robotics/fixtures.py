"""Load and cross-check the committed, network-free robotics contract fixture."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from llb.core.contracts.robotics import (
    ActionProposal,
    ActionReceipt,
    DeviceReference,
    DeviceSnapshot,
    GateDecision,
    RoboticsEvidence,
)
from llb.robotics.digests import file_digest, value_digest
from llb.robotics.models import (
    BoundaryRecords,
    FakeExercise,
    FixtureManifest,
    MhsPublicRecord,
    UpstreamPin,
    UpstreamPins,
)
from llb.robotics.upstreams import HFLOW_RELEASE, HFLOW_REVISION

MANIFEST_NAME = "fixture-manifest.json"
REQUIRED_FILES = frozenset(
    {
        "boundary-records.json",
        "fake-exercise.json",
        "mhs-public-semantics.json",
        "upstreams.json",
    }
)

_SCHEMA_MODELS = (
    ActionProposal,
    ActionReceipt,
    BoundaryRecords,
    DeviceReference,
    DeviceSnapshot,
    FakeExercise,
    FixtureManifest,
    GateDecision,
    MhsPublicRecord,
    RoboticsEvidence,
    UpstreamPin,
    UpstreamPins,
)
_M = TypeVar("_M", bound=BaseModel)


@dataclass(frozen=True)
class LoadedFixture:
    root: Path
    manifest: FixtureManifest
    upstreams: UpstreamPins
    records: BoundaryRecords
    exercise: FakeExercise
    mhs_public: MhsPublicRecord


def contract_schema_digest() -> str:
    schemas = {model.__name__: model.model_json_schema() for model in _SCHEMA_MODELS}
    return value_digest(schemas)


def _load_model(path: Path, model: type[_M]) -> _M:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ValueError(f"{path}: invalid robotics fixture -- {exc}") from None


def _fixture_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"fixture path must stay below its root: {relative}")
    return root / path


def _source(upstreams: UpstreamPins, source_id: str) -> UpstreamPin:
    matches = [source for source in upstreams.sources if source.id == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {source_id} upstream pin")
    return matches[0]


def _validate_files(root: Path, manifest: FixtureManifest) -> None:
    paths = [item.path for item in manifest.files]
    if len(paths) != len(set(paths)):
        raise ValueError("fixture manifest contains duplicate file paths")
    if set(paths) != REQUIRED_FILES:
        raise ValueError(
            f"fixture manifest files must be exactly {sorted(REQUIRED_FILES)}, got {sorted(paths)}"
        )
    for item in manifest.files:
        observed = file_digest(_fixture_path(root, item.path))
        if observed != item.sha256:
            raise ValueError(
                f"stale fixture file {item.path}: expected {item.sha256}, observed {observed}"
            )


def _validate_upstreams(loaded: LoadedFixture) -> None:
    expectations = {item.id: item.pin_digest for item in loaded.manifest.upstream_pins}
    if set(expectations) != {"hflow", "mhs"}:
        raise ValueError("fixture manifest must pin exactly hflow and mhs")
    for source in loaded.upstreams.sources:
        observed = value_digest(source.model_dump(mode="json"))
        if expectations.get(source.id) != observed:
            raise ValueError(
                f"stale {source.id} upstream pin: expected {expectations.get(source.id)}, "
                f"observed {observed}"
            )

    hflow = _source(loaded.upstreams, "hflow")
    if (
        hflow.contract_status != "contract-inspectable"
        or hflow.release != HFLOW_RELEASE
        or hflow.revision != HFLOW_REVISION
        or hflow.license is None
        or hflow.normative_reference is None
    ):
        raise ValueError("HFlow pin must name an inspectable revision, license, and contract")

    mhs = _source(loaded.upstreams, "mhs")
    required_semantics = {"discover", "read", "write", "reference", "limits"}
    if set(mhs.semantics) != required_semantics:
        raise ValueError("MHS public-semantics pin is incomplete")
    public_digest = file_digest(loaded.root / "mhs-public-semantics.json")
    if not any(reference.sha256 == public_digest for reference in mhs.references):
        raise ValueError("MHS pin does not bind the normalized public-semantics fixture")
    if loaded.mhs_public.publication != mhs.release:
        raise ValueError("MHS publication identifier differs from its upstream pin")
    if (
        mhs.contract_status != "public-semantics-only"
        or mhs.revision is not None
        or mhs.license is not None
        or mhs.license_url is not None
        or mhs.normative_reference is not None
        or mhs.conformance_input is not None
    ):
        raise ValueError("MHS pin claims more than the normalized public-semantics record")


def _record_digest(record: BaseModel, field: str) -> str:
    return value_digest(record.model_dump(mode="json", exclude={field}))


def _validate_unique_names(reference: DeviceReference, snapshot: DeviceSnapshot) -> None:
    operation_names = [operation.name for operation in reference.operations]
    if len(operation_names) != len(set(operation_names)):
        raise ValueError("device reference contains duplicate operation names")
    state_names = [item.name for item in snapshot.state]
    if len(state_names) != len(set(state_names)):
        raise ValueError("device snapshot contains duplicate state names")


def _validate_records(loaded: LoadedFixture) -> None:
    records = loaded.records
    reference = records.device_reference
    snapshot = records.device_snapshot
    proposal = records.action_proposal
    decision = records.gate_decision
    receipt = records.action_receipt

    _validate_unique_names(reference, snapshot)

    if reference.reference_digest != _record_digest(reference, "reference_digest"):
        raise ValueError("device reference digest does not match its contract")
    if proposal.proposal_digest != _record_digest(proposal, "proposal_digest"):
        raise ValueError("action proposal digest does not match its contract")
    if snapshot.reference_digest != reference.reference_digest:
        raise ValueError("device snapshot is not bound to the device reference")
    if snapshot.operations != reference.operations:
        raise ValueError("device snapshot operations differ from the device reference")
    if snapshot.device_id != proposal.device_id:
        raise ValueError("proposal targets a different device")
    if snapshot.state_revision != proposal.expected_state_revision:
        raise ValueError("proposal expects a different state revision")
    if records.evidence.evidence_id not in proposal.evidence_ids:
        raise ValueError("proposal does not name the fixture evidence")
    if decision.proposal_id != proposal.proposal_id:
        raise ValueError("gate decision names a different proposal")
    if decision.proposal_digest != proposal.proposal_digest:
        raise ValueError("gate decision is not bound to the proposal digest")
    if decision.snapshot_id != snapshot.snapshot_id:
        raise ValueError("gate decision names a different snapshot")
    if receipt.proposal_id != proposal.proposal_id:
        raise ValueError("action receipt names a different proposal")
    if receipt.proposal_digest != proposal.proposal_digest:
        raise ValueError("action receipt is not bound to the proposal digest")


def load_fixture(root: Path) -> LoadedFixture:
    """Load a fixture only after every byte, schema, pin, and cross-record link agrees."""
    root = Path(root).resolve()
    manifest = _load_model(root / MANIFEST_NAME, FixtureManifest)
    observed_schema = contract_schema_digest()
    if manifest.contract_schema_digest != observed_schema:
        raise ValueError(
            "stale robotics contract schema: "
            f"expected {manifest.contract_schema_digest}, observed {observed_schema}"
        )
    _validate_files(root, manifest)
    loaded = LoadedFixture(
        root=root,
        manifest=manifest,
        upstreams=_load_model(root / "upstreams.json", UpstreamPins),
        records=_load_model(root / "boundary-records.json", BoundaryRecords),
        exercise=_load_model(root / "fake-exercise.json", FakeExercise),
        mhs_public=_load_model(root / "mhs-public-semantics.json", MhsPublicRecord),
    )
    _validate_upstreams(loaded)
    _validate_records(loaded)
    return loaded
