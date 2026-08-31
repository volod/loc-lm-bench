"""Offline robotics contract classification, fake replay, and report persistence."""

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from llb.bench.common import new_run_timestamp
from llb.core.contracts.robotics import RoboticsContract
from llb.core.fsutil import atomic_write_text
from llb.core.paths import PROJECT_ROOT, resolve_data_dir
from llb.robotics.digests import file_digest
from llb.robotics.fake_driver import DriverContractError, ProtocolNeutralFakeDriver
from llb.robotics.fixtures import LoadedFixture, contract_schema_digest, load_fixture
from llb.robotics.models import CompatibilityResult, UpstreamPin

LOGGER = logging.getLogger(__name__)
METHOD_NAME = "robotics-contract"


def classify_mhs(pin: UpstreamPin) -> CompatibilityResult:
    """Return only the strongest label justified by inspectable inputs."""
    inspectable = all(
        (
            pin.contract_status == "contract-inspectable",
            pin.revision,
            pin.license,
            pin.license_url,
            pin.normative_reference,
        )
    )
    if not inspectable:
        return CompatibilityResult(
            outcome="protocol-neutral",
            reason="no public normative MHS schema, revision, and license are jointly inspectable",
        )
    if pin.conformance_input is None:
        return CompatibilityResult(
            outcome="contract-inspectable",
            reason="the contract is inspectable but no named conformance input is attached",
        )
    if pin.conformance_input.schema_revision != pin.revision:
        return CompatibilityResult(
            outcome="contract-inspectable",
            reason=(
                f"conformance input {pin.conformance_input.name} targets "
                f"{pin.conformance_input.schema_revision}, not {pin.revision}"
            ),
        )
    if pin.conformance_input.verdict != "pass":
        return CompatibilityResult(
            outcome="contract-inspectable",
            reason=f"conformance input {pin.conformance_input.name} did not pass",
        )
    return CompatibilityResult(
        outcome="MHS-compatible",
        reason=f"named conformance input {pin.conformance_input.name} passed",
    )


def _round_trip(record: BaseModel) -> bool:
    observed = type(record).model_validate_json(record.model_dump_json())
    return observed == record


def _exercise_fake(loaded: LoadedFixture) -> dict[str, str]:
    records = loaded.records
    exercise = loaded.exercise
    proposal = records.action_proposal
    if proposal.proposal_id != exercise.write_proposal_id:
        raise ValueError("fake exercise names a different write proposal")
    if records.gate_decision.decision != "approve":
        raise ValueError("fake write exercise requires a committed approved decision record")

    driver = ProtocolNeutralFakeDriver(records.device_reference, records.device_snapshot)
    if driver.discover() != records.device_snapshot:
        raise ValueError("fake discover result differs from the committed snapshot")
    if driver.reference() != records.device_reference:
        raise ValueError("fake reference result differs from the committed reference")
    read_result = driver.read(exercise.read_operation)
    if read_result != records.device_snapshot.state:
        raise ValueError("fake read result differs from the committed state")
    try:
        driver.validate_limit_probe(exercise.limit_operation, exercise.limit_arguments)
    except DriverContractError:
        pass
    else:
        raise ValueError("fake driver accepted the planted hard-limit violation")
    receipt = driver.write(proposal)
    if receipt != records.action_receipt:
        raise ValueError("fake write receipt differs from the committed receipt")
    return {
        "discover": "pass",
        "read": "pass",
        "write": "pass",
        "reference": "pass",
        "hard_limit_refusal": "pass",
    }


def evaluate_fixture(loaded: LoadedFixture) -> dict[str, object]:
    """Evaluate one already pinned fixture without network or hardware access."""
    sources = {source.id: source for source in loaded.upstreams.sources}
    mhs_result = classify_mhs(sources["mhs"])
    records: tuple[RoboticsContract, ...] = (
        loaded.records.evidence,
        loaded.records.device_reference,
        loaded.records.device_snapshot,
        loaded.records.action_proposal,
        loaded.records.gate_decision,
        loaded.records.action_receipt,
    )
    if not all(_round_trip(record) for record in records):
        raise ValueError("a robotics record failed strict JSON round-trip validation")
    fake_checks = _exercise_fake(loaded)
    return {
        "schema_version": 1,
        "verdict": "pass",
        "compatibility_label": mhs_result.outcome,
        "compatibility_reason": mhs_result.reason,
        "contract_schema_digest": contract_schema_digest(),
        "fixture_manifest_digest": file_digest(loaded.root / "fixture-manifest.json"),
        "validated_boundary_records": len(records),
        "upstreams": [
            {
                "id": source.id,
                "release": source.release,
                "revision": source.revision,
                "contract_status": source.contract_status,
                "license": source.license,
                "transports": list(source.transports),
                "normative_reference": source.normative_reference,
                "conformance_input": (
                    source.conformance_input.name if source.conformance_input else None
                ),
            }
            for source in loaded.upstreams.sources
        ],
        "fake_driver": fake_checks,
    }


def _render_markdown(report: dict[str, object]) -> str:
    upstreams = report["upstreams"]
    assert isinstance(upstreams, list)
    lines = [
        "# Robotics contract compatibility report",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Compatibility label: `{report['compatibility_label']}`",
        f"- Reason: {report['compatibility_reason']}",
        f"- Boundary records validated: {report['validated_boundary_records']}",
        f"- Contract schema digest: `{report['contract_schema_digest']}`",
        f"- Fixture manifest digest: `{report['fixture_manifest_digest']}`",
        "",
        "## Upstream pins",
        "",
        "| Source | Release | Revision | Contract | License |",
        "| --- | --- | --- | --- | --- |",
    ]
    for source in upstreams:
        assert isinstance(source, dict)
        revision = source["revision"] or "--"
        license_name = source["license"] or "--"
        lines.append(
            f"| {source['id']} | {source['release']} | {revision} | "
            f"{source['contract_status']} | {license_name} |"
        )
    lines.extend(["", "## Protocol-neutral fake", ""])
    fake_driver = report["fake_driver"]
    assert isinstance(fake_driver, dict)
    for check, verdict in fake_driver.items():
        lines.append(f"- `{check}`: `{verdict}`")
    return "\n".join(lines) + "\n"


def run_contract_check(
    fixture_dir: Path, data_dir: Path | str | None = None
) -> tuple[Path, dict[str, object]]:
    """Replay the fixture and persist a self-contained report under the data root."""
    loaded = load_fixture(fixture_dir)
    report = evaluate_fixture(loaded)
    run_id, run_timestamp = new_run_timestamp()
    report["run_id"] = run_id
    report["run_timestamp"] = run_timestamp
    try:
        report["fixture"] = str(loaded.root.relative_to(PROJECT_ROOT))
    except ValueError:
        report["fixture"] = str(loaded.root)
    output_dir = resolve_data_dir(data_dir) / METHOD_NAME / run_timestamp
    atomic_write_text(
        output_dir / "report.json",
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(output_dir / "report.md", _render_markdown(report))
    LOGGER.info("robotics contract report written to %s", output_dir)
    return output_dir, report
