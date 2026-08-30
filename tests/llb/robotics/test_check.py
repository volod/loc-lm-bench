import json

from llb.core.paths import PROJECT_ROOT
from llb.robotics.check import classify_mhs, evaluate_fixture, run_contract_check
from llb.robotics.fixtures import load_fixture
from llb.robotics.models import ConformanceInput

FIXTURE_ROOT = PROJECT_ROOT / "samples" / "robotics" / "contracts"


def _mhs_pin():
    loaded = load_fixture(FIXTURE_ROOT)
    return next(source for source in loaded.upstreams.sources if source.id == "mhs")


def test_committed_fixture_is_protocol_neutral() -> None:
    report = evaluate_fixture(load_fixture(FIXTURE_ROOT))
    assert report["verdict"] == "pass"
    assert report["compatibility_label"] == "protocol-neutral"
    assert report["validated_boundary_records"] == 6
    assert set(report["fake_driver"].values()) == {"pass"}


def test_inspectable_contract_without_conformance_does_not_claim_compatibility() -> None:
    pin = _mhs_pin().model_copy(
        update={
            "revision": "mhs-schema-v1",
            "contract_status": "contract-inspectable",
            "license": "Apache-2.0",
            "license_url": "https://example.invalid/mhs-license",
            "normative_reference": "mhs.schema.json",
        }
    )
    result = classify_mhs(pin)
    assert result.outcome == "contract-inspectable"
    assert "no named conformance input" in result.reason


def test_mhs_label_requires_a_named_passing_conformance_input() -> None:
    pin = _mhs_pin().model_copy(
        update={
            "revision": "mhs-schema-v1",
            "contract_status": "contract-inspectable",
            "license": "Apache-2.0",
            "license_url": "https://example.invalid/mhs-license",
            "normative_reference": "mhs.schema.json",
            "conformance_input": ConformanceInput(
                name="authorized-preview-suite-v1",
                schema_revision="mhs-schema-v1",
                sha256=f"sha256:{'a' * 64}",
                verdict="pass",
            ),
        }
    )
    assert classify_mhs(pin).outcome == "MHS-compatible"


def test_mhs_label_refuses_a_conformance_input_for_another_schema() -> None:
    pin = _mhs_pin().model_copy(
        update={
            "revision": "mhs-schema-v1",
            "contract_status": "contract-inspectable",
            "license": "Apache-2.0",
            "license_url": "https://example.invalid/mhs-license",
            "normative_reference": "mhs.schema.json",
            "conformance_input": ConformanceInput(
                name="authorized-preview-suite-v2",
                schema_revision="mhs-schema-v2",
                sha256=f"sha256:{'b' * 64}",
                verdict="pass",
            ),
        }
    )
    assert classify_mhs(pin).outcome == "contract-inspectable"


def test_run_writes_json_and_markdown_reports(tmp_path) -> None:
    output_dir, report = run_contract_check(FIXTURE_ROOT, tmp_path)
    saved = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    assert saved["run_id"] == report["run_id"]
    assert saved["compatibility_label"] == "protocol-neutral"
    assert "Compatibility label: `protocol-neutral`" in markdown
    assert "MHS-compatible" not in markdown
