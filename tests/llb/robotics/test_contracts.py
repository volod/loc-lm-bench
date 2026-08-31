import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from llb.core.paths import PROJECT_ROOT
from llb.robotics.digests import file_digest
from llb.robotics.fixtures import load_fixture

FIXTURE_ROOT = PROJECT_ROOT / "samples" / "robotics" / "contracts"


@pytest.mark.parametrize(
    "record_name",
    [
        "evidence",
        "device_reference",
        "device_snapshot",
        "action_proposal",
        "gate_decision",
        "action_receipt",
    ],
)
def test_boundary_records_round_trip_and_refuse_unknown_fields(record_name: str) -> None:
    records = load_fixture(FIXTURE_ROOT).records
    record = getattr(records, record_name)
    assert type(record).model_validate_json(record.model_dump_json()) == record

    payload = record.model_dump(mode="json")
    payload["unexpected"] = "must not be discarded"
    with pytest.raises(ValidationError, match="unexpected"):
        type(record).model_validate(payload)


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "contracts"
    shutil.copytree(FIXTURE_ROOT, target)
    return target


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _refresh_upstreams_file_pin(root: Path) -> None:
    manifest_path = root / "fixture-manifest.json"
    manifest = _read_json(manifest_path)
    files = manifest["files"]
    assert isinstance(files, list)
    for item in files:
        assert isinstance(item, dict)
        if item["path"] == "upstreams.json":
            item["sha256"] = file_digest(root / "upstreams.json")
    _write_json(manifest_path, manifest)


def test_contract_schema_change_makes_fixture_stale(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    manifest_path = root / "fixture-manifest.json"
    manifest = _read_json(manifest_path)
    manifest["contract_schema_digest"] = f"sha256:{'0' * 64}"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="stale robotics contract schema"):
        load_fixture(root)


@pytest.mark.parametrize("change", ["release", "reference"])
def test_upstream_pin_change_makes_fixture_stale(tmp_path: Path, change: str) -> None:
    root = _copy_fixture(tmp_path)
    upstream_path = root / "upstreams.json"
    upstreams = _read_json(upstream_path)
    sources = upstreams["sources"]
    assert isinstance(sources, list)
    hflow = sources[0]
    assert isinstance(hflow, dict)
    if change == "release":
        hflow["release"] = "v9.9.9"
    else:
        references = hflow["references"]
        assert isinstance(references, list)
        reference = references[0]
        assert isinstance(reference, dict)
        reference["sha256"] = f"sha256:{'f' * 64}"
    _write_json(upstream_path, upstreams)
    _refresh_upstreams_file_pin(root)

    with pytest.raises(ValueError, match="stale hflow upstream pin"):
        load_fixture(root)
