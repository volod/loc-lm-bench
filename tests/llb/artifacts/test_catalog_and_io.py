import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from llb.artifacts.constants import ODCS_SCHEMA_RELATIVE_PATH
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.definitions import ContractDefinition
from llb.artifacts.errors import DatasetReadError, InvalidSourceRecordError
from llb.artifacts.generation import EXPORT_ROOT, check_exports, generated_exports, write_exports
from llb.artifacts.io import read_bound_member
from llb.core.contracts.artifact_catalog import ArtifactCatalog
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.artifacts import (
    CompatibilityProbeV1,
    CompatibilityProbeV2,
    ContractReference,
    DatasetManifest,
    DatasetMember,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "samples" / "artifact_contracts"


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _member(path: Path, artifact_format: str) -> DatasetMember:
    media_types = {
        "json": "application/json",
        "jsonl": "application/x-ndjson",
        "yaml": "application/yaml",
        "csv": "text/csv",
        "parquet": "application/vnd.apache.parquet",
    }
    granularity = "document" if artifact_format in {"json", "yaml"} else "row"
    return DatasetMember(
        member_id=artifact_format,
        path=path.name,
        format=artifact_format,
        media_type=media_types[artifact_format],
        granularity=granularity,
        digest=_digest(path),
        record_contract=ContractReference(
            schema_id="llb.artifact-contract.compatibility-probe",
            schema_version="1.0.0",
        ),
    )


def test_committed_dataset_manifest_binds_and_reads_both_versions() -> None:
    """The committed manifest is at version 1: it reads through the registry, then binds."""
    raw = json.loads((FIXTURES / "dataset-manifest.json").read_text(encoding="utf-8"))
    manifest = DEFAULT_REGISTRY.read_current(raw, source="dataset-manifest.json")
    assert isinstance(manifest, DatasetManifest)
    read = [read_bound_member(FIXTURES, member, DEFAULT_REGISTRY) for member in manifest.members]

    assert [records[0].schema_version for records in read] == ["2.0.0", "2.0.0"]


def test_version_one_manifest_binding_an_opaque_member_cannot_reach_the_current_version() -> None:
    """The one thing version 1 could not say: whose format an opaque member is written in."""
    raw = {
        "schema_id": "llb.dataset-manifest",
        "schema_version": "1.0.0",
        "dataset_id": "legacy-store",
        "description": "A version 1 manifest that bound an index it could not attribute.",
        "owner": "loc-lm-bench maintainers",
        "members": [
            {
                "member_id": "vector-index",
                "path": "index.faiss",
                "format": "opaque",
                "media_type": "application/octet-stream",
                "granularity": "opaque",
                "digest": f"sha256:{'0' * 64}",
            }
        ],
        "quality_checks": [
            {"check_id": "member-digest", "kind": "structural", "description": "bytes match"}
        ],
    }

    with pytest.raises(InvalidSourceRecordError) as excinfo:
        DEFAULT_REGISTRY.read_current(raw, source="dataset-manifest.json")

    assert "opaque_binding" in str(excinfo.value)


def test_json_jsonl_yaml_and_csv_bindings_read_through_registry(tmp_path: Path) -> None:
    row = {
        "schema_id": "llb.artifact-contract.compatibility-probe",
        "schema_version": "1.0.0",
        "name": "portable row",
    }
    paths = {
        "json": tmp_path / "row.json",
        "jsonl": tmp_path / "row.jsonl",
        "yaml": tmp_path / "row.yaml",
        "csv": tmp_path / "row.csv",
    }
    paths["json"].write_text(json.dumps(row), encoding="utf-8")
    paths["jsonl"].write_text(json.dumps(row) + "\n", encoding="utf-8")
    paths["yaml"].write_text(yaml.safe_dump(row), encoding="utf-8")
    paths["csv"].write_text(
        "schema_id,schema_version,name\n"
        "llb.artifact-contract.compatibility-probe,1.0.0,portable row\n",
        encoding="utf-8",
    )

    for artifact_format, path in paths.items():
        parsed = read_bound_member(tmp_path, _member(path, artifact_format), DEFAULT_REGISTRY)
        assert parsed[0].model_dump()["label"] == "portable row"


def test_parquet_binding_reads_through_registry(tmp_path: Path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / "row.parquet"
    parquet.write_table(
        pyarrow.table(
            {
                "schema_id": ["llb.artifact-contract.compatibility-probe"],
                "schema_version": ["1.0.0"],
                "name": ["portable row"],
            }
        ),
        path,
    )

    parsed = read_bound_member(tmp_path, _member(path, "parquet"), DEFAULT_REGISTRY)
    assert parsed[0].model_dump()["label"] == "portable row"


def test_digest_and_manifest_binding_mismatches_refuse(tmp_path: Path) -> None:
    path = tmp_path / "row.json"
    path.write_text(
        json.dumps(
            {
                "schema_id": "llb.artifact-contract.compatibility-probe",
                "schema_version": "2.0.0",
                "label": "wrong bound version",
            }
        ),
        encoding="utf-8",
    )
    member = _member(path, "json")
    with pytest.raises(DatasetReadError, match="manifest binds"):
        read_bound_member(tmp_path, member, DEFAULT_REGISTRY)
    bad_digest = member.model_copy(update={"digest": "sha256:" + "0" * 64})
    with pytest.raises(DatasetReadError, match="digest mismatch"):
        read_bound_member(tmp_path, bad_digest, DEFAULT_REGISTRY)


def test_dataset_member_rejects_media_type_that_disagrees_with_format(tmp_path: Path) -> None:
    path = tmp_path / "row.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="json formats require media_type"):
        DatasetMember(
            member_id="row",
            path=path.name,
            format="json",
            media_type="text/csv",
            granularity="document",
            digest=_digest(path),
            record_contract=ContractReference(
                schema_id="llb.artifact-contract.compatibility-probe",
                schema_version="1.0.0",
            ),
        )


def test_optional_missing_dataset_member_reads_as_empty(tmp_path: Path) -> None:
    missing = DatasetMember(
        member_id="optional",
        path="optional.json",
        format="json",
        media_type="application/json",
        granularity="document",
        required=False,
        digest="sha256:" + "0" * 64,
        record_contract=ContractReference(
            schema_id="llb.artifact-contract.compatibility-probe",
            schema_version="1.0.0",
        ),
    )
    assert read_bound_member(tmp_path, missing, DEFAULT_REGISTRY) == ()


def test_generated_catalog_lists_all_format_bindings_and_validates_itself() -> None:
    raw = json.loads(generated_exports()["catalog.json"])
    catalog = ArtifactCatalog.model_validate(raw, strict=True)
    probe = next(
        entry
        for entry in catalog.contracts
        if entry.schema_id == "llb.artifact-contract.compatibility-probe"
    )

    assert {binding.format for binding in probe.bindings} == {
        "json",
        "jsonl",
        "yaml",
        "csv",
        "parquet",
    }
    assert probe.compatibility[0].from_version == "1.0.0"


def test_generation_drift_is_a_failure(tmp_path: Path) -> None:
    write_exports(tmp_path)
    vendor = tmp_path / ODCS_SCHEMA_RELATIVE_PATH
    vendor.parent.mkdir(parents=True)
    shutil.copy2(EXPORT_ROOT / ODCS_SCHEMA_RELATIVE_PATH, vendor)
    assert check_exports(tmp_path) == ()

    schema = tmp_path / "llb.dataset-manifest" / "1.0.0.schema.json"
    schema.write_text(schema.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert any("generated export drift" in problem for problem in check_exports(tmp_path))


def test_old_model_without_compatibility_declaration_cannot_generate_catalog() -> None:
    definition = ContractDefinition(
        schema_id="llb.artifact-contract.compatibility-probe",
        description="Missing declaration test.",
        current_version="2.0.0",
        models={"1.0.0": CompatibilityProbeV1, "2.0.0": CompatibilityProbeV2},
        bindings=(),
        deprecation_policy="Test only.",
    )
    with pytest.raises(ValueError, match="expected one migration path, found 0"):
        generated_exports(ContractRegistry((definition,)))


def test_external_consumer_validates_json_schemas_and_odcs_without_llb_import() -> None:
    script = FIXTURES / "external_validate.py"
    assert "import llb" not in script.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "external JSON Schema and ODCS validation passed" in result.stdout
