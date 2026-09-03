"""Validate public contract exports without importing the llb package."""

import json
import logging
from pathlib import Path
import sys

import yaml
from jsonschema import ValidationError
from jsonschema.validators import validator_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = PROJECT_ROOT / "schemas" / "artifacts"
FIXTURE_ROOT = PROJECT_ROOT / "samples" / "artifact_contracts"
DATA_PREP_ROOT = FIXTURE_ROOT / "data_prep"


def validate(instance_path: Path, schema_path: Path) -> None:
    validator, instance = _validator_and_instance(instance_path, schema_path)
    validator.validate(instance)


def refuse(instance_path: Path, schema_path: Path) -> None:
    validator, instance = _validator_and_instance(instance_path, schema_path)
    try:
        validator.validate(instance)
    except ValidationError:
        return
    raise AssertionError(f"{instance_path} unexpectedly passed {schema_path}")


def validate_row(instance_path: Path, schema_path: Path) -> None:
    """Validate the first record of a JSONL member, which is how a row contract is bound."""
    row = json.loads(next(
        line for line in instance_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ))
    _validator(schema_path).validate(row)


def validate_pre_contract(instance_path: Path, schema_id: str) -> None:
    """Validate a file written before the registry existed, the way an external reader must.

    Such a file carries no identity of its own; the catalog's `legacy_read_version` is what tells
    a reader which version to assume, so this stamps exactly that and nothing else.
    """
    catalog = json.loads((SCHEMA_ROOT / "catalog.json").read_text(encoding="utf-8"))
    entry = next(item for item in catalog["contracts"] if item["schema_id"] == schema_id)
    version = entry["legacy_read_version"]
    text = instance_path.read_text(encoding="utf-8")
    first = next((line for line in text.splitlines() if line.strip()), text)
    record = json.loads(first if instance_path.suffix == ".jsonl" else text)
    stamped = {"schema_id": schema_id, "schema_version": version, **record}
    _validator(SCHEMA_ROOT / entry["schema_paths"][version]).validate(stamped)


def _validator(schema_path: Path):
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    return validator_class(schema)


def _validator_and_instance(instance_path: Path, schema_path: Path):
    if instance_path.suffix in {".yaml", ".yml"}:
        instance = yaml.safe_load(instance_path.read_text(encoding="utf-8"))
    else:
        instance = json.loads(instance_path.read_text(encoding="utf-8"))
    return _validator(schema_path), instance


def _validate_data_prep() -> None:
    """The data-prep exchange surface, current and pre-contract, without importing llb."""
    validate(
        DATA_PREP_ROOT / "corpus" / "corpus_manifest.json",
        SCHEMA_ROOT / "llb.corpus-manifest" / "1.0.0.schema.json",
    )
    validate(
        DATA_PREP_ROOT / "draft-bundle" / "provenance.json",
        SCHEMA_ROOT / "llb.ontology-provenance" / "2.0.0.schema.json",
    )
    validate(
        DATA_PREP_ROOT / "current" / "linkage-settings.json",
        SCHEMA_ROOT / "llb.linkage-settings" / "2.0.0.schema.json",
    )
    validate_row(
        DATA_PREP_ROOT / "draft-bundle" / "goldset.jsonl",
        SCHEMA_ROOT / "llb.gold-item" / "2.0.0.schema.json",
    )
    validate_row(
        DATA_PREP_ROOT / "draft-bundle" / "chains.jsonl",
        SCHEMA_ROOT / "llb.gold-chain" / "1.0.0.schema.json",
    )
    validate_pre_contract(DATA_PREP_ROOT / "legacy" / "goldset.jsonl", "llb.gold-item")
    validate_pre_contract(DATA_PREP_ROOT / "legacy" / "provenance.json", "llb.ontology-provenance")
    validate_pre_contract(
        DATA_PREP_ROOT / "legacy" / "linkage-settings.json", "llb.linkage-settings"
    )
    refuse(
        DATA_PREP_ROOT / "unsupported-future" / "corpus_manifest.json",
        SCHEMA_ROOT / "llb.corpus-manifest" / "1.0.0.schema.json",
    )


def main() -> None:
    validate(
        FIXTURE_ROOT / "current.json",
        SCHEMA_ROOT / "llb.artifact-contract.compatibility-probe" / "2.0.0.schema.json",
    )
    validate(
        FIXTURE_ROOT / "supported-old.json",
        SCHEMA_ROOT / "llb.artifact-contract.compatibility-probe" / "1.0.0.schema.json",
    )
    validate(
        SCHEMA_ROOT / "catalog.json",
        SCHEMA_ROOT / "llb.artifact-catalog" / "1.1.0.schema.json",
    )
    validate(
        FIXTURE_ROOT / "dataset-manifest.json",
        SCHEMA_ROOT / "llb.dataset-manifest" / "1.0.0.schema.json",
    )
    validate(
        SCHEMA_ROOT / "catalog.odcs.yaml",
        SCHEMA_ROOT / "vendor" / "odcs-json-schema-v3.1.0.json",
    )
    _validate_data_prep()
    refuse(
        FIXTURE_ROOT / "missing-identity.json",
        SCHEMA_ROOT / "llb.artifact-contract.compatibility-probe" / "2.0.0.schema.json",
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger(__name__).info(
        "[artifact-contracts] external JSON Schema and ODCS validation passed"
    )


if __name__ == "__main__":
    main()
