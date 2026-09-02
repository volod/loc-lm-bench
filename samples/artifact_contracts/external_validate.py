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


def _validator_and_instance(instance_path: Path, schema_path: Path):
    if instance_path.suffix in {".yaml", ".yml"}:
        instance = yaml.safe_load(instance_path.read_text(encoding="utf-8"))
    else:
        instance = json.loads(instance_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    return validator_class(schema), instance


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
        SCHEMA_ROOT / "llb.artifact-catalog" / "1.0.0.schema.json",
    )
    validate(
        FIXTURE_ROOT / "dataset-manifest.json",
        SCHEMA_ROOT / "llb.dataset-manifest" / "1.0.0.schema.json",
    )
    validate(
        SCHEMA_ROOT / "catalog.odcs.yaml",
        SCHEMA_ROOT / "vendor" / "odcs-json-schema-v3.1.0.json",
    )
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
