"""Deterministically generate and drift-check portable artifact contract exports."""

import hashlib
import json
from pathlib import Path

import yaml

from llb.artifacts.constants import (
    CATALOG_FILE_NAME,
    ODCS_API_VERSION,
    ODCS_CATALOG_FILE_NAME,
    ODCS_SCHEMA_RELATIVE_PATH,
    ODCS_SCHEMA_SHA256,
)
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.registry_validation import validate_registry
from llb.core.contracts.artifact_catalog import (
    ArtifactCatalog,
    CompatibilityDeclaration,
    ContractCatalogEntry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPORT_ROOT = PROJECT_ROOT / "schemas" / "artifacts"


def schema_relative_path(schema_id: str, version: str) -> str:
    return f"{schema_id}/{version}.schema.json"


def generated_exports(registry: ContractRegistry = DEFAULT_REGISTRY) -> dict[str, bytes]:
    validate_registry(registry)
    exports: dict[str, bytes] = {}
    for definition in registry.definitions:
        for version, model in sorted(definition.models.items()):
            relative = schema_relative_path(definition.schema_id, version)
            schema = model.model_json_schema(mode="validation")
            schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
            schema["$id"] = f"urn:llb:artifact-contract:{definition.schema_id}:{version}"
            exports[relative] = _json_bytes(schema)
    catalog = _build_catalog(registry)
    exports[CATALOG_FILE_NAME] = _json_bytes(catalog.model_dump(mode="json"))
    exports[ODCS_CATALOG_FILE_NAME] = yaml.safe_dump(
        _odcs_projection(catalog), sort_keys=False, allow_unicode=False
    ).encode("ascii")
    return exports


def write_exports(root: Path = EXPORT_ROOT) -> tuple[Path, ...]:
    exports = generated_exports()
    written: list[Path] = []
    for relative, content in exports.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written.append(path)
    return tuple(written)


def check_exports(root: Path = EXPORT_ROOT) -> tuple[str, ...]:
    problems = _vendor_schema_problems(root)
    expected = generated_exports()
    for relative, content in expected.items():
        path = root / relative
        if not path.is_file():
            problems.append(f"missing generated export: {path}")
        elif path.read_bytes() != content:
            problems.append(f"generated export drift: {path}")
    expected_paths = {root / relative for relative in expected}
    actual_paths = {
        path for path in root.rglob("*") if path.is_file() and "vendor" not in path.parts
    }
    for unexpected in sorted(actual_paths - expected_paths):
        problems.append(f"unregistered generated export: {unexpected}")
    return tuple(problems)


def _build_catalog(registry: ContractRegistry) -> ArtifactCatalog:
    entries = []
    for definition in registry.definitions:
        declarations = [
            CompatibilityDeclaration(
                from_version=step.from_version,
                to_version=step.to_version,
                kind="migration",
                description=step.description,
            )
            for step in definition.migrations
        ]
        declarations.extend(
            CompatibilityDeclaration(
                from_version=refusal.from_version,
                to_version=definition.current_version,
                kind="refusal",
                description=refusal.description,
            )
            for refusal in definition.refusals
        )
        entries.append(
            ContractCatalogEntry(
                schema_id=definition.schema_id,
                description=definition.description,
                current_version=definition.current_version,
                supported_read_versions=list(definition.supported_versions),
                deprecation_policy=definition.deprecation_policy,
                schema_paths={
                    version: schema_relative_path(definition.schema_id, version)
                    for version in definition.supported_versions
                },
                bindings=list(definition.bindings),
                compatibility=declarations,
                extension_point=definition.extension_point,
            )
        )
    return ArtifactCatalog(
        schema_id="llb.artifact-catalog",
        schema_version="1.0.0",
        odcs_api_version=ODCS_API_VERSION,
        contracts=entries,
    )


def _odcs_projection(catalog: ArtifactCatalog) -> dict[str, object]:
    schema_objects = []
    for entry in catalog.contracts:
        schema_objects.append(
            {
                "id": entry.schema_id.replace(".", "-"),
                "name": entry.schema_id,
                "logicalType": "object",
                "physicalType": "file",
                "physicalName": entry.schema_paths[entry.current_version],
                "description": entry.description,
                "customProperties": [
                    {"property": "schemaVersion", "value": entry.current_version},
                    {
                        "property": "physicalFormats",
                        "value": [binding.format for binding in entry.bindings],
                    },
                ],
                "quality": [
                    {
                        "name": "portableJsonSchemaValidation",
                        "type": "text",
                        "description": "Validate records with the generated JSON Schema before use.",
                    }
                ],
            }
        )
    return {
        "version": catalog.schema_version,
        "kind": "DataContract",
        "apiVersion": ODCS_API_VERSION,
        "id": "llb-artifact-contract-catalog",
        "name": "loc-lm-bench artifact contract catalog",
        "status": "active",
        "domain": "artifact-contracts",
        "description": {
            "purpose": "Publish project-owned record identities, physical bindings, and checks.",
            "usage": "Resolve compatibility before reading or acting on an artifact.",
            "limitations": "Schema validation does not replace domain quality or authorization gates.",
        },
        "team": {
            "id": "llb-maintainers",
            "name": "loc-lm-bench maintainers",
            "description": "Owners of project artifact contract evolution.",
        },
        "schema": schema_objects,
    }


def _vendor_schema_problems(root: Path) -> list[str]:
    path = root / ODCS_SCHEMA_RELATIVE_PATH
    if not path.is_file():
        return [f"missing pinned ODCS validator schema: {path}"]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != ODCS_SCHEMA_SHA256:
        return [f"ODCS validator schema digest mismatch: {path}; observed sha256:{observed}"]
    return []


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")
