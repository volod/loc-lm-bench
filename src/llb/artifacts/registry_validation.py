"""Static invariants for registry definitions before exports are generated."""

from llb.artifacts.definitions import ContractDefinition
from llb.artifacts.registry import ContractRegistry
from llb.artifacts.versioning import SemanticVersion


def validate_registry(registry: ContractRegistry) -> None:
    errors: list[str] = []
    for definition in registry.definitions:
        errors.extend(_definition_errors(definition))
    if errors:
        raise ValueError("invalid artifact contract registry:\n- " + "\n- ".join(errors))


def _definition_errors(definition: ContractDefinition) -> list[str]:
    return [
        *_version_and_model_errors(definition),
        *_evolution_errors(definition),
    ]


def _version_and_model_errors(definition: ContractDefinition) -> list[str]:
    errors: list[str] = []
    try:
        SemanticVersion.parse(definition.current_version)
        for version in definition.models:
            SemanticVersion.parse(version)
    except ValueError as exc:
        errors.append(f"{definition.schema_id}: {exc}")
    if definition.current_version not in definition.models:
        errors.append(f"{definition.schema_id}: current version has no model")
    for version, model in definition.models.items():
        properties = model.model_json_schema().get("properties", {})
        schema_id = properties.get("schema_id", {})
        schema_version = properties.get("schema_version", {})
        if schema_id.get("const") != definition.schema_id:
            errors.append(
                f"{definition.schema_id}@{version}: model schema_id is not a matching const"
            )
        if schema_version.get("const") != version:
            errors.append(
                f"{definition.schema_id}@{version}: model version is not a matching const"
            )
    return errors


def _evolution_errors(definition: ContractDefinition) -> list[str]:
    refusal_versions = {refusal.from_version for refusal in definition.refusals}
    return [
        *_declaration_errors(definition, refusal_versions),
        *_old_version_errors(definition, refusal_versions),
    ]


def _declaration_errors(definition: ContractDefinition, refusal_versions: set[str]) -> list[str]:
    errors: list[str] = []
    if len(refusal_versions) != len(definition.refusals):
        errors.append(f"{definition.schema_id}: compatibility refusal versions must be unique")
    for refusal in definition.refusals:
        if refusal.from_version not in definition.models:
            errors.append(
                f"{definition.schema_id}@{refusal.from_version}: refusal has no source model"
            )
    edges: set[tuple[str, str]] = set()
    for step in definition.migrations:
        edge = (step.from_version, step.to_version)
        if edge in edges:
            errors.append(f"{definition.schema_id}: duplicate migration edge {edge}")
        edges.add(edge)
        if step.from_version not in definition.models or step.to_version not in definition.models:
            errors.append(f"{definition.schema_id}: migration edge {edge} has no version model")
        if not step.description.strip():
            errors.append(f"{definition.schema_id}: migration edge {edge} needs a description")
    return errors


def _old_version_errors(definition: ContractDefinition, refusal_versions: set[str]) -> list[str]:
    errors: list[str] = []
    for version in definition.models:
        if version == definition.current_version:
            continue
        paths = ContractRegistry._migration_paths(definition, version)
        if version in refusal_versions and paths:
            errors.append(f"{definition.schema_id}@{version}: both migration and refusal declared")
        elif version not in refusal_versions and len(paths) != 1:
            errors.append(
                f"{definition.schema_id}@{version}: expected one migration path, found {len(paths)}"
            )
    return errors
