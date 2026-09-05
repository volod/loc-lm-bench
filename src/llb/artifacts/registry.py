"""Contract dispatch, source validation, and deterministic migration resolution."""

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from llb.artifacts.definitions import ContractDefinition, MigrationStep
from llb.artifacts.errors import (
    AmbiguousMigrationError,
    DeclaredCompatibilityRefusal,
    InvalidSourceRecordError,
    MigrationPathError,
    MissingIdentityError,
    UnknownContractError,
    UnsupportedFutureVersionError,
    UnsupportedVersionError,
)
from llb.artifacts.versioning import SemanticVersion


def _identity(schema_id: str, version: str) -> dict[str, object]:
    return {"schema_id": schema_id, "schema_version": version}


@dataclass(frozen=True)
class CompatibilityResolution:
    schema_id: str
    source_version: str
    current_version: str
    migration_path: tuple[str, ...]

    @property
    def requires_migration(self) -> bool:
        return bool(self.migration_path)


class ContractRegistry:
    def __init__(self, definitions: tuple[ContractDefinition, ...]) -> None:
        self._definitions = {definition.schema_id: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("contract schema ids must be unique")

    @property
    def definitions(self) -> tuple[ContractDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    def definition(self, schema_id: str) -> ContractDefinition:
        try:
            return self._definitions[schema_id]
        except KeyError as exc:
            raise UnknownContractError(
                f"unknown artifact contract schema_id={schema_id!r}"
            ) from exc

    def resolve(
        self, record: Mapping[str, object], *, source: str = "<record>"
    ) -> CompatibilityResolution:
        schema_id, source_version = self._identity(record, source=source)
        try:
            definition = self.definition(schema_id)
        except UnknownContractError as exc:
            registered = ", ".join(sorted(self._definitions))
            raise UnknownContractError(
                f"{source}: schema_id={schema_id!r}, schema_version={source_version!r}, "
                f"registered=[{registered}]"
            ) from exc
        self._check_supported(definition, source_version, source=source)
        refusal = next(
            (item for item in definition.refusals if item.from_version == source_version), None
        )
        if refusal is not None:
            raise DeclaredCompatibilityRefusal(
                self._context(definition, source_version, source)
                + f"; declared refusal: {refusal.description}"
            )
        self._validate_source(definition, source_version, record, source=source)
        if source_version == definition.current_version:
            return CompatibilityResolution(schema_id, source_version, source_version, ())
        paths = self._migration_paths(definition, source_version)
        if not paths:
            raise MigrationPathError(
                self._context(definition, source_version, source)
                + "; no declared migration path to current version"
            )
        if len(paths) > 1:
            rendered = [" -> ".join(step.to_version for step in path) for path in paths]
            raise AmbiguousMigrationError(
                self._context(definition, source_version, source)
                + f"; ambiguous migration paths: {rendered}"
            )
        return CompatibilityResolution(
            schema_id,
            source_version,
            definition.current_version,
            tuple(step.to_version for step in paths[0]),
        )

    def read_as(
        self,
        schema_id: str,
        record: Mapping[str, object],
        *,
        version: str | None = None,
        source: str = "<record>",
    ) -> BaseModel:
        """Read a record whose family the caller already knows, current or pre-contract.

        A record carrying its own identity dispatches exactly as `read_current` does, and one
        naming a different family -- or a version a binding contradicts -- is refused rather than
        coerced. A record carrying NO `schema_id` is a file written before this family joined the
        registry, or one whose physical form encodes its version some other way: it is stamped
        with `version` when a caller supplies one and with the declared `legacy_version`
        otherwise, then migrated forward like any other old record. A caller that does not know
        what it opened still meets the missing-identity refusal through `read_current`.
        """
        definition = self.definition(schema_id)
        observed = record.get("schema_id")
        if observed is None:
            assumed = version or definition.legacy_version
            if assumed is None:
                raise MissingIdentityError(
                    f"{source}: no schema_id and {schema_id!r} declares no legacy read version"
                )
            return self.read_current({**record, **_identity(schema_id, assumed)}, source=source)
        if observed != schema_id:
            raise UnknownContractError(
                f"{source}: expected schema_id={schema_id!r}, observed schema_id={observed!r}"
            )
        if version is not None and record.get("schema_version") != version:
            raise UnsupportedVersionError(
                f"{source}: binding declares schema_version={version!r}, observed "
                f"{record.get('schema_version')!r}"
            )
        return self.read_current(record, source=source)

    def read_current(self, record: Mapping[str, object], *, source: str = "<record>") -> BaseModel:
        resolution = self.resolve(record, source=source)
        definition = self.definition(resolution.schema_id)
        migrated = dict(record)
        for step in self._migration_paths(definition, resolution.source_version)[0]:
            migrated = step.transform(migrated)
            self._validate_source(definition, step.to_version, migrated, source=source)
        model = definition.models[definition.current_version]
        return model.model_validate(migrated, strict=True)

    @staticmethod
    def _identity(record: Mapping[str, object], *, source: str) -> tuple[str, str]:
        schema_id = record.get("schema_id")
        version = record.get("schema_version")
        if not isinstance(schema_id, str) or not isinstance(version, str):
            raise MissingIdentityError(
                f"{source}: missing string schema_id or schema_version; observed "
                f"schema_id={schema_id!r}, schema_version={version!r}"
            )
        return schema_id, version

    @staticmethod
    def _validate_source(
        definition: ContractDefinition,
        version: str,
        record: Mapping[str, object],
        *,
        source: str,
    ) -> None:
        model = definition.models[version]
        try:
            model.model_validate(record, strict=True)
        except ValidationError as exc:
            raise InvalidSourceRecordError(
                ContractRegistry._context(definition, version, source)
                + f"; source record is invalid: {exc}"
            ) from exc

    @staticmethod
    def _check_supported(definition: ContractDefinition, version: str, *, source: str) -> None:
        try:
            observed = SemanticVersion.parse(version)
            current = SemanticVersion.parse(definition.current_version)
        except ValueError as exc:
            raise UnsupportedVersionError(
                ContractRegistry._context(definition, version, source) + f"; {exc}"
            ) from exc
        if version in definition.models:
            return
        error_type = (
            UnsupportedFutureVersionError
            if observed.major > current.major
            else UnsupportedVersionError
        )
        raise error_type(
            ContractRegistry._context(definition, version, source) + "; version is not supported"
        )

    @staticmethod
    def _context(definition: ContractDefinition, version: str, source: str) -> str:
        supported = ", ".join(definition.supported_versions)
        return (
            f"{source}: schema_id={definition.schema_id!r}, schema_version={version!r}, "
            f"supported=[{supported}], current={definition.current_version!r}"
        )

    @staticmethod
    def _migration_paths(
        definition: ContractDefinition, source_version: str
    ) -> list[list[MigrationStep]]:
        if source_version == definition.current_version:
            return [[]]
        by_source: dict[str, list[MigrationStep]] = {}
        for step in definition.migrations:
            by_source.setdefault(step.from_version, []).append(step)
        paths: list[list[MigrationStep]] = []

        def walk(version: str, path: list[MigrationStep], seen: set[str]) -> None:
            for step in by_source.get(version, []):
                if step.to_version in seen:
                    continue
                next_path = [*path, step]
                if step.to_version == definition.current_version:
                    paths.append(next_path)
                else:
                    walk(step.to_version, next_path, {*seen, step.to_version})

        walk(source_version, [], {source_version})
        return paths
