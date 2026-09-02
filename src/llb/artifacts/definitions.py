"""Immutable registry declarations for contract versions and evolution edges."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel

from llb.core.contracts.artifact_catalog import FormatBinding

MigrationTransform = Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True)
class MigrationStep:
    from_version: str
    to_version: str
    description: str
    transform: MigrationTransform


@dataclass(frozen=True)
class CompatibilityRefusal:
    from_version: str
    description: str


@dataclass(frozen=True)
class ContractDefinition:
    schema_id: str
    description: str
    current_version: str
    models: Mapping[str, type[BaseModel]]
    bindings: tuple[FormatBinding, ...]
    deprecation_policy: str
    migrations: tuple[MigrationStep, ...] = ()
    refusals: tuple[CompatibilityRefusal, ...] = ()
    extension_point: str | None = None
    schema_path_prefix: str = field(default="schemas/artifacts")

    @property
    def supported_versions(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))
