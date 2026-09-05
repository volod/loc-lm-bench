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
    """One family: its versions, how they evolve, and how a pre-contract file is read.

    ``legacy_version`` names the version a reader assumes when a caller reads a file AS this
    family and the file carries no identity at all. Every artifact this project wrote before the
    registry existed is such a file, so a family without the declaration simply cannot read its
    own history; one with it stamps that version and migrates forward like any other old record.
    """

    schema_id: str
    description: str
    current_version: str
    models: Mapping[str, type[BaseModel]]
    bindings: tuple[FormatBinding, ...]
    deprecation_policy: str
    migrations: tuple[MigrationStep, ...] = ()
    refusals: tuple[CompatibilityRefusal, ...] = ()
    extension_point: str | None = None
    legacy_version: str | None = None
    schema_path_prefix: str = field(default="schemas/artifacts")

    @property
    def supported_versions(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))
