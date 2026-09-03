"""Immutable registry declarations for contract versions and evolution edges.

Almost every family here declares ONE version and no migration: nothing this project writes has
been released, so a second version would describe a form nobody ever wrote, and a field an older
writer simply never recorded is an optional field rather than a version. The exceptions are the
compatibility probe -- which carries two versions on purpose, so the mechanism has a worked example
CI executes -- and the data-prep families whose older forms describe bundles already on disk.

When a version IS finally warranted -- a field changes meaning, is renamed, or a required field
appears that no older writer could have produced -- the recipe is
`docs/impl/current/artifact-contracts/foundation-and-evolution.md`, and
`llb.artifact-contract.compatibility-probe` in `default_registry.py` is the executable template:
the old model keeps its fields and its version `Literal`, the new model takes the plain name, and
one `MigrationStep` names the transform between them.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel

from llb.core.contracts.artifact_catalog import FormatBinding

MigrationTransform = Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True)
class MigrationStep:
    """One declared evolution edge: how a record at `from_version` reaches `to_version`.

    `transform` receives a record that already validated against `from_version` and must return one
    that validates against `to_version` -- the registry re-validates every intermediate result, so a
    transform cannot quietly widen a contract. It must never invent a value: take a default from the
    same constant the reader already applied, or state the absence as `None`.
    """

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

    ``legacy_document_field`` covers the families whose pre-contract file is not a record at all:
    an anthology was a bare array, a community-summary file was the bare summary map, and neither
    has anywhere to put an identity. It names the field of the current record that whole value
    became, so the domain reader, the dataset reader, and an outside reader working from the
    published catalog all wrap it the same way instead of each inventing a shape.
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
    legacy_document_field: str | None = None
    schema_path_prefix: str = field(default="schemas/artifacts")

    @property
    def supported_versions(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))
