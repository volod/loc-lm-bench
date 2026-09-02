import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import Field

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.definitions import ContractDefinition, MigrationStep
from llb.artifacts.errors import (
    AmbiguousMigrationError,
    InvalidSourceRecordError,
    MissingIdentityError,
    UnknownContractError,
    UnsupportedFutureVersionError,
)
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.artifact_catalog import FormatBinding
from llb.core.contracts.artifacts import (
    ArtifactContract,
    CompatibilityProbeV1,
    CompatibilityProbeV2,
)

FIXTURES = Path(__file__).resolve().parents[3] / "samples" / "artifact_contracts"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_current_record_resolves_without_migration_and_round_trips() -> None:
    record = _fixture("current.json")
    resolution = DEFAULT_REGISTRY.resolve(record, source="current.json")
    parsed = DEFAULT_REGISTRY.read_current(record, source="current.json")

    assert not resolution.requires_migration
    assert isinstance(parsed, CompatibilityProbeV2)
    assert parsed.model_dump(mode="json") == record


def test_supported_old_record_migrates_to_current_meaning() -> None:
    record = _fixture("supported-old.json")
    resolution = DEFAULT_REGISTRY.resolve(record, source="supported-old.json")
    parsed = DEFAULT_REGISTRY.read_current(record, source="supported-old.json")

    assert resolution.migration_path == ("2.0.0",)
    assert parsed == CompatibilityProbeV2(
        schema_id="llb.artifact-contract.compatibility-probe",
        schema_version="2.0.0",
        label="supported old contract fixture",
        extensions={},
    )


def test_future_major_refuses_before_read() -> None:
    with pytest.raises(UnsupportedFutureVersionError, match=r"unsupported-future.json.*3\.0\.0"):
        DEFAULT_REGISTRY.read_current(
            _fixture("unsupported-future.json"), source="unsupported-future.json"
        )


def test_missing_identity_refusal_names_observed_values() -> None:
    with pytest.raises(MissingIdentityError, match=r"missing-identity.json.*schema_version=None"):
        DEFAULT_REGISTRY.resolve(_fixture("missing-identity.json"), source="missing-identity.json")


def test_unknown_identity_refusal_names_source_version_and_registered_families() -> None:
    record = {"schema_id": "unknown.contract", "schema_version": "1.0.0"}
    with pytest.raises(
        UnknownContractError,
        match=r"unknown.json.*unknown\.contract.*1\.0\.0.*llb\.artifact-catalog",
    ):
        DEFAULT_REGISTRY.resolve(record, source="unknown.json")


def test_invalid_source_is_checked_before_migration() -> None:
    with pytest.raises(
        InvalidSourceRecordError, match=r"invalid-source.json.*source record is invalid"
    ):
        DEFAULT_REGISTRY.read_current(_fixture("invalid-source.json"), source="invalid-source.json")


class CompatibilityProbeV15(ArtifactContract):
    schema_id: Literal["llb.artifact-contract.compatibility-probe"]
    schema_version: Literal["1.5.0"]
    label: str = Field(min_length=1)


def _direct(record: dict[str, object]) -> dict[str, object]:
    return {
        "schema_id": record["schema_id"],
        "schema_version": "2.0.0",
        "label": record["name"],
        "extensions": {},
    }


def _via_intermediate(record: dict[str, object]) -> dict[str, object]:
    return {
        "schema_id": record["schema_id"],
        "schema_version": "1.5.0",
        "label": record["name"],
    }


def _intermediate_to_current(record: dict[str, object]) -> dict[str, object]:
    return {
        "schema_id": record["schema_id"],
        "schema_version": "2.0.0",
        "label": record["label"],
        "extensions": {},
    }


def test_ambiguous_migration_is_a_named_refusal() -> None:
    binding = FormatBinding(format="json", media_type="application/json", granularity="document")
    definition = ContractDefinition(
        schema_id="llb.artifact-contract.compatibility-probe",
        description="Ambiguity test definition.",
        current_version="2.0.0",
        models={
            "1.0.0": CompatibilityProbeV1,
            "1.5.0": CompatibilityProbeV15,
            "2.0.0": CompatibilityProbeV2,
        },
        bindings=(binding,),
        deprecation_policy="Test only.",
        migrations=(
            MigrationStep("1.0.0", "2.0.0", "direct", _direct),
            MigrationStep("1.0.0", "1.5.0", "intermediate", _via_intermediate),
            MigrationStep("1.5.0", "2.0.0", "finish", _intermediate_to_current),
        ),
    )
    registry = ContractRegistry((definition,))

    with pytest.raises(AmbiguousMigrationError, match=r"ambiguous-migration.json.*ambiguous"):
        registry.resolve(_fixture("ambiguous-migration.json"), source="ambiguous-migration.json")


def test_extension_point_is_scalar_only_and_unknown_root_fields_are_forbidden() -> None:
    identity = {
        "schema_id": "llb.artifact-contract.compatibility-probe",
        "schema_version": "2.0.0",
        "label": "strict",
    }
    with pytest.raises(InvalidSourceRecordError):
        DEFAULT_REGISTRY.read_current({**identity, "unknown": True})
    with pytest.raises(InvalidSourceRecordError):
        DEFAULT_REGISTRY.read_current({**identity, "extensions": {"nested": {"no": "objects"}}})
