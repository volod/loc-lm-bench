"""Built-in artifact contract declarations."""

from llb.artifacts.definitions import ContractDefinition, MigrationStep
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.artifact_catalog import ArtifactCatalog, FormatBinding
from llb.core.contracts.artifacts import (
    CompatibilityProbeV1,
    CompatibilityProbeV2,
    DatasetManifest,
)


def _probe_v1_to_v2(record: dict[str, object]) -> dict[str, object]:
    return {
        "schema_id": record["schema_id"],
        "schema_version": "2.0.0",
        "label": record["name"],
        "extensions": {},
    }


DOCUMENT_BINDINGS = (
    FormatBinding(format="json", media_type="application/json", granularity="document"),
    FormatBinding(format="yaml", media_type="application/yaml", granularity="document"),
)
ROW_BINDINGS = (
    FormatBinding(format="jsonl", media_type="application/x-ndjson", granularity="row"),
    FormatBinding(format="csv", media_type="text/csv", granularity="row"),
    FormatBinding(format="parquet", media_type="application/vnd.apache.parquet", granularity="row"),
)


def build_default_registry() -> ContractRegistry:
    return ContractRegistry(
        (
            ContractDefinition(
                schema_id="llb.artifact-catalog",
                description="Catalog of artifact schema families readable by this build.",
                current_version="1.0.0",
                models={"1.0.0": ArtifactCatalog},
                bindings=DOCUMENT_BINDINGS,
                deprecation_policy="Catalog version 1 remains readable for this release line.",
            ),
            ContractDefinition(
                schema_id="llb.dataset-manifest",
                description="Physical dataset members bound to logical record contracts.",
                current_version="1.0.0",
                models={"1.0.0": DatasetManifest},
                bindings=DOCUMENT_BINDINGS,
                deprecation_policy="Manifest version 1 remains readable for this release line.",
                extension_point="extensions: scalar values only",
            ),
            ContractDefinition(
                schema_id="llb.artifact-contract.compatibility-probe",
                description="Conformance family proving strict validation and deterministic migration.",
                current_version="2.0.0",
                models={"1.0.0": CompatibilityProbeV1, "2.0.0": CompatibilityProbeV2},
                bindings=(*DOCUMENT_BINDINGS, *ROW_BINDINGS),
                deprecation_policy="Version 1 is read-and-migrate; version 2 is the only write form.",
                migrations=(
                    MigrationStep(
                        from_version="1.0.0",
                        to_version="2.0.0",
                        description="Rename name to label without changing its meaning.",
                        transform=_probe_v1_to_v2,
                    ),
                ),
                extension_point="extensions: scalar values only",
            ),
        )
    )


DEFAULT_REGISTRY = build_default_registry()
