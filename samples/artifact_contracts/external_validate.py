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
RETRIEVAL_GRAPH_ROOT = FIXTURE_ROOT / "retrieval_graph"
RUN_BUNDLE_ROOT = FIXTURE_ROOT / "run_bundles"


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


def validate_bound_row(instance_path: Path, schema_id: str) -> None:
    """Validate a row member whose identity comes from its dataset binding, not from the row.

    A built store holds hundreds of thousands of chunk rows and a graph holds tens of thousands of
    node rows; none of them repeats an identity per line. The catalog's `current_version` for the
    family is what the store's manifest binds them at, so that is what an external reader stamps.
    """
    catalog = json.loads((SCHEMA_ROOT / "catalog.json").read_text(encoding="utf-8"))
    entry = next(item for item in catalog["contracts"] if item["schema_id"] == schema_id)
    version = entry["current_version"]
    first = next(
        line for line in instance_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    stamped = {"schema_id": schema_id, "schema_version": version, **json.loads(first)}
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


def _validate_retrieval_and_graph() -> None:
    """The store, graph, prompt-system, and comparison surfaces, without importing llb."""
    store = RETRIEVAL_GRAPH_ROOT / "store"
    validate(
        store / "current" / "store_meta.json",
        SCHEMA_ROOT / "llb.rag-store-meta" / "2.0.0.schema.json",
    )
    validate_pre_contract(store / "legacy" / "store_meta.json", "llb.rag-store-meta")
    validate_bound_row(store / "current" / "chunks.jsonl", "llb.rag-chunk")
    validate_bound_row(store / "current" / "parents.jsonl", "llb.rag-chunk")
    validate_pre_contract(store / "legacy" / "chunks.jsonl", "llb.rag-chunk")
    refuse(
        store / "unsupported-future" / "store_meta.json",
        SCHEMA_ROOT / "llb.rag-store-meta" / "2.0.0.schema.json",
    )

    graph = RETRIEVAL_GRAPH_ROOT / "graph" / "current"
    validate(graph / "graph_meta.json", SCHEMA_ROOT / "llb.graph-meta" / "1.0.0.schema.json")
    validate(
        graph / "community_summaries.json",
        SCHEMA_ROOT / "llb.graph-community-summaries" / "1.0.0.schema.json",
    )
    validate_bound_row(graph / "nodes.jsonl", "llb.graph-node")
    validate_bound_row(graph / "edges.jsonl", "llb.graph-edge")
    validate_pre_contract(
        RETRIEVAL_GRAPH_ROOT / "graph" / "legacy" / "nodes.jsonl", "llb.graph-node"
    )

    package = RETRIEVAL_GRAPH_ROOT / "prompt_system" / "current"
    for name, schema_id in (
        ("manifest.json", "llb.prompt-system-manifest"),
        ("anthology.json", "llb.prompt-system-anthology"),
        ("doc_metadata.json", "llb.prompt-system-doc-metadata"),
        ("graph_rag_mapping.json", "llb.prompt-system-mapping"),
        ("candidates.json", "llb.prompt-system-candidates"),
    ):
        validate(package / name, SCHEMA_ROOT / schema_id / "1.0.0.schema.json")

    sidecars = RETRIEVAL_GRAPH_ROOT / "sidecars"
    validate(
        sidecars / "retrieval-comparison.json",
        SCHEMA_ROOT / "llb.retrieval-comparison" / "1.0.0.schema.json",
    )
    validate(
        sidecars / "routing-calibration.json",
        SCHEMA_ROOT / "llb.fusion-routing-calibration" / "1.0.0.schema.json",
    )


def _validate_run_bundles() -> None:
    """The run bundle surface an external consumer validates, without importing llb.

    The manifest is the entry point: a current one carries its identity, a pre-contract one is
    read at the version the catalog publishes for the family, and one from a future major is
    refused. The score and retrieval rows are then validated the way the manifest binds them --
    stamped from the catalog, never from the line -- because a bundle holds one row per case and
    none of them repeats an identity.

    The study record beside them is deliberately not validated here. Its file is the study's own
    local form, and the mapping from that form onto `llb.study-design` is this project's, exactly
    as the conflict bundle's integer version is: an external reader validates what the bundle
    publishes with an identity, and reads the rest through the declaration in the manifest.
    """
    validate(
        RUN_BUNDLE_ROOT / "current" / "manifest.json",
        SCHEMA_ROOT / "llb.run-manifest" / "2.0.0.schema.json",
    )
    validate_pre_contract(RUN_BUNDLE_ROOT / "legacy" / "manifest.json", "llb.run-manifest")
    validate_bound_row(RUN_BUNDLE_ROOT / "current" / "scores.jsonl", "llb.case-score")
    validate_bound_row(RUN_BUNDLE_ROOT / "current" / "retrieval.jsonl", "llb.retrieval-case")
    refuse(
        RUN_BUNDLE_ROOT / "unsupported-future" / "manifest.json",
        SCHEMA_ROOT / "llb.run-manifest" / "2.0.0.schema.json",
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
    _validate_retrieval_and_graph()
    _validate_run_bundles()
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
