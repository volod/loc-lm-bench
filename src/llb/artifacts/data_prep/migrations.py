"""One-step transformations that carry an old data-prep record to its current contract.

Every transform receives a record that already validated against its source version and must
return one that validates against the version it names -- the registry re-validates each result,
so a transform cannot quietly widen a contract. None of them invent a value: an absent field
becomes the explicit absence the current form states, and a re-encoded record is decoded through
the reader that already understood the old form.
"""

from typing import Any

from llb.core.contracts.data_prep.goldset import DEFAULT_LANG, DEFAULT_VERIFIED

ACQUIRED_DOCUMENT_FIELDS = (
    "source_uri",
    "capture_time",
    "capture_id",
    "payload_digest",
    "licence",
    "acquisition_run_id",
    "revision_of",
)


def gold_item_v1_to_v2(record: dict[str, object]) -> dict[str, object]:
    """State the two fields a pre-contract gold row left to its reader's defaults."""
    lang = record.get("lang")
    verified = record.get("verified")
    return {
        **record,
        "schema_version": "2.0.0",
        "lang": lang if isinstance(lang, str) and lang else DEFAULT_LANG,
        "verified": verified if isinstance(verified, bool) else DEFAULT_VERIFIED,
    }


def ontology_provenance_v1_to_v2(record: dict[str, object]) -> dict[str, object]:
    """Add the corpus binding and the per-document acquisition fields as stated absences.

    A bundle drafted before the binding existed names no corpus version and its documents carry no
    upstream capture, which is exactly what `None` means in the current form -- so the reading is
    the same one and the migration adds no claim.
    """
    documents = record.get("documents")
    migrated_documents: list[dict[str, Any]] | None = None
    if isinstance(documents, list):
        migrated_documents = [
            {**row, **{field: row.get(field) for field in ACQUIRED_DOCUMENT_FIELDS}}
            for row in documents
            if isinstance(row, dict)
        ]
    migrated: dict[str, object] = {**record, "schema_version": "2.0.0", "corpus_version": None}
    if migrated_documents is not None:
        migrated["documents"] = migrated_documents
    return migrated


def linkage_settings_v1_to_v2(record: dict[str, object]) -> dict[str, object]:
    """State every tuning knob the old bundle left for `LinkageSpec.from_payload` to default.

    The values come from the same constants that reader applies, so the migrated bundle replays
    identically -- it simply no longer depends on the reading build's defaults to do so.
    """
    from llb.linkage.constants import (
        DEFAULT_DUCKDB_THREADS,
        DEFAULT_EM_CONVERGENCE,
        DEFAULT_EM_MAX_ITERATIONS,
        DEFAULT_MATCH_THRESHOLD,
        DEFAULT_MAX_PAIRS,
        DEFAULT_MIN_LEVEL_PROBABILITY,
        DEFAULT_RANDOM_MATCH_PROBABILITY,
        DEFAULT_SEED,
        DEFAULT_UNIQUE_ID_COLUMN,
    )

    defaults: dict[str, object] = {
        "unique_id_column": DEFAULT_UNIQUE_ID_COLUMN,
        "retain_columns": [],
        "match_threshold": DEFAULT_MATCH_THRESHOLD,
        "max_pairs": DEFAULT_MAX_PAIRS,
        "seed": DEFAULT_SEED,
        "em_max_iterations": DEFAULT_EM_MAX_ITERATIONS,
        "em_convergence": DEFAULT_EM_CONVERGENCE,
        "random_match_probability": DEFAULT_RANDOM_MATCH_PROBABILITY,
        "duckdb_threads": DEFAULT_DUCKDB_THREADS,
        "min_level_probability": DEFAULT_MIN_LEVEL_PROBABILITY,
        "retain_matching_columns": True,
    }
    specification = record.get("specification")
    source = specification if isinstance(specification, dict) else {}
    filled = {
        **{key: value for key, value in source.items() if value is not None},
        **{key: value for key, value in defaults.items() if source.get(key) is None},
    }
    return {**record, "schema_version": "2.0.0", "specification": filled}


def stage_inputs_to_current(record: dict[str, object]) -> dict[str, object]:
    """Re-encode an older conflict stage-inputs record at the current form.

    The decode is the bundle reader's own -- `documents_of` and `recorded_inputs` already
    understand every form the project has written -- so the readings a bundle answers are
    preserved by construction rather than by a second implementation of the same forms.
    """
    from llb.conflicts.bundle.record import (
        SCHEMA_KEY,
        STAGE_INPUTS_SCHEMA_VERSION,
        documents_of,
        recorded_inputs,
        stage_attribution_inputs,
    )

    from llb.artifacts.data_prep.families import contract_version, local_stage_inputs_version

    local = {
        key: value
        for key, value in record.items()
        if key not in ("schema_id", "schema_version", SCHEMA_KEY)
    }
    local[SCHEMA_KEY] = local_stage_inputs_version(str(record["schema_version"]))
    current = stage_attribution_inputs(documents_of(local), recorded_inputs(local))
    current.pop(SCHEMA_KEY, None)
    return {
        "schema_id": record["schema_id"],
        "schema_version": contract_version(STAGE_INPUTS_SCHEMA_VERSION),
        **current,
    }


def catalog_v1_to_v1_1(record: dict[str, object]) -> dict[str, object]:
    """Add the legacy read version each catalog entry now publishes, absent by default."""
    contracts = record.get("contracts")
    entries = contracts if isinstance(contracts, list) else []
    return {
        **record,
        "schema_version": "1.1.0",
        "contracts": [
            {**entry, "legacy_read_version": entry.get("legacy_read_version")}
            for entry in entries
            if isinstance(entry, dict)
        ],
    }
