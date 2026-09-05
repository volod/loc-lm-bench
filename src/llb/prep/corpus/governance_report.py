"""Ingestion-time report for the governance needed to derive supersession."""

from collections.abc import Sequence

from llb.conflicts.grouping.census import counted
from llb.conflicts.governance.coverage import document_coverage, document_pair_orderability
from llb.conflicts.governance.editions import ORDERING_FIELDS
from llb.core.contracts.common import JsonObject

INGEST_GOVERNANCE_COVERAGE_SCHEMA_VERSION = 1


def ingestion_governance_coverage(items: Sequence[JsonObject]) -> JsonObject:
    """Count governance only on documents admitted to the staged corpus."""
    governance = [
        {field: item.get(field) for field in ORDERING_FIELDS}
        for item in items
        if item.get("status") == "ok"
    ]
    coverage = {
        "schema_version": INGEST_GOVERNANCE_COVERAGE_SCHEMA_VERSION,
        **document_coverage(governance),
        **document_pair_orderability(governance),
    }
    coverage["consequence"] = ingestion_governance_consequence(coverage)
    return coverage


def ingestion_governance_consequence(coverage: JsonObject) -> str:
    """State what the corpus's pair orderability permits, without gating ingestion."""
    orderable = int(coverage.get("orderable_document_pairs") or 0)
    if orderable == 0:
        return (
            "No supersession can ever be derived on this corpus: no document pair has distinct, "
            "comparable `effective_date` or `version` values."
        )
    return (
        "Supersession can be derived only for these orderable document pairs when their claims "
        "contradict."
    )


def format_ingestion_governance_coverage(coverage: JsonObject) -> str:
    """Render the manifest payload as the one-line CLI ingest summary."""
    fields = coverage.get("documents_by_field")
    by_field = fields if isinstance(fields, dict) else {}
    field_counts = ", ".join(
        f"{int(by_field.get(field) or 0)} `{field}`" for field in ORDERING_FIELDS
    )
    documents = counted(int(coverage["documents"]), "document")
    document_pairs = counted(int(coverage["document_pairs"]), "document pair")
    return (
        f"{int(coverage['dated_documents'])} of {documents} with `effective_date` or `version` "
        f"({field_counts}); {int(coverage['orderable_document_pairs'])} of {document_pairs} "
        "orderable by `compare_editions`. "
        f"{coverage['consequence']}"
    )
