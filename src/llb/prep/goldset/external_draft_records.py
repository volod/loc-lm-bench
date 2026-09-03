"""The sidecars an external-draft import writes, built through their registered contracts.

Split from `external_draft.py`, which runs the import; this module only shapes what the import
leaves behind -- the per-item label rows and the provenance record that says which service drafted
the bundle, under which data classification, and what the import kept.
"""

import json
from pathlib import Path
from typing import Any

from llb.artifacts.serialization import stated_sections
from llb.core.contracts.data_prep.external_draft import (
    ExternalDraftItemRow,
    ExternalDraftProvenance,
    ImportReportRecord,
)
from llb.goldset.schema import GoldItem
from llb.prep.goldset.external_draft_schema import (
    EXTERNAL_DRAFT_ITEM_SCHEMA_ID,
    EXTERNAL_DRAFT_PROVENANCE_SCHEMA_ID,
    EXTERNAL_DRAFT_SCHEMA_VERSION,
    IMPORT_REPORT_FILENAME,
    ITEM_PROVENANCE_FILENAME,
    ImportReport,
    PROVENANCE_EXTERNAL,
    PROVENANCE_FILENAME,
    _label_distribution,
)
from llb.prep.ontology.models import ItemLabels


def write_item_provenance(
    out_dir: Path,
    items: list[GoldItem],
    item_labels: dict[str, ItemLabels],
    retrieval_ranks: dict[str, int | None] | None,
    retrieval_k: int | None,
) -> None:
    """One label row per item, with the retrieval columns only when an index actually ran."""
    with (out_dir / ITEM_PROVENANCE_FILENAME).open("w", encoding="utf-8") as handle:
        for item in items:
            label = item_labels[item.id]
            row = ExternalDraftItemRow(
                schema_id=EXTERNAL_DRAFT_ITEM_SCHEMA_ID,
                schema_version=EXTERNAL_DRAFT_SCHEMA_VERSION,
                id=item.id,
                question_type=label.question_type,
                difficulty=label.difficulty,
                retrieval_rank=(
                    retrieval_ranks.get(item.id) if retrieval_ranks is not None else None
                ),
                retrieval_k=retrieval_k if retrieval_ranks is not None else None,
            )
            handle.write(json.dumps(_row_payload(row, retrieval_ranks), ensure_ascii=False) + "\n")


def write_import_provenance(
    out_dir: Path,
    items: list[GoldItem],
    item_labels: dict[str, ItemLabels],
    sidecar: dict[str, Any],
    report: ImportReport,
    needle_report: dict[str, Any] | None,
) -> None:
    """The import's own provenance and its report, both at their registered contracts."""
    report_record = ImportReportRecord.model_validate(report.to_dict())
    provenance = ExternalDraftProvenance(
        schema_id=EXTERNAL_DRAFT_PROVENANCE_SCHEMA_ID,
        schema_version=EXTERNAL_DRAFT_SCHEMA_VERSION,
        kind="external-draft-import",
        provenance=PROVENANCE_EXTERNAL,
        synthetic=False,
        verified=False,
        service=_declared(sidecar, "service"),
        service_model=_declared(sidecar, "service_model"),
        export_date=_declared(sidecar, "export_date"),
        data_classification=_declared(sidecar, "data_classification"),
        operator=_declared(sidecar, "operator"),
        n_items=len(items),
        question_type_distribution=_label_distribution(
            [item_labels[item.id].question_type for item in items]
        ),
        difficulty_distribution=_label_distribution(
            [item_labels[item.id].difficulty for item in items]
        ),
        import_report=report_record,
        needle_retrieval=needle_report,
    )
    (out_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(stated_sections(provenance), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / IMPORT_REPORT_FILENAME).write_text(
        json.dumps(report_record.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _row_payload(
    row: ExternalDraftItemRow, retrieval_ranks: dict[str, int | None] | None
) -> dict[str, Any]:
    """A null `retrieval_rank` is a needle searched for and missed.

    The two columns ABSENT is a run that searched for nothing. Collapsing the two would lose the
    difference the drafting report is read for, so the columns are dropped rather than nulled.
    """
    payload: dict[str, Any] = row.model_dump()
    if retrieval_ranks is None:
        payload.pop("retrieval_rank")
        payload.pop("retrieval_k")
    return payload


def _declared(sidecar: dict[str, Any], name: str) -> str | None:
    """One declared sidecar field, or None where the external service recorded none."""
    value = sidecar.get(name)
    return value if isinstance(value, str) else None
