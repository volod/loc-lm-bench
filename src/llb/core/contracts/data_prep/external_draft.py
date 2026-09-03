"""Contracts for the sidecars an external-draft import writes.

The operator-supplied `external_provenance.json` is authored by the external service, so it is
deliberately not registered: this project reads its classification declaration and never writes
it. What is registered is what the import itself produces beside the bundle.
"""

from typing import Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.corpus import DataPrepRow


class ImportReportRecord(DataPrepRow):
    """`import_report.json`: what one external-draft import loaded, kept, dropped, and repaired."""

    loaded: int = Field(ge=0)
    kept: int = Field(ge=0)
    counts: dict[str, int] = Field(default_factory=dict)
    dropped: list[dict[str, str]] = Field(default_factory=list)
    repaired: list[dict[str, str]] = Field(default_factory=list)


class ExternalDraftProvenance(ArtifactContract):
    """`provenance.json` of an imported external draft: which service drafted it, under what
    classification, and what the import kept.

    """

    schema_id: Literal["llb.external-draft-provenance"]
    schema_version: Literal["1.0.0"]
    kind: Literal["external-draft-import"]
    provenance: Literal["frontier-drafted"]
    synthetic: bool
    verified: bool
    service: str | None = None
    service_model: str | None = None
    export_date: str | None = None
    data_classification: str | None = None
    operator: str | None = None
    n_items: int = Field(ge=0)
    question_type_distribution: dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    import_report: ImportReportRecord
    needle_retrieval: JsonObject | None = None


class ExternalDraftItemRow(ArtifactContract):
    """One `item_provenance.jsonl` row: the labels an item carries outside the gold schema."""

    schema_id: Literal["llb.external-draft-item"]
    schema_version: Literal["1.0.0"]
    id: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    retrieval_rank: int | None = None
    retrieval_k: int | None = None
