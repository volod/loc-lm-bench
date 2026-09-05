"""Verification worksheet contract: the CSV a reviewer decides in.

The worksheet is the one data-prep member whose rows are read by a person as well as by this
project, so its columns are its exchange surface: a sampler that renamed one, or a session that
wrote a decision under a name the accept path does not read, would look like an empty review
rather than a broken one. Every value is a string because CSV carries no types; the closed value
sets are enforced where they are load-bearing and left open where a reviewer types free text.
"""

from collections.abc import Mapping
from typing import Final, Literal

from pydantic import Field, ValidationError

from llb.core.contracts.artifacts import ArtifactContract

CheckValue = Literal["pass", "fail", ""]
DecisionValue = Literal["accept", "reject", ""]
StatusValue = Literal["pending", "decided", ""]


class VerificationWorksheetRow(ArtifactContract):
    """One sampled item as the worksheet carries it: read-only context, then the human columns."""

    schema_id: Literal["llb.verification-worksheet-row"]
    schema_version: Literal["1.0.0"]
    item_kind: str = ""
    item_id: str = Field(min_length=1)
    provenance: str = ""
    split: str = ""
    source_doc_id: str = ""
    synthetic: str = ""
    stratum: str = ""
    question: str = ""
    reference_answer: str = ""
    span_doc_id: str = ""
    span_text: str = ""
    context: str = ""
    retrieval_rank: str = ""
    page_citation: str = ""
    chain_steps: str = ""
    cc_grounded: str = ""
    cc_non_circular: str = ""
    cc_supported: str = ""
    cc_answerable: str = ""
    cc_note: str = ""
    chk_grounded: CheckValue = ""
    chk_answerable: CheckValue = ""
    chk_reference: CheckValue = ""
    chk_planted: CheckValue = ""
    decision: DecisionValue = ""
    reject_code: str = ""
    edited_answer: str = ""
    human_note: str = ""
    human_status: StatusValue = ""
    reviewer_id: str = ""
    # Additive columns: the ambiguous-evidence flag appears only when a sampled span repeats, and
    # the translation profile carries the source text a reviewer compares against.
    span_occurrences: str = ""
    review_profile: str = ""
    source_answer: str = ""
    source_hash: str = ""
    translation_hash: str = ""


WORKSHEET_ROW_SCHEMA_ID: Final[Literal["llb.verification-worksheet-row"]] = (
    "llb.verification-worksheet-row"
)
WORKSHEET_ROW_SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"


def validate_worksheet_row(row: Mapping[str, str], source: object) -> None:
    """Check one worksheet row's registered columns, ignoring the ones a profile adds.

    A profile's own columns are written through unchecked: this contract owns the shared
    verification surface -- the columns every profile's accept path reads -- and claiming more
    would make adding a profile column a contract change.
    """
    known = set(VerificationWorksheetRow.model_fields)
    try:
        VerificationWorksheetRow.model_validate(
            {
                "schema_id": WORKSHEET_ROW_SCHEMA_ID,
                "schema_version": WORKSHEET_ROW_SCHEMA_VERSION,
                **{name: value for name, value in row.items() if name in known},
            }
        )
    except ValidationError as exc:
        raise ValueError(f"{source}: invalid verification worksheet row: {exc}") from exc
