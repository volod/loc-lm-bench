"""Read and write a study record without moving the bytes a citation resolves against.

A study design and its analysis are written in the study's own local form -- the JSON object the
study composed, and nothing wrapped around it. That is deliberate: a published aggregate is cited
by digest, so re-encoding an archived record to carry an identity would break every citation that
already points at it, and a design edited after the run is not a design.

So identity is STAMPED rather than stored, the same way `llb.conflict-stage-inputs` maps its
integer version in [data-prep contracts]. The producer builds the contract model from the local
form and validates it before publication; a reader rebuilds the same model from the file plus the
study id its bundle declared.
"""

import json
from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import DatasetReadError, MissingIdentityError
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.run_bundle.studies import (
    STUDY_ANALYSIS_SCHEMA_ID,
    STUDY_DESIGN_SCHEMA_ID,
    LOCAL_VERSION_KEY,
    StudyRecord,
)

STUDY_SCHEMA_IDS = (STUDY_DESIGN_SCHEMA_ID, STUDY_ANALYSIS_SCHEMA_ID)
# The keys a design states about itself. Everything else in the local form is the study's body,
# including these when the record is an analysis, which states none of them.
_IDENTITY_KEYS = ("study_id", "study_kind")


def study_record(
    payload: object,
    schema_id: str,
    *,
    study_id: str | None = None,
    registry: ContractRegistry = DEFAULT_REGISTRY,
    source: str = "<record>",
) -> StudyRecord:
    """The contract model for one study record written in its local form.

    A design names the study it belongs to and states its own integer version, so both are read
    off the record. An analysis states neither -- it is written beside the design that produced it
    -- so the caller supplies the study id, which is the bundle's declaration of what it published.
    """
    study_id = study_id or None
    if isinstance(payload, dict):
        declared = payload.get("study_id")
        stamped = {
            "study_id": declared if isinstance(declared, str) else study_id,
            "study_kind": payload.get("study_kind"),
            "local_version": payload.get(LOCAL_VERSION_KEY),
            "body": payload,
        }
    else:
        stamped = {"study_id": study_id, "study_kind": None, "local_version": None, "body": payload}
    if stamped["study_id"] is None:
        raise MissingIdentityError(
            f"{source}: study record states no study_id and none was declared for it"
        )
    read = registry.read_as(schema_id, stamped, source=source)
    assert isinstance(read, StudyRecord)
    return read


def local_form(record: StudyRecord) -> object:
    """The bytes a study record is written as: its body, exactly as the study composed it."""
    return record.body


def read_study_record(
    path: Path,
    schema_id: str,
    *,
    study_id: str | None = None,
    registry: ContractRegistry = DEFAULT_REGISTRY,
) -> StudyRecord:
    """Read one published study record at the current contract version."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetReadError(f"{path}: cannot read study record: {exc}") from exc
    return study_record(payload, schema_id, study_id=study_id, registry=registry, source=str(path))
