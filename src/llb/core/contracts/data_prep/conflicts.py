"""Conflict audit record contracts: the per-document stage-inputs record and the applied overlay.

The stage-inputs record is the one artifact in this project that already carried a version of its
own -- seven of them (`llb.conflicts.bundle.record`), each a real form written by a real run. They
are registered here rather than re-described: the record is a set of deliberately compressed maps
whose entries are a string at one form and an object at another, so the contract fixes the
envelope every form shares and the registered migration normalizes an old body through the
decoders that already read it.
"""

from typing import Final, Literal

from pydantic import Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject

STAGE_INPUTS_SCHEMA_ID: Final[Literal["llb.conflict-stage-inputs"]] = "llb.conflict-stage-inputs"
CONFLICT_OVERLAY_SCHEMA_ID: Final[Literal["llb.conflict-overlay"]] = "llb.conflict-overlay"


class _StageInputsEnvelope(ArtifactContract):
    """What every stage-inputs form carries: the id table plus the three optional recorded parts.

    Absence is read off the KEY, never off an empty value: a run below the semantic tier records
    no chunk accounting at all, while an empty accounting is a store that held nothing.
    """

    schema_id: Literal["llb.conflict-stage-inputs"]
    documents: list[JsonObject | str] = Field(default_factory=list)
    chunks: JsonObject | None = None
    exclusions: JsonObject | None = None
    candidates: JsonObject | None = None
    extra_document_ids: list[str] | None = None


class ConflictStageInputsV1(_StageInputsEnvelope):
    """Ordering fields and per-document chunk accounting, documents named by id."""

    schema_version: Literal["1.0.0"]


class ConflictStageInputsV2(_StageInputsEnvelope):
    """Adds the per-document exclusion reasons and the ranked candidate list."""

    schema_version: Literal["2.0.0"]


class ConflictStageInputsV3(_StageInputsEnvelope):
    """Adds the cap the candidate prefix was written at."""

    schema_version: Literal["3.0.0"]


class ConflictStageInputsV4(_StageInputsEnvelope):
    """Names every document outside `documents` by its corpus position instead of by its id."""

    schema_version: Literal["4.0.0"]


class ConflictStageInputsV5(_StageInputsEnvelope):
    """Drops the label from a document entry that has nothing to label."""

    schema_version: Literal["5.0.0"]


class _FoldedStageInputs(_StageInputsEnvelope):
    """The forms that fold the head and tail every document id shares out of the table."""

    document_id_prefix: str | None = None
    document_id_suffix: str | None = None


class ConflictStageInputsV6(_FoldedStageInputs):
    """Records the shared id head and tail once; entries are stems."""

    schema_version: Literal["6.0.0"]


class ConflictStageInputs(_FoldedStageInputs):
    """Current: a count map records the value most documents share once, under `default`."""

    schema_version: Literal["7.0.0"]


class ConflictOverlay(ArtifactContract):
    """`conflict_overlay.json`: the additive resolution overlay corpus chunking reads.

    The overlay is folded into each document's fingerprint, so its shape is load-bearing for store
    identity: a member that lost `documents` would silently republish a generation that changes
    nothing.
    """

    schema_id: Literal["llb.conflict-overlay"]
    schema_version: Literal["1.0.0"]
    policy: str | None = None
    source_findings_sha256: str | None = None
    documents: dict[str, JsonObject] = Field(default_factory=dict)
