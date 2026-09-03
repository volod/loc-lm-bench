"""The retrieval comparison and calibration sidecar contract.

Every `llb` retrieval comparison writes a machine-readable sidecar beside its Markdown report:
`comparison.json` for a lane sweep, `calibration.json` for a held-out threshold selection,
`probe.json` for the multi-hop probe, `report.json` for a bake-off, `run_config.json` for the
configuration fingerprint the reading was taken under. They were bare payloads, so a reader could
not tell a graph-fusion sweep from an embedding bake-off without opening it, and could not tell a
sidecar this build understands from one a newer build wrote.

This contract owns the ENVELOPE, not the measurement body. The body's shape is the lane's own --
its rows are named by swept parameters, its slices by the corpus's question types, and a new lever
adds a section -- so freezing it would make every added measurement a contract change. What the
envelope fixes is what a reader needs before it opens anything: the identity, the version, which
command produced it, and which kind of reading it is.
"""

from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject

RETRIEVAL_COMPARISON_SCHEMA_ID = "llb.retrieval-comparison"

SidecarKind: TypeAlias = Literal["comparison", "calibration", "probe", "validation", "run-config"]

SIDECAR_KIND_COMPARISON: SidecarKind = "comparison"
SIDECAR_KIND_CALIBRATION = "calibration"
SIDECAR_KIND_PROBE = "probe"
SIDECAR_KIND_VALIDATION = "validation"
SIDECAR_KIND_RUN_CONFIG = "run-config"
# The producer an archived, pre-envelope sidecar did not record.
UNRECORDED_PRODUCER = "unrecorded"
SIDECAR_KINDS = (
    SIDECAR_KIND_COMPARISON,
    SIDECAR_KIND_CALIBRATION,
    SIDECAR_KIND_PROBE,
    SIDECAR_KIND_VALIDATION,
    SIDECAR_KIND_RUN_CONFIG,
)


class RetrievalComparisonSidecar(ArtifactContract):
    """One machine-readable retrieval-comparison sidecar and what produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.retrieval-comparison"]
    schema_version: Literal["1.0.0"]
    # A sidecar written before the envelope existed says neither what produced it nor which kind
    # of reading it is, so both default to what such a file actually recorded: a comparison, by a
    # command it did not name.
    kind: SidecarKind = SIDECAR_KIND_COMPARISON
    produced_by: str = Field(default=UNRECORDED_PRODUCER, min_length=1)
    report: JsonObject = Field(default_factory=dict)
