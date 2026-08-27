"""What adopting a model generation invalidates: the re-measurement cost, read before the swap.

The register says which generation of a family is current and the upstream currency probe says
whether a newer one exists. Neither says what changing the answer costs. This package does: it
resolves every model identity recorded in the repo's evidence -- the committed run aggregates, the
values registered designs publish out of them, and the baseline tables in the delivered docs -- back
to the family generation it was measured on, and lists the ones a proposed swap would void.

It reports only, for the same reason the currency probe does: adopting a generation is an operator
decision, and this is the half of that decision nobody can see from the board.
"""

from llb.backends.invalidation.identity import ModelIndex, ResolvedModel
from llb.backends.invalidation.report import (
    ADOPTION,
    ROLLBACK,
    UNORDERED,
    InvalidationReport,
    report_invalidation,
    swap_direction,
)
from llb.backends.invalidation.render import render_json, render_text, report_payload
from llb.backends.invalidation.surfaces import (
    BASELINE_TABLES,
    COMMITTED_AGGREGATES,
    EVIDENCE_SURFACES,
    PUBLISHED_VALUES,
    MeasuredRecord,
    SurfaceReading,
    read_evidence,
)

__all__ = [
    "ADOPTION",
    "BASELINE_TABLES",
    "COMMITTED_AGGREGATES",
    "EVIDENCE_SURFACES",
    "PUBLISHED_VALUES",
    "ROLLBACK",
    "UNORDERED",
    "InvalidationReport",
    "MeasuredRecord",
    "ModelIndex",
    "ResolvedModel",
    "SurfaceReading",
    "read_evidence",
    "render_json",
    "render_text",
    "report_invalidation",
    "report_payload",
    "swap_direction",
]
