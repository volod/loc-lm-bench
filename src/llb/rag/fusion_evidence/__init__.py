"""Graph-vector fusion evidence: does graph context recover MULTI-HOP retrieval misses?

`compare-retrieval` answers "which backend has the higher recall@k" over a whole gold set. This
lane answers the narrower question a graph-weight recommendation actually needs: on the items
whose answer requires more than one source span, does fusing the graph lane into the vector lane
retrieve MORE of that evidence, at which graph weight, and does it cost anything elsewhere -- with
uncertainty, because a multi-hop slice is a dozen items, not a thousand.

Entry point: `evaluate_fusion_evidence` (pure, fake-store testable); `build_sweep_rows` assembles
the compared rows for a weight grid; `format_report` renders the Markdown artifact.
"""

from llb.rag.fusion_evidence.models import (
    FOCUS_SLICE,
    EvidenceItem,
    FusionEvidenceReport,
    Verdict,
)
from llb.rag.fusion_evidence.report import format_report
from llb.rag.fusion_evidence.grids import (
    parse_candidates,
    parse_merge_ratios,
    parse_span_identities,
    parse_weights,
)
from llb.rag.fusion_evidence.rows import (
    DEFAULT_GRAPH_CANDIDATES,
    DEFAULT_GRAPH_WEIGHTS,
    DEFAULT_SPAN_IDENTITIES,
    DEFAULT_SPAN_MERGE_RATIOS,
    VECTOR_ROW,
    build_sweep_rows,
)
from llb.rag.fusion_evidence.sweep import evaluate_fusion_evidence

__all__ = [
    "DEFAULT_GRAPH_CANDIDATES",
    "DEFAULT_GRAPH_WEIGHTS",
    "DEFAULT_SPAN_IDENTITIES",
    "DEFAULT_SPAN_MERGE_RATIOS",
    "FOCUS_SLICE",
    "VECTOR_ROW",
    "EvidenceItem",
    "FusionEvidenceReport",
    "Verdict",
    "build_sweep_rows",
    "evaluate_fusion_evidence",
    "format_report",
    "parse_candidates",
    "parse_merge_ratios",
    "parse_span_identities",
    "parse_weights",
]
