"""Per-hop retrievability probe: why does a two-hop item miss `all-spans@k`?

`compare-graph-fusion` measures WHETHER both hops are retrieved; every ranking knob it exposes
leaves that number where it was. This lane measures WHY: it ranks each labeled span twice -- once
by the item's own question at a deep pool, once by the span's own text -- and reads the
`all-spans@k` curve over a budget grid against those ranks. The outcome is a named explanation
(budget, query, or unreachable), because the two lead to opposite fixes.

Entry points: `probe_multihop_hops` diagnoses raw retrieval;
`compare_multihop_query_prep` pairs that diagnosis with one prepared plan per focus item and
counts conversions by the raw cohort. Both paths are pure and fake-store testable.
"""

from llb.rag.multihop_probe.models import (
    DEFAULT_BUDGETS,
    DEFAULT_PROBE_DEPTH,
    DIAGNOSIS_BUDGET,
    DIAGNOSIS_COVERED,
    DIAGNOSIS_QUERY,
    DIAGNOSIS_UNREACHABLE,
    EXPLANATION_BUDGET,
    EXPLANATION_MIXED,
    EXPLANATION_NONE,
    EXPLANATION_QUERY,
    EXPLANATION_UNREACHABLE,
    EvidenceItem,
    MultiHopProbeReport,
    MultiHopQueryPrepReport,
    parse_budgets,
)
from llb.rag.multihop_probe.conversion_report import format_query_prep_probe_report
from llb.rag.multihop_probe.prepared import compare_multihop_query_prep
from llb.rag.multihop_probe.probe import probe_multihop_hops
from llb.rag.multihop_probe.report import format_probe_report

__all__ = [
    "DEFAULT_BUDGETS",
    "DEFAULT_PROBE_DEPTH",
    "DIAGNOSIS_BUDGET",
    "DIAGNOSIS_COVERED",
    "DIAGNOSIS_QUERY",
    "DIAGNOSIS_UNREACHABLE",
    "EXPLANATION_BUDGET",
    "EXPLANATION_MIXED",
    "EXPLANATION_NONE",
    "EXPLANATION_QUERY",
    "EXPLANATION_UNREACHABLE",
    "EvidenceItem",
    "MultiHopProbeReport",
    "MultiHopQueryPrepReport",
    "compare_multihop_query_prep",
    "format_probe_report",
    "format_query_prep_probe_report",
    "parse_budgets",
    "probe_multihop_hops",
]
