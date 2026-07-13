"""Explain a finalized run's wrong answers (miss analysis).

After any run or sweep, `llb analyze-misses` classifies every miss of one run bundle into
exactly one class -- retrieval miss (gold span absent from the retrieved context), generation
miss (evidence present, answer wrong), refusal, format/scoring artifact, or judge disagreement
-- clusters the misses by document, topic, and question type, and emits ranked, evidence-backed
recommendations (raise or lower `top_k`, change chunking, add prompt-system dictionary terms,
try the named alternative model). Every recommendation line names its numeric evidence.

Classification is span-overlap based: it reads the additive per-case `retrieval.jsonl` record
the runner persists beside `scores.jsonl` (falling back to the scored `retrieval_hit` for
legacy bundles). Everything here is pure and file-driven -- no endpoint, GPU, or store -- so the
whole classifier is unit-testable over a synthetic scored bundle. The bounded probe mode that
re-runs the miss subset at alternative retrieval depths lives in `miss_probe.py`; run bundles
are never mutated.

The implementation is split into `model` (vocabulary + dataclasses), `load` (bundle reading),
`classify` (classification + clustering + `analyze_run`), `recommendations` (ranked advice), and
`report` (Markdown + JSON artifacts); the public API is re-exported here so callers keep importing
`llb.board.miss_analysis`.
"""

from llb.board.miss_analysis.classify import (
    analyze_run,
    classify_case,
    question_type_of,
    retrieval_hit_from_record,
    topic_of,
)
from llb.board.miss_analysis.load import (
    load_item_provenance,
    load_scored_bundle,
)
from llb.board.miss_analysis.model import (
    ANALYSIS_FILENAME,
    ARTIFACT_STATUSES,
    CLUSTER_DIMENSIONS,
    DEFAULT_MISS_THRESHOLD,
    DEFAULT_QUESTION_TYPE,
    DICTIONARY_CLUSTER_MIN,
    DICTIONARY_CLUSTER_SHARE,
    ITEM_PROVENANCE_FILENAME,
    JUDGE_AGREEMENT_MIN,
    MISS_ANALYSIS_METHOD,
    MISS_ARTIFACT,
    MISS_CLASSES,
    MISS_GENERATION,
    MISS_JUDGE,
    MISS_REFUSAL,
    MISS_RETRIEVAL,
    MISSES_FILENAME,
    PROBE_CONFIRM_MIN,
    PROBE_MIN_OBJECTIVE_GAIN,
    RAG_CONFIG_KEYS,
    REPORT_FILENAME,
    RETRIEVAL_FILENAME,
    ClusterRow,
    MissAnalysis,
    MissRecord,
)
from llb.board.miss_analysis.recommendations import (
    build_recommendations,
    refresh_recommendations,
)
from llb.board.miss_analysis.report import (
    analysis_out_dir,
    analysis_payload,
    format_report_md,
    latest_analysis,
    write_analysis,
)

__all__ = [
    "ANALYSIS_FILENAME",
    "ARTIFACT_STATUSES",
    "CLUSTER_DIMENSIONS",
    "DEFAULT_MISS_THRESHOLD",
    "DEFAULT_QUESTION_TYPE",
    "DICTIONARY_CLUSTER_MIN",
    "DICTIONARY_CLUSTER_SHARE",
    "ITEM_PROVENANCE_FILENAME",
    "JUDGE_AGREEMENT_MIN",
    "MISSES_FILENAME",
    "MISS_ANALYSIS_METHOD",
    "MISS_ARTIFACT",
    "MISS_CLASSES",
    "MISS_GENERATION",
    "MISS_JUDGE",
    "MISS_REFUSAL",
    "MISS_RETRIEVAL",
    "PROBE_CONFIRM_MIN",
    "PROBE_MIN_OBJECTIVE_GAIN",
    "RAG_CONFIG_KEYS",
    "REPORT_FILENAME",
    "RETRIEVAL_FILENAME",
    "ClusterRow",
    "MissAnalysis",
    "MissRecord",
    "analysis_out_dir",
    "analysis_payload",
    "analyze_run",
    "build_recommendations",
    "classify_case",
    "format_report_md",
    "latest_analysis",
    "load_item_provenance",
    "load_scored_bundle",
    "question_type_of",
    "refresh_recommendations",
    "retrieval_hit_from_record",
    "topic_of",
    "write_analysis",
]
