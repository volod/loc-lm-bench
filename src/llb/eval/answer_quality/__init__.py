"""End-to-end answer-quality comparison of two retrieval lanes on the multi-hop slice.

Retrieval coverage is not answer quality: `compare-graph-fusion` measures whether the context
CARRIES every span a multi-hop answer needs, not whether the model then uses both. This lane scores
the identical item set end to end under each retrieval lane and reports the objective per
question-type slice, so a measured coverage gain is either confirmed as an answer-quality gain or
recorded as a retrieval-only effect.
"""

from llb.eval.answer_quality.compare import compare_answer_quality
from llb.eval.answer_quality.lanes import (
    lane_config,
    lane_labels_from_comparison,
    parse_lane_label,
    parse_lanes,
)
from llb.eval.answer_quality.models import FOCUS_SLICE, LaneSpec
from llb.eval.answer_quality.report import format_report
from llb.eval.answer_quality.run import run_answer_quality

__all__ = [
    "FOCUS_SLICE",
    "LaneSpec",
    "compare_answer_quality",
    "format_report",
    "lane_config",
    "lane_labels_from_comparison",
    "parse_lane_label",
    "parse_lanes",
    "run_answer_quality",
]
