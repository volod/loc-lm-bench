"""Resolve fusion sweep row labels into scored `run-eval` lanes.

The fusion sweep names its rows `vector`, `graph/<strategy>`, and
`fused/<strategy>@<weight>/d<depth>[/i<span-identity>]`; its verdict names the best one. This
module parses exactly those labels back into retrieval knobs, so the answer-quality lane scores THE row the retrieval
sweep recommended instead of a hand-copied approximation of it. A round-trip test pins the parser
to `fused_row_label`, which is the one place the label is formatted.
"""

import json
from pathlib import Path

from llb.core.config import RunConfig
from llb.eval.answer_quality.models import LaneSpec
from llb.rag.fusion_evidence.models import (
    FUSED_ROW_PREFIX,
    GRAPH_ROW_PREFIX,
    IDENTITY_MARKER,
    VECTOR_ROW,
)
from llb.rag.fusion_spans import DEFAULT_SPAN_IDENTITY, resolve_span_identity

BACKEND_VECTOR = "faiss"
BACKEND_GRAPH = "graph"
BACKEND_FUSED = "fused"
DEPTH_MARKER = "/d"


def parse_lane_label(label: str) -> LaneSpec:
    """`vector` | `graph/<s>` | `fused/<s>@<weight>[/d<depth>][/i<identity>]` -> a `LaneSpec`."""
    text = label.strip()
    if text == VECTOR_ROW:
        return LaneSpec(label=text, retrieval_backend=BACKEND_VECTOR)
    if text.startswith(GRAPH_ROW_PREFIX):
        strategy = text[len(GRAPH_ROW_PREFIX) :]
        if not strategy:
            raise ValueError(f"graph lane label needs a strategy, got {label!r}")
        return LaneSpec(text, BACKEND_GRAPH, retrieval_strategy=strategy)
    if not text.startswith(FUSED_ROW_PREFIX):
        raise ValueError(
            f"unknown lane label {label!r}: expected {VECTOR_ROW!r}, "
            f"{GRAPH_ROW_PREFIX}<strategy>, or "
            f"{FUSED_ROW_PREFIX}<strategy>@<weight>[/d<depth>][/i<identity>]"
        )
    body = text[len(FUSED_ROW_PREFIX) :]
    identity = DEFAULT_SPAN_IDENTITY
    if IDENTITY_MARKER in body:
        body, _, identity_token = body.partition(IDENTITY_MARKER)
        try:
            identity = resolve_span_identity(identity_token)
        except ValueError as exc:
            raise ValueError(f"{exc} in lane label {label!r}") from None
    depth: int | None = None
    if DEPTH_MARKER in body:
        body, _, depth_token = body.partition(DEPTH_MARKER)
        depth = _positive_int(depth_token, label, "candidate depth")
    strategy, marker, weight_token = body.partition("@")
    if not marker or not strategy:
        raise ValueError(f"fused lane label needs <strategy>@<weight>, got {label!r}")
    return LaneSpec(
        text,
        BACKEND_FUSED,
        retrieval_strategy=strategy,
        graph_weight=_weight(weight_token, label),
        graph_fusion_candidates=depth,
        graph_fusion_span_identity=identity,
    )


def _positive_int(token: str, label: str, what: str) -> int:
    try:
        value = int(token)
    except ValueError:
        raise ValueError(f"{what} must be an integer in lane label {label!r}") from None
    if value < 1:
        raise ValueError(f"{what} must be at least 1 in lane label {label!r}")
    return value


def _weight(token: str, label: str) -> float:
    try:
        weight = float(token)
    except ValueError:
        raise ValueError(f"graph weight must be a number in lane label {label!r}") from None
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"graph weight must be within [0, 1] in lane label {label!r}")
    return weight


def parse_lanes(spec: str) -> list[LaneSpec]:
    """Parse a comma-separated lane selection, de-duplicated in the order given."""
    labels = [token.strip() for token in spec.split(",") if token.strip()]
    if not labels:
        raise ValueError("no lane parsed from the lane selection")
    return [parse_lane_label(label) for label in dict.fromkeys(labels)]


def lane_labels_from_comparison(path: Path) -> list[str]:
    """The baseline plus the best fused row of a `compare-graph-fusion` `comparison.json`.

    Scoring answers under the row the retrieval sweep actually recommended is the whole point of
    the comparison, so the two lanes are read from its verdict rather than retyped.
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    verdict = report.get("verdict") if isinstance(report, dict) else None
    if not isinstance(verdict, dict):
        raise ValueError(f"{path}: not a compare-graph-fusion comparison (no verdict)")
    baseline = str(verdict.get("baseline") or VECTOR_ROW)
    best = verdict.get("best_lane") or verdict.get("best_row")
    if not best:
        raise ValueError(f"{path}: the sweep verdict names no fused row to score")
    return list(dict.fromkeys([baseline, str(best)]))


def lane_config(config: RunConfig, lane: LaneSpec, *, run_name_prefix: str) -> RunConfig:
    """`config` with this lane's retrieval knobs applied and a lane-identifying run name.

    Built by revalidating an explicit field mapping rather than `with_overrides`, because a lane
    must be able to set `graph_fusion_candidates` back to `None` (each lane asked for exactly
    `top_k`), and `with_overrides` drops `None` by design.
    """
    values = config.model_dump()
    values.update(
        run_name=f"{run_name_prefix}-{lane.label}",
        retrieval_backend=lane.retrieval_backend,
        graph_fusion_candidates=lane.graph_fusion_candidates,
        graph_fusion_span_identity=lane.graph_fusion_span_identity,
    )
    if lane.retrieval_strategy is not None:
        values["retrieval_strategy"] = lane.retrieval_strategy
    if lane.graph_weight is not None:
        values["graph_weight"] = lane.graph_weight
    return RunConfig.model_validate(values)
