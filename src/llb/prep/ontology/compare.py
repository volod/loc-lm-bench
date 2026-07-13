"""Exact shared-seed local-vs-frontier drafting comparison."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from llb.core.paths import resolve_data_dir
from llb.prep.frontier_telemetry import DraftBudgetExceeded, LLMComplete
from llb.prep.ontology.compare_artifacts import (
    draft_shared_seeds,
    lane_report,
    ranking,
    verification_report,
    write_comparison_report,
)
from llb.prep.ontology.endpoint_config import (
    EndpointCompleters,
    EndpointConfig,
    EndpointLogs,
    EndpointPlan,
)
from llb.prep.ontology.pipeline.bundle import write_budget_abort
from llb.prep.ontology.pipeline.run import draft_goldset

COMPARE_METHOD_DIR = "draft-compare"
LOCAL_LANE = "local"
FRONTIER_LANE = "frontier"


def default_compare_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return resolve_data_dir() / COMPARE_METHOD_DIR / stamp


def compare_drafters(
    corpus_root: Path | str,
    local_endpoint: EndpointConfig,
    frontier_endpoint: EndpointConfig,
    *,
    seeds: int,
    seed: int = 13,
    out_dir: Path | str | None = None,
    local_completers: EndpointCompleters | None = None,
    frontier_complete: LLMComplete | None = None,
    local_verification: Path | None = None,
    frontier_verification: Path | None = None,
) -> dict[str, object]:
    """Draft the same locally extracted seed objects in both lanes and write a ranked report."""
    root = Path(out_dir) if out_dir is not None else default_compare_dir()
    local_dir = root / LOCAL_LANE
    frontier_dir = root / FRONTIER_LANE
    local = draft_goldset(
        corpus_root,
        EndpointPlan.single(local_endpoint),
        completers=local_completers,
        max_items=seeds,
        seed=seed,
        out_dir=local_dir,
    )
    try:
        frontier_logs = EndpointLogs()
        frontier = draft_shared_seeds(
            local,
            frontier_endpoint,
            local_endpoint,
            frontier_dir,
            seed,
            FRONTIER_LANE,
            complete=frontier_complete,
            logs=frontier_logs,
        )
    except DraftBudgetExceeded as exc:
        write_budget_abort(
            frontier_dir,
            EndpointPlan(extraction=local_endpoint, drafting=frontier_endpoint),
            frontier_logs,
            {"comparison_lane": FRONTIER_LANE, "max_items": seeds, "seed": seed},
            exc.reason,
            elapsed_s=0.0,
        )
        raise
    sample_n = min(seeds, max(len(local.items), len(frontier.items)))
    lanes = {
        LOCAL_LANE: lane_report(
            local, verification_report(local_dir, local_verification, sample_n, seed)
        ),
        FRONTIER_LANE: lane_report(
            frontier, verification_report(frontier_dir, frontier_verification, sample_n, seed)
        ),
    }
    return write_comparison_report(
        root,
        corpus_root,
        local,
        lanes,
        kind="frontier-draft-comparison",
        lane_order=[LOCAL_LANE, FRONTIER_LANE],
        execution={
            "mode": "local-then-frontier",
            "model_order": [local_endpoint.model, frontier_endpoint.model],
            "shared_extraction_model": local_endpoint.model,
        },
    )


def refresh_comparison_acceptance(
    report_path: Path | str,
    local_verification: Path,
    frontier_verification: Path,
) -> dict[str, object]:
    """Update only reviewed accept rates and rankings; never call either model lane."""
    path = Path(report_path)
    report = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    lanes = cast(dict[str, dict[str, object]], report["lanes"])
    for name, worksheet in (
        (LOCAL_LANE, local_verification),
        (FRONTIER_LANE, frontier_verification),
    ):
        bundle = Path(str(lanes[name]["bundle"]))
        lanes[name]["verify_sample"] = verification_report(bundle, worksheet, 0, 0)
    rankings = cast(dict[str, list[str]], report["rankings"])
    rankings["accept_rate"] = ranking(lanes, "accept_rate")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
