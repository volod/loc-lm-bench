"""Shared-seed comparison bundle, verification, ranking, and report helpers."""

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import cast

from llb.goldset.schema import Split
from llb.goldset.splits import assign_splits
from llb.goldset.verify import build_sample_worksheet
from llb.goldset.verify_acceptance import acceptance_report
from llb.goldset.verify_base import load_worksheet
from llb.prep.frontier_telemetry import LLMComplete
from llb.prep.ontology.draft import draft_items
from llb.prep.ontology.endpoint import build_complete
from llb.prep.ontology.endpoint_config import EndpointConfig, EndpointLogs, EndpointPlan
from llb.prep.ontology.induce import ontology_constraints
from llb.prep.ontology.pipeline.bundle import _write_bundle
from llb.prep.ontology.pipeline.settings import PipelineResult
from llb.prep.ontology.refine import refine_drafts_labeled

COMPARE_REPORT_FILENAME = "comparison.json"


def seed_fingerprint(seed: object) -> str:
    payload = seed.model_dump_json()  # type: ignore[attr-defined]
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_item_splits(result: PipelineResult, seed: int) -> None:
    splits = assign_splits([item.id for item in result.items], seed=seed)
    for item in result.items:
        item.split = cast(Split, splits[item.id])


def draft_shared_seeds(
    source: PipelineResult,
    drafting_endpoint: EndpointConfig,
    extraction_endpoint: EndpointConfig,
    out_dir: Path,
    seed: int,
    lane: str,
    *,
    complete: LLMComplete | None = None,
    logs: EndpointLogs | None = None,
) -> PipelineResult:
    """Draft a source lane's exact seed objects through one other endpoint."""
    started = perf_counter()
    active_logs = logs or EndpointLogs()
    drafting_complete = complete or build_complete(drafting_endpoint, active_logs.drafting)
    raw = draft_items(
        drafting_complete,
        source.docs,
        source.seeds,
        ontology_constraints(source.ontology),
    )
    items, labels = refine_drafts_labeled(source.docs, raw)
    result = PipelineResult(
        out_dir=out_dir,
        docs=source.docs,
        extractions=source.extractions,
        ontology=source.ontology,
        seeds=source.seeds,
        items=items,
        corpus_root=source.corpus_root,
        elapsed_s=perf_counter() - started,
        draft_attempts=len(source.seeds),
        draft_parsed=len(raw),
        item_labels=labels,
        coverage_report=source.coverage_report,
        endpoint_logs=active_logs,
    )
    assign_item_splits(result, seed)
    _write_bundle(
        result,
        EndpointPlan(extraction=extraction_endpoint, drafting=drafting_endpoint),
        seed,
        {
            "comparison_lane": lane,
            "shared_local_extraction": True,
            "max_items": len(source.seeds),
            "seed": seed,
        },
    )
    return result


def verification_report(
    bundle: Path, reviewed: Path | None, n: int, seed: int
) -> dict[str, object]:
    worksheet = reviewed or bundle / "verify_sample.csv"
    if n == 0 and reviewed is None:
        return {
            "worksheet": str(worksheet),
            "decided": 0,
            "accepted": 0,
            "accept_rate": None,
            "status": "no-kept-items",
        }
    if reviewed is None:
        build_sample_worksheet(bundle, worksheet, n=n, seed=seed)
    rows, _ = load_worksheet(worksheet)
    report = acceptance_report(rows)
    decided = cast(int, report["decided"])
    accepted = cast(int, report["accepted"])
    return {
        "worksheet": str(worksheet),
        "decided": decided,
        "accepted": accepted,
        "accept_rate": (accepted / decided) if decided else None,
        "status": "reviewed" if decided else "pending-human-review",
    }


def lane_report(result: PipelineResult, verification: dict[str, object]) -> dict[str, object]:
    gates = (result.calibration_report or {}).get("gates", {})
    seed_count = len(result.seeds)
    return {
        "bundle": str(result.out_dir),
        "seeds": seed_count,
        "kept": len(result.items),
        "kept_yield": (len(result.items) / seed_count) if seed_count else 0.0,
        "parse_rate": (
            result.draft_parsed / result.draft_attempts if result.draft_attempts else 0.0
        ),
        "gates": gates,
        "verify_sample": verification,
    }


def ranking(lanes: dict[str, dict[str, object]], metric: str) -> list[str]:
    def raw_value(name: str) -> object:
        lane = lanes[name]
        return (
            lane[metric]
            if metric in lane
            else cast(dict[str, object], lane["verify_sample"])[metric]
        )

    available = [name for name in lanes if raw_value(name) is not None]
    return sorted(
        available,
        key=lambda name: (-float(cast(float | int, raw_value(name))), name),
    )


def write_comparison_report(
    root: Path,
    corpus_root: Path | str,
    source: PipelineResult,
    lanes: dict[str, dict[str, object]],
    *,
    kind: str,
    lane_order: list[str],
    execution: dict[str, object],
) -> dict[str, object]:
    report: dict[str, object] = {
        "kind": kind,
        "out_dir": str(root),
        "corpus_root": str(corpus_root),
        "lane_order": lane_order,
        "execution": execution,
        "shared_seed_fingerprints": [seed_fingerprint(item) for item in source.seeds],
        "lanes": lanes,
        "rankings": {
            "kept_yield": ranking(lanes, "kept_yield"),
            "accept_rate": ranking(lanes, "accept_rate"),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / COMPARE_REPORT_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
