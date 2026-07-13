"""Human-review status and mechanical acceptance gates for draft comparisons."""

import json
from pathlib import Path
from typing import cast

from llb.goldset.verify_base import load_worksheet
from llb.prep.ontology.compare import FRONTIER_LANE, LOCAL_LANE
from llb.prep.ontology.compare_artifacts import ranking, verification_report
from llb.prep.ontology.local_compare import BASELINE_LANE, PROBE_LANE


def comparison_worksheets(report_path: Path | str) -> dict[str, Path]:
    """Return the generated worksheet for each comparison lane."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    lanes = cast(dict[str, dict[str, object]], report["lanes"])
    return {
        lane: Path(str(cast(dict[str, object], lanes[lane]["verify_sample"])["worksheet"]))
        for lane in cast(list[str], report.get("lane_order") or list(lanes))
    }


def worksheet_progress(path: Path | str) -> tuple[int, int]:
    """Return ``(decided, total)`` for one comparison worksheet."""
    rows, _ = load_worksheet(Path(path))
    decided = sum(1 for row in rows if (row.get("decision") or "").strip() in ("accept", "reject"))
    return decided, len(rows)


def _budget_status(frontier_bundle: Path) -> dict[str, object]:
    provenance = json.loads((frontier_bundle / "provenance.json").read_text(encoding="utf-8"))
    endpoint = cast(dict[str, object], provenance["endpoint"])
    stages = cast(dict[str, dict[str, object]], endpoint["stages"])
    egress = [stage for stage in stages.values() if stage.get("egress") is True]
    max_calls = [
        int(cast(int | str, stage["max_calls"]))
        for stage in egress
        if stage.get("max_calls") is not None
    ]
    max_usd = [
        float(cast(float | int | str, stage["max_usd"]))
        for stage in egress
        if stage.get("max_usd") is not None
    ]
    calls = int(cast(int | str, endpoint["calls"]))
    cost_usd = float(cast(float | int | str, endpoint["cost_usd"]))
    return {
        "calls": calls,
        "max_calls": min(max_calls) if max_calls else None,
        "cost_usd": cost_usd,
        "max_usd": min(max_usd) if max_usd else None,
        "call_cap_passed": bool(max_calls) and calls <= min(max_calls),
        "spend_cap_passed": not max_usd or cost_usd <= min(max_usd),
    }


def _refresh_reviews(path: Path, worksheets: dict[str, Path]) -> dict[str, object]:
    report = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    lanes = cast(dict[str, dict[str, object]], report["lanes"])
    for lane, worksheet in worksheets.items():
        lanes[lane]["verify_sample"] = verification_report(
            Path(str(lanes[lane]["bundle"])), worksheet, 0, 0
        )
    rankings = cast(dict[str, list[str]], report["rankings"])
    rankings["accept_rate"] = ranking(lanes, "accept_rate")
    return report


def finalize_comparison(report_path: Path | str) -> dict[str, object]:
    """Refresh human metrics, evaluate task gates, and persist the finalization result."""
    path = Path(report_path)
    worksheets = comparison_worksheets(path)
    report = _refresh_reviews(path, worksheets)
    lanes = cast(dict[str, dict[str, object]], report["lanes"])
    rankings = cast(dict[str, list[str]], report["rankings"])
    progress = {lane: worksheet_progress(worksheet) for lane, worksheet in worksheets.items()}
    expected_ranking = set(worksheets)
    checks: dict[str, bool] = {
        "worksheets_nonempty": all(total > 0 for _, total in progress.values()),
        "worksheets_complete": all(decided == total for decided, total in progress.values()),
        **{
            f"{lane}_calibration": bool(cast(dict[str, object], lanes[lane]["gates"]).get("passed"))
            for lane in expected_ranking
        },
        "kept_yield_ranking_complete": set(rankings.get("kept_yield", [])) == expected_ranking,
        "accept_rate_ranking_complete": set(rankings.get("accept_rate", [])) == expected_ranking
        and all(
            cast(dict[str, object], lanes[lane]["verify_sample"])["accept_rate"] is not None
            for lane in expected_ranking
        ),
    }
    details: dict[str, object] = {}
    if expected_ranking == {LOCAL_LANE, FRONTIER_LANE}:
        budget = _budget_status(Path(str(lanes[FRONTIER_LANE]["bundle"])))
        checks.update(
            {
                "frontier_parse_rate_at_least_local": float(
                    cast(float | int | str, lanes[FRONTIER_LANE]["parse_rate"])
                )
                >= float(cast(float | int | str, lanes[LOCAL_LANE]["parse_rate"])),
                "frontier_call_cap": bool(budget["call_cap_passed"]),
                "frontier_spend_cap": bool(budget["spend_cap_passed"]),
            }
        )
        details["frontier_budget"] = budget
    elif expected_ranking == {BASELINE_LANE, PROBE_LANE}:
        execution = cast(dict[str, object], report.get("execution") or {})
        checks.update(
            {
                "sequential_local_execution": execution.get("mode") == "sequential-local-ollama",
                "model_unload_between_lanes": execution.get("unload_between_lanes") is True,
            }
        )
    else:
        checks["recognized_lane_schema"] = False
    finalization: dict[str, object] = {
        "passed": all(checks.values()),
        "checks": checks,
        "worksheet_progress": {
            lane: {"decided": decided, "total": total}
            for lane, (decided, total) in progress.items()
        },
        **details,
    }
    report["finalization"] = finalization
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
