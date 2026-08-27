"""The confirmation run's own artifact: `long_run.json` plus the page an operator reads.

`scoreboard.json` answers "which row scored best". This answers the question an adoption decision
actually turns on: what was declared before the run, what the search cost, how certain the ranking
is, what the public tracks say, and what the verdict therefore licenses. It sits beside the
scoreboard rather than replacing it -- the leak fence on that writer stays exactly as it was.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.optimize.joint_search.long_run.plan import LONG_RUN_METHOD, LongRunPlan
from llb.optimize.joint_search.long_run.uncertainty import BoardUncertainty
from llb.optimize.joint_search.long_run.verdict import AdoptionVerdict
from llb.optimize.tuning_space import FINAL_SPLIT
from llb.rag.fusion_evidence.paired import format_randomization_p
from llb.rag.fusion_evidence.power_report import power_summary

LONG_RUN_JSON = "long_run.json"
LONG_RUN_MD = "long_run.md"


def build_payload(
    *,
    run_id: str,
    plan: LongRunPlan,
    search: Mapping[str, Any],
    ledger: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    uncertainty: BoardUncertainty,
    realized_power: Mapping[str, Any] | None,
    public: Mapping[str, Any],
    verdict: AdoptionVerdict,
) -> dict[str, Any]:
    """Assemble the whole record; nothing here re-derives a number, it only arranges them."""
    return {
        "method": LONG_RUN_METHOD,
        "run_id": run_id,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predeclaration": plan.to_dict(),
        "search": dict(search),
        "screen_ledger": {
            "split": ledger.get("split"),
            "rounds": len(ledger.get("rounds") or []),
            "finalists": list(ledger.get("finalists") or []),
        },
        "final": {
            "split": FINAL_SPLIT,
            "entries": [dict(entry) for entry in entries],
            "uncertainty": uncertainty.to_dict(),
            "power": dict(realized_power) if realized_power else None,
        },
        "public_screen": dict(public),
        "verdict": verdict.to_dict(),
    }


def write_long_run(run_dir: Path, payload: Mapping[str, Any]) -> dict[str, Path]:
    """Write `long_run.json` and its Markdown rendering into the joint-search run dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / LONG_RUN_JSON
    md_path = run_dir / LONG_RUN_MD
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def render_markdown(payload: Mapping[str, Any]) -> str:
    """The operator page: verdict first, then what licenses it."""
    verdict = payload["verdict"]
    lines = [
        f"# Roster confirmation: {payload['run_id']}",
        "",
        f"**Verdict: {str(verdict['decision']).upper()}"
        + (f" `{verdict['model']}`**" if verdict["model"] else "**"),
        "",
        f"{verdict['reason']}.",
        "",
        f"Quality versus latency: {verdict['tradeoff']}.",
        "",
    ]
    lines += _predeclaration_block(payload["predeclaration"])
    lines += _search_block(payload["search"])
    lines += _board_block(payload["final"])
    lines += _public_block(payload["public_screen"])
    return "\n".join(lines) + "\n"


def _predeclaration_block(plan: Mapping[str, Any]) -> list[str]:
    screen = plan["screen"]
    lines = [
        "## Predeclared before the run",
        "",
        f"- minimum detectable objective gain: {plan['minimum_detectable_gain']:+.3f}",
        f"- tuning-screen size: {screen['applied_n']} of {screen['required_n']} required "
        f"({screen['available_n']} available; binding floor: {screen['binding']})",
        f"- stopping rule: {plan['stopping_rule']}",
        "",
    ]
    return lines + power_summary(plan["power"], title="Paired power (declared)")


def _search_block(search: Mapping[str, Any]) -> list[str]:
    blocks = search.get("blocks") or []
    lines = [
        "## Search",
        "",
        f"Stopped by **{search.get('stopped_by', '-')}** after "
        f"{search.get('trials_per_finalist', 0)} trials per finalist "
        f"({search.get('consumed_total', 0)} trials consumed in total).",
        "",
        "| block | trials/finalist | ranking | rank agreement | leader held | stable streak |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for block in blocks:
        agreement = block.get("agreement")
        lines.append(
            f"| {block['index']} | {block['trials_per_finalist']} | "
            f"{' > '.join(block['ranking'])} | "
            f"{'-' if agreement is None else f'{agreement:.2f}'} | "
            f"{_yes_no(block.get('leader_held'))} | {block['stable_streak']} |"
        )
    return lines + [""]


def _board_block(final: Mapping[str, Any]) -> list[str]:
    uncertainty = final["uncertainty"]
    quality = uncertainty["quality"]
    latency = uncertainty["latency"]
    paired = uncertainty["paired_vs_baseline"]
    frontier = set(uncertainty["pareto_frontier"])
    lines = [
        f"## Held-out board (`{final['split']}`, {uncertainty['n_items']} items, "
        f"{uncertainty['confidence']:.0%} intervals over {uncertainty['resamples']} resamples)",
        "",
        f"Baseline row: `{uncertainty['baseline'] or '-'}`",
        "",
        "| row | quality | latency (s) | delta vs baseline | W/L/T | rand p | frontier |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(quality):
        comparison = paired.get(row)
        lines.append(
            f"| {row} | {_interval(quality[row])} | {_interval(latency[row])} | "
            f"{_interval(comparison['delta']) if comparison else '-'} | "
            f"{_ledger(comparison)} | "
            f"{format_randomization_p(comparison) if comparison else '-'} | "
            f"{'yes' if row in frontier else 'no'} |"
        )
    realized = final.get("power")
    if realized:
        lines += power_summary(realized, title="Paired power (realized on the held-out split)")
    unreadable = uncertainty.get("unreadable_rows") or []
    if unreadable:
        lines += ["", "Rows that could not be re-read per case:", ""]
        lines += [f"- `{row['row']}`: {row['reason']}" for row in unreadable]
    return lines + [""]


def _public_block(public: Mapping[str, Any]) -> list[str]:
    reports = public.get("reports") or {}
    lines = [
        "## Public Ukrainian screen",
        "",
        f"Tracks: {', '.join(public.get('tracks') or ['-'])} "
        f"(cross-track ranking is refused: comparable={public.get('comparable')})",
        "",
        "| model | track | coverage | task | metric | score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, report in sorted(reports.items()):
        coverage = f"{len(report['covered'])}/{len(report['requested_tasks'])}"
        results = report.get("results") or []
        if not results:
            lines.append(f"| {name} | {report['track']} | {coverage} | - | - | - |")
        for result in results:
            lines.append(
                f"| {name} | {report['track']} | {coverage} | {result['task']} | "
                f"{result['metric']} | {result['score']:.3f} |"
            )
    for failure in public.get("failures") or []:
        lines.append(f"| {failure['model']} | - | FAILED | - | - | {failure['reason']} |")
    return lines + [""]


def _interval(interval: Mapping[str, Any]) -> str:
    return f"{interval['mean']:+.3f} [{interval['lo']:+.3f}, {interval['hi']:+.3f}]"


def _ledger(comparison: Mapping[str, Any] | None) -> str:
    if not comparison:
        return "-"
    return f"{comparison['wins']}/{comparison['losses']}/{comparison['ties']}"


def _yes_no(value: Any) -> str:
    return "-" if value is None else ("yes" if value else "no")


__all__ = ["LONG_RUN_JSON", "LONG_RUN_MD", "build_payload", "render_markdown", "write_long_run"]
