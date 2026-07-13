"""Extra `llb recommend` report sections beyond the headline picks: miss analysis, self-improvement,
the multi-model fine-tune campaign, and the context-policy comparison.

Each `latest_*` loads the newest persisted payload (or None), and each `format_*_section_md` renders
it to Markdown -- returning '' when the artifact is absent so the summary stays unchanged for
operators who never ran that flow.
"""

import json
from pathlib import Path

from llb.board.recommend.model import (
    MISS_SECTION_MAX_RECOMMENDATIONS,
    _float_for_sort,
    _fmt_float,
    _short,
    _t,
)
from llb.core.contracts import JsonObject


def format_miss_section_md(analysis: JsonObject | None) -> str:
    """Render the recommend summary's miss-analysis section from the latest persisted
    `analysis.json` payload (see `llb.board.miss_analysis.latest_analysis`); '' when no
    analysis exists so the summary stays unchanged for operators who never ran one."""
    if not analysis:
        return ""
    class_counts = analysis.get("class_counts") or {}
    classes = ", ".join(f"{cls}={n}" for cls, n in class_counts.items() if n) or "none"
    lines = [
        "## Miss analysis",
        "",
        _t(
            "misses_intro",
            n_misses=analysis.get("n_misses", 0),
            n_cases=analysis.get("n_cases", 0),
            model=_short(str(analysis.get("model", "?"))),
            split=analysis.get("split", "?"),
            classes=classes,
            report=analysis.get("report_path", "?"),
        ),
    ]
    recommendations = analysis.get("recommendations") or []
    if recommendations:
        lines += [""] + [
            f"{rank}. {rec.get('line', '')}"
            for rank, rec in enumerate(recommendations[:MISS_SECTION_MAX_RECOMMENDATIONS], 1)
        ]
    return "\n".join(lines)


def latest_self_improvement(data_dir: Path | str) -> JsonObject | None:
    """Newest `$DATA_DIR/self-improve/*/state.json` with report path attached."""
    root = Path(data_dir) / "self-improve"
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        state_path = candidate / "state.json"
        if not state_path.is_file():
            continue
        try:
            payload: JsonObject = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["report_path"] = str(candidate / "report.md")
        payload["campaign_dir"] = str(candidate)
        return payload
    return None


def format_self_improvement_section_md(campaign: JsonObject | None) -> str:
    """Render latest self-improvement campaign status for `llb recommend`."""
    if not campaign:
        return ""
    rounds = campaign.get("rounds") or []
    if not isinstance(rounds, list) or not rounds:
        return ""
    lines = [
        "## Self-improvement",
        "",
        f"Campaign: `{campaign.get('campaign_dir', '?')}`",
        f"Report: `{campaign.get('report_path', '?')}`",
        "",
        "| round | base objective | tuned objective | delta | verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rounds:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("round", "?")),
                    _fmt_float(row.get("base_objective")),
                    _fmt_float(row.get("tuned_objective")),
                    _fmt_float(row.get("delta")),
                    str(row.get("verdict", "?")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def latest_finetune_campaign(data_dir: Path | str) -> JsonObject | None:
    """Newest multi-model fine-tune campaign payload, or None when no campaign exists."""
    from llb.finetune.campaign import latest_campaign

    return latest_campaign(data_dir)


def format_finetune_campaign_section_md(campaign: JsonObject | None) -> str:
    """Render latest multi-model tunability ranking for `llb recommend`."""
    if not campaign:
        return ""
    entries = campaign.get("entries") or []
    if not isinstance(entries, list) or not entries:
        return ""
    completed = [entry for entry in entries if isinstance(entry, dict)]
    ranked = sorted(
        completed,
        key=lambda entry: (
            _float_for_sort(entry.get("delta")),
            -_float_for_sort(entry.get("train_wall_clock_s")),
            -_float_for_sort(entry.get("peak_vram_mb")),
        ),
        reverse=True,
    )
    lines = [
        "## Fine-tune campaign",
        "",
        f"Campaign: `{campaign.get('campaign_dir', '?')}`",
        f"Report: `{campaign.get('report_path', '?')}`",
        "",
        "| rank | model | base objective | adapted objective | delta | train s | peak VRAM | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(ranked, 1):
        status = str(row.get("status", "?"))
        reason = row.get("reason")
        if reason:
            status = f"{status}: {reason}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank) if row.get("status") == "completed" else "",
                    _short(str(row.get("model", "?"))),
                    _fmt_float(row.get("base_objective")),
                    _fmt_float(row.get("tuned_objective")),
                    _fmt_float(row.get("delta")),
                    _fmt_float(row.get("train_wall_clock_s")),
                    _fmt_float(row.get("peak_vram_mb")),
                    status,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def latest_chain_context(data_dir: Path | str) -> JsonObject | None:
    """Newest-per-policy context-policy comparison, grouped by model, or None when no bundles."""
    from llb.board.chain_context import load_chain_context_records

    records = load_chain_context_records(data_dir)
    if not records:
        return None
    models: dict[str, list[JsonObject]] = {}
    for record in records:
        models.setdefault(record.model, []).append(
            {
                "policy": record.policy,
                "final_objective": record.result.objective_score,
                "per_step_objective": record.per_step_objective,
            }
        )
    return {"models": models}


def format_chain_context_section_md(payload: JsonObject | None) -> str:
    """Render the context-policy ranking per model for `llb recommend` (best policy first)."""
    if not payload:
        return ""
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        return ""
    lines = [
        "## Context policy",
        "",
        "Per model, the chain set + scoring stay fixed; only the context policy varies "
        "(fresh / history / summary / roles).",
        "",
        "| model | best policy | final objective | per-step objective | ranking |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in sorted(models):
        rows = [row for row in models[model] if isinstance(row, dict)]
        if not rows:
            continue
        ranked = sorted(rows, key=lambda r: _float_for_sort(r.get("final_objective")), reverse=True)
        best = ranked[0]
        ranking = ", ".join(
            f"{r.get('policy', '?')} {_fmt_float(r.get('final_objective'))}" for r in ranked
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _short(str(model)),
                    str(best.get("policy", "?")),
                    _fmt_float(best.get("final_objective")),
                    _fmt_float(best.get("per_step_objective")),
                    ranking,
                ]
            )
            + " |"
        )
    return "\n".join(lines)
