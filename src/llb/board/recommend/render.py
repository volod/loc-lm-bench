"""Render a `Recommendation` as the operator Markdown summary, the machine-readable payload, and the
detailed (model x config) proof table.

All report prose is sourced from `board.recommend.*` templates; these functions compute the values
and assemble the line lists (headers and the comparison tables are pure markdown).
"""

from llb.board.recommend.build import _fits_host, select_cohort
from llb.board.recommend.model import (
    RAG_CONFIG_KEYS,
    Recommendation,
    RunSummary,
    _md_table,
    _qpw,
    _short,
    _t,
    _vram,
)
from llb.core.contracts.common import JsonObject


def _top_k_note(summary: RunSummary) -> str:
    """ ", best RAG top_k N" -- the retrieval depth this model's winning cell used (from a RAG grid
    sweep, this is the best top_k for that model); empty when the config has no top_k recorded."""
    config = summary.record.config
    top_k = config.get("top_k") if isinstance(config, dict) else None
    return _t("top_k_note", top_k=top_k) if top_k is not None else ""


def _excluded_line(excluded: list[RunSummary]) -> str:
    """A one-line note naming off-cohort runs that were not ranked, or '' when there are none."""
    if not excluded:
        return ""
    listed = ", ".join(
        f"{_short(s.model)} n={s.result.n_cases}"
        for s in sorted(excluded, key=lambda s: s.result.n_cases, reverse=True)
    )
    return _t("excluded", listed=listed)


def _too_slow_note(rec: "Recommendation") -> str:
    """Name VRAM-fitting models that out-score the host pick but fall below the performance floor,
    so the operator sees exactly what accuracy the floor traded away. '' when the floor is off or
    nothing was dropped for speed alone."""
    floor = rec.min_tokens_per_s
    if floor <= 0:
        return ""
    pick_obj = rec.recommended_for_host.result.objective_score
    slower = [
        s
        for s in rec.summaries
        if s.result.tokens_per_s < floor
        and s.result.objective_score > pick_obj
        and _fits_host(s, rec.host.total_mb)
    ]
    if not slower:
        return ""
    listed = ", ".join(
        f"{_short(s.model)} ({s.result.objective_score:.3f} obj, {s.result.tokens_per_s:.1f} tok/s)"
        for s in sorted(slower, key=lambda s: s.result.objective_score, reverse=True)
    )
    return _t("too_slow", floor=f"{floor:.0f}", listed=listed)


def format_summary_md(rec: Recommendation) -> str:
    """Render the recommendation as a Markdown report (host-adaptive, plain-language picks)."""
    host = rec.host
    host_line = (
        _t(
            "host_detected",
            gpu_name=host.gpu_name or "GPU",
            tier_gb=host.tier_gb,
            total_mb=host.total_mb,
        )
        if host.detected
        else _t("host_planning", tier_gb=host.tier_gb)
    )
    bq, be, fast, rec_host = (
        rec.best_quality,
        rec.best_efficiency,
        rec.fastest,
        rec.recommended_for_host,
    )
    recall_str = f"{rec.recall_at_k:.3f}" if rec.recall_at_k is not None else "n/a"
    mrr_str = f"{rec.mrr:.3f}" if rec.mrr is not None else "n/a"
    config_str = ", ".join(
        f"{k}={rec.rag_config.get(k)}" for k in RAG_CONFIG_KEYS if rec.rag_config.get(k) is not None
    )
    # The ranked set is one cohort, so every row shares this case count (apples-to-apples).
    n_cases = rec.summaries[0].result.n_cases
    excluded_line = _excluded_line(rec.excluded)
    floor = rec.min_tokens_per_s
    fit_clause = _t("fit_clause", tier_gb=host.tier_gb)
    if floor > 0:
        fit_clause += _t("fit_clause_floor", floor=f"{floor:.0f}")
    too_slow_line = _too_slow_note(rec)
    lines = [
        "# loc-lm-bench recommendation summary",
        "",
        f"Host: {host_line}",
        _t("models_compared", n_models=len(rec.summaries), n_cases=n_cases),
        *([excluded_line] if excluded_line else []),
        "",
        "## Recommendations",
        "",
        _t(
            "recommended",
            model=_short(rec_host.model),
            backend=rec_host.result.backend,
            fit_clause=fit_clause,
            objective=f"{rec_host.result.objective_score:.3f}",
            tokens_per_s=f"{rec_host.result.tokens_per_s:.1f}",
            vram=_vram(rec_host),
            top_k_note=_top_k_note(rec_host),
        ),
        *([too_slow_line] if too_slow_line else []),
        _t(
            "best_accuracy",
            model=_short(bq.model),
            objective=f"{bq.result.objective_score:.3f}",
            quality_per_watt=_qpw(bq),
        ),
        _t(
            "best_efficiency",
            model=_short(be.model) if be else "n/a",
            quality_per_watt=_qpw(be),
            note="" if be is None else _t("best_efficiency_note"),
        ),
        _t("fastest", model=_short(fast.model), tokens_per_s=f"{fast.result.tokens_per_s:.1f}"),
        "",
        "## Retrieval (RAG) health",
        "",
        _t("retrieval_health", top_k=rec.top_k or "?", recall=recall_str, mrr=mrr_str),
        _t("best_rag_config", config=config_str),
        "",
        "## Model comparison",
        "",
        _comparison_table(rec),
        "",
        rec.policy,
        _t("policy_axis"),
    ]
    return "\n".join(lines)


def _comparison_table(rec: Recommendation) -> str:
    headers = ["model", "backend", "objective", "reliab", "tok/s", "peak VRAM", "quality/W", "n"]
    by_model = {s.model: s for s in rec.summaries}
    rows = []
    for row in rec.ranked:
        s = by_model[row["model"]]
        rows.append(
            [
                _short(row["model"]),
                str(row["backend"]),
                f"{row['objective']:.3f}",
                f"{row['reliability']:.3f}",
                f"{row['tokens_per_s']:.1f}",
                _vram(s),
                _qpw(s),
                str(row["n_cases"]),
            ]
        )
    return _md_table(headers, rows)


def recommendation_payload(rec: Recommendation) -> JsonObject:
    """Machine-readable form of the recommendation summary for orchestration scripts."""
    by_model = {s.model: s for s in rec.summaries}

    def item(summary: RunSummary | None) -> JsonObject | None:
        if summary is None:
            return None
        config = summary.record.config if isinstance(summary.record.config, dict) else {}
        return {
            "model": summary.model,
            "label": _short(summary.model),
            "backend": summary.result.backend,
            "objective": summary.result.objective_score,
            "reliability": summary.result.reliability,
            "tokens_per_s": summary.result.tokens_per_s,
            "peak_vram_mb": summary.result.peak_vram_mb,
            "quality_per_watt": summary.quality_per_watt,
            "n_cases": summary.result.n_cases,
            "top_k": config.get("top_k"),
            "recall_at_k": summary.recall_at_k,
            "mrr": summary.mrr,
        }

    candidates = [item(by_model[row["model"]]) for row in rec.ranked]
    return {
        "host": {
            "tier_gb": rec.host.tier_gb,
            "total_mb": rec.host.total_mb,
            "gpu_name": rec.host.gpu_name,
            "detected": rec.host.detected,
        },
        "selection": {
            "recommended_for_host": item(rec.recommended_for_host),
            "best_quality": item(rec.best_quality),
            "best_efficiency": item(rec.best_efficiency),
            "fastest": item(rec.fastest),
        },
        "rag_config": rec.rag_config,
        "candidates": [candidate for candidate in candidates if candidate is not None],
    }


def format_config_detail_md(cells: list[RunSummary]) -> str:
    """Render the detailed (model x config) proof: one row per (model, top_k) cell of the ranked
    cohort, grouped by model with each model's best config marked. '' when there are no cells.

    This is the evidence behind the headline picks -- how each model's accuracy and speed move with
    retrieval depth. When no model was swept at more than one config it appends a RAG-grid note.
    """
    if not cells:
        return ""
    cohort, _ = select_cohort(cells)
    by_model: dict[str, list[RunSummary]] = {}
    for cell in cohort:
        by_model.setdefault(cell.model, []).append(cell)
    ordered = sorted(
        by_model,
        key=lambda m: max(c.result.objective_score for c in by_model[m]),
        reverse=True,
    )
    rows: list[list[str]] = []
    for model in ordered:
        group = sorted(by_model[model], key=lambda c: c.result.objective_score, reverse=True)
        rows += [_config_detail_row(model, rank, cell) for rank, cell in enumerate(group)]
    table = _md_table(["model", "top_k", "best", "objective", "tok/s", "peak VRAM", "recall"], rows)
    lines = [_t("config_detail_heading"), "", _t("config_detail_intro"), "", table]
    if max(len(group) for group in by_model.values()) == 1:
        lines += ["", _t("config_detail_single")]
    return "\n".join(lines)


def _config_detail_row(model: str, rank: int, cell: RunSummary) -> list[str]:
    """One (model, top_k) table row; rank 0 within the model group is marked as best."""
    config = cell.record.config if isinstance(cell.record.config, dict) else {}
    recall = cell.recall_at_k
    return [
        _short(model),
        str(config.get("top_k", "?")),
        "*" if rank == 0 else "",
        f"{cell.result.objective_score:.3f}",
        f"{cell.result.tokens_per_s:.1f}",
        _vram(cell),
        f"{recall:.3f}" if recall is not None else "n/a",
    ]
