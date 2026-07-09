"""Operator recommendation summary from final-split run bundles.

Turns the ranked leaderboard into a few plain-language picks an operator actually needs after a
sweep: the best RAG accuracy, the most efficient model for THIS host (quality per watt), the fastest,
and the model we recommend running here -- the highest-accuracy candidate that is feasible,
Pareto-optimal, and fits the GPU tier's VRAM budget with headroom. Selection is pure and testable;
host detection and chart rendering live behind injectable seams / a guarded matplotlib import.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from llb.board.runs import RunRecord, best_per_model, load_run_records
from llb.core.contracts import BoardRow, JsonObject
from llb.prompts import render_text
from llb.scoring.aggregate import (
    ModelResult,
    pareto_front,
    rank_board,
    ranking_policy_note,
)

_LOG = logging.getLogger(__name__)

# Keep some VRAM headroom so the "recommended for this host" pick is not a card pinned at 100%.
SAFE_VRAM_FRACTION = 0.92
RAG_CONFIG_KEYS = ("strategy", "chunk_size", "chunk_overlap", "top_k", "retrieval_mode")
# The recommend summary quotes at most this many ranked miss-analysis recommendation lines;
# the full ranked list stays in the analysis report it links.
MISS_SECTION_MAX_RECOMMENDATIONS = 5


def _t(name: str, **values: object) -> str:
    """Render a `board.recommend.<name>` text template. The report prose lives in prompt templates
    (`prompts/templates/board/recommend/`) so the wording is reviewable in files, not inline here."""
    return render_text(f"board.recommend.{name}", values)


@dataclass
class RunSummary:
    """One model's best final-split run plus the host-efficiency + retrieval fields the board omits."""

    record: RunRecord
    quality_per_watt: float | None
    mean_power_w: float | None
    recall_at_k: float | None
    mrr: float | None

    @property
    def result(self) -> ModelResult:
        return self.record.result

    @property
    def model(self) -> str:
        return self.record.result.model


@dataclass
class HostInfo:
    tier_gb: int
    total_mb: int
    gpu_name: str
    detected: bool


@dataclass
class Recommendation:
    host: HostInfo
    summaries: list[RunSummary]  # the ranked cohort (shared split + n_cases)
    excluded: list[RunSummary]  # off-cohort runs named but not ranked (different split/n_cases)
    ranked: list[BoardRow]
    policy: str
    best_quality: RunSummary
    best_efficiency: RunSummary | None
    fastest: RunSummary
    recommended_for_host: RunSummary
    recall_at_k: float | None
    mrr: float | None
    top_k: int | None
    rag_config: JsonObject
    min_tokens_per_s: float = (
        0.0  # good-enough-performance floor applied to the host pick (0 = off)
    )


def _manifest_extras(
    run_dir: Path,
) -> tuple[float | None, float | None, float | None, float | None]:
    """(quality_per_watt, mean_power_w, recall_at_k, mrr) from a run bundle manifest; None when absent."""
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, None, None
    metrics = manifest.get("metrics") or {}
    retrieval = manifest.get("retrieval") or {}
    recall = retrieval.get("recall_at_k", retrieval.get("recall"))
    return (
        _as_float(metrics.get("quality_per_watt")),
        _as_float(metrics.get("mean_power_w")),
        _as_float(recall),
        _as_float(retrieval.get("mrr")),
    )


def _as_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_summary(record: RunRecord) -> RunSummary:
    """Enrich a run record with the host-efficiency + retrieval fields the board omits."""
    qpw, power, recall, mrr = _manifest_extras(Path(record.run_dir))
    return RunSummary(record, qpw, power, recall, mrr)


def load_run_summaries(run_root: Path | str, *, min_cases: int = 1) -> list[RunSummary]:
    """Best final-split run per model, enriched with host-efficiency + retrieval fields.

    `min_cases` drops partial/smoke bundles (e.g. a 3-case manual run) so they do not pollute the
    comparison against full-split sweeps. The filter runs BEFORE the best-per-model dedup so a
    high-scoring partial run can never shadow the model's full-split run. The split/dedup policy is
    the board's, reused verbatim.
    """
    records = [r for r in load_run_records(run_root) if r.result.n_cases >= min_cases]
    return [_to_summary(record) for record in best_per_model(records)]


def _cell_key(record: RunRecord) -> tuple[str, tuple[tuple[str, object], ...]]:
    """A (model, RAG-config) fingerprint: two runs share a cell iff model AND every RAG knob match.
    top_k is in the fingerprint, so grid points at different depths are DISTINCT cells (not merged)."""
    config = record.config if isinstance(record.config, dict) else {}
    return record.result.model, tuple((key, config.get(key)) for key in RAG_CONFIG_KEYS)


def load_config_cells(run_root: Path | str, *, min_cases: int = 1) -> list[RunSummary]:
    """Every final-split (model, RAG-config) cell -- the per-configuration evidence a RAG grid sweep
    produces. Unlike `load_run_summaries` this does NOT collapse to best-per-model: it keeps one row
    per (model, top_k) cell (best re-run of that exact cell), so a model swept at several retrieval
    depths shows all of them for the model x config comparison."""
    records = [r for r in load_run_records(run_root) if r.result.n_cases >= min_cases]
    best: dict[tuple[str, tuple[tuple[str, object], ...]], RunRecord] = {}
    for record in records:
        key = _cell_key(record)
        current = best.get(key)
        if current is None or record.result.objective_score > current.result.objective_score:
            best[key] = record
    return [_to_summary(record) for record in best.values()]


def select_cohort(
    summaries: list[RunSummary],
) -> tuple[list[RunSummary], list[RunSummary]]:
    """Split into the dominant `(split, n_cases)` cohort and the off-cohort remainder.

    Ranking models is only apples-to-apples within a shared split AND case count: a sweep at n=82
    and a platform-matrix row at n=20 are not one comparison, and a 2-case smoke run is noise.
    Keep the cohort with the most models (ties -> the larger n_cases, the more robust comparison)
    and return the rest as `excluded` so the summary can name them rather than silently rank them
    together. `--min-cases` still pre-filters smoke runs before this; the cohort split is the
    backstop when several real case counts coexist.
    """
    groups: dict[tuple[str, int], list[RunSummary]] = {}
    for summary in summaries:
        groups.setdefault((summary.record.split, summary.result.n_cases), []).append(summary)
    dominant = max(groups, key=lambda key: (len(groups[key]), key[1]))
    cohort = groups[dominant]
    excluded = [s for s in summaries if (s.record.split, s.result.n_cases) != dominant]
    return cohort, excluded


def _fits_host(summary: RunSummary, total_mb: int) -> bool:
    """Does the model's measured peak VRAM leave headroom on the host? (unknown VRAM -> assume yes)."""
    vram = summary.result.peak_vram_mb
    if total_mb <= 0 or vram is None:
        return True
    return vram <= SAFE_VRAM_FRACTION * total_mb


def _recommended_for_host(
    summaries: list[RunSummary], front: set[str], total_mb: int, min_tokens_per_s: float = 0.0
) -> RunSummary:
    """Highest-accuracy model that is Pareto-optimal, fits the host VRAM budget with headroom, AND
    clears the good-enough-performance floor (`min_tokens_per_s`, 0 = off).

    This is quality optimization SUBJECT TO host constraints: an accurate model that is too slow or
    VRAM-bound is not a good local default. Constraints relax in order (performance -> VRAM -> Pareto)
    so a host where nothing clears them still yields a (clearly caveated) pick rather than nothing.
    """
    by_quality = sorted(summaries, key=lambda s: s.result.objective_score, reverse=True)
    pareto = [s for s in by_quality if s.model in front]
    fitting = [s for s in pareto if _fits_host(s, total_mb)]
    fast_enough = [s for s in fitting if s.result.tokens_per_s >= min_tokens_per_s]
    return (fast_enough or fitting or pareto or by_quality)[0]


def build_recommendation(
    summaries: list[RunSummary],
    host: HostInfo,
    *,
    judge_trusted: bool = False,
    min_tokens_per_s: float = 0.0,
) -> Recommendation:
    """Rank the runs and pick the operator-facing winners. Requires at least one summary.

    Only the dominant `(split, n_cases)` cohort is ranked; off-cohort runs (a smaller
    platform-matrix or smoke bundle) are reported separately so the headline does not mix sample
    sizes. `min_tokens_per_s` (0 = off) is the good-enough-performance floor the host pick must
    clear on top of the VRAM-fit constraint.
    """
    if not summaries:
        raise ValueError("no final-split run bundles to recommend from")
    cohort, excluded = select_cohort(summaries)
    results = [s.result for s in cohort]
    ranked = rank_board(results, judge_trusted=judge_trusted)
    policy = ranking_policy_note(results, judge_trusted)
    front = pareto_front(results, judge_trusted, weight_judge=0.5)

    by_model = {s.model: s for s in cohort}
    best_quality = by_model[ranked[0]["model"]]
    with_power = [s for s in cohort if s.quality_per_watt]
    best_efficiency = (
        max(with_power, key=lambda s: s.quality_per_watt or 0.0) if with_power else None
    )
    fastest = max(cohort, key=lambda s: s.result.tokens_per_s)
    recommended = _recommended_for_host(cohort, front, host.total_mb, min_tokens_per_s)

    config = best_quality.record.config
    return Recommendation(
        host=host,
        summaries=cohort,
        excluded=excluded,
        ranked=ranked,
        policy=policy,
        best_quality=best_quality,
        best_efficiency=best_efficiency,
        fastest=fastest,
        recommended_for_host=recommended,
        recall_at_k=best_quality.recall_at_k,
        mrr=best_quality.mrr,
        top_k=config.get("top_k") if isinstance(config, dict) else None,
        rag_config={k: config.get(k) for k in RAG_CONFIG_KEYS} if isinstance(config, dict) else {},
        min_tokens_per_s=min_tokens_per_s,
    )


def _short(model: str) -> str:
    """A compact label for a model: the last path/tag segment, trimmed of a GGUF quant suffix."""
    tail = model.rstrip("/").split("/")[-1]
    return tail


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


def _qpw(summary: RunSummary | None) -> str:
    if summary is None or summary.quality_per_watt is None:
        return "n/a"
    return f"{summary.quality_per_watt:.3f}"


def format_summary_md(rec: Recommendation) -> str:
    """Render the recommendation as a Markdown report (host-adaptive, plain-language picks).

    All report prose is sourced from `board.recommend.*` templates; this function only computes the
    values and assembles the line list (headers and the comparison table are pure markdown).
    """
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


def _vram(summary: RunSummary) -> str:
    vram = summary.result.peak_vram_mb
    return "n/a" if vram is None else f"{vram:.0f} MiB"


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


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


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


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.4f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"


def format_config_detail_md(cells: list[RunSummary]) -> str:
    """Render the detailed (model x config) proof: one row per (model, top_k) cell of the ranked
    cohort, grouped by model with each model's best config marked. '' when there are no cells.

    This is the evidence behind the headline picks -- it shows how each model's accuracy and speed
    move with retrieval depth, so the winning configuration is demonstrated, not assumed. When no
    model was swept at more than one config it appends a note pointing at the RAG grid.
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
        for rank, cell in enumerate(group):
            config = cell.record.config if isinstance(cell.record.config, dict) else {}
            recall = cell.recall_at_k
            rows.append(
                [
                    _short(model),
                    str(config.get("top_k", "?")),
                    "*" if rank == 0 else "",
                    f"{cell.result.objective_score:.3f}",
                    f"{cell.result.tokens_per_s:.1f}",
                    _vram(cell),
                    f"{recall:.3f}" if recall is not None else "n/a",
                ]
            )
    table = _md_table(["model", "top_k", "best", "objective", "tok/s", "peak VRAM", "recall"], rows)
    lines = [_t("config_detail_heading"), "", _t("config_detail_intro"), "", table]
    if max(len(group) for group in by_model.values()) == 1:
        lines += ["", _t("config_detail_single")]
    return "\n".join(lines)
