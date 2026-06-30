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
from llb.contracts import BoardRow, JsonObject
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


def load_run_summaries(run_root: Path | str, *, min_cases: int = 1) -> list[RunSummary]:
    """Best final-split run per model, enriched with host-efficiency + retrieval fields.

    `min_cases` drops partial/smoke bundles (e.g. a 3-case manual run) so they do not pollute the
    comparison against full-split sweeps. The filter runs BEFORE the best-per-model dedup so a
    high-scoring partial run can never shadow the model's full-split run. The split/dedup policy is
    the board's, reused verbatim.
    """
    records = [r for r in load_run_records(run_root) if r.result.n_cases >= min_cases]
    summaries: list[RunSummary] = []
    for record in best_per_model(records):
        qpw, power, recall, mrr = _manifest_extras(Path(record.run_dir))
        summaries.append(RunSummary(record, qpw, power, recall, mrr))
    return summaries


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
    return f", best RAG top_k {top_k}" if top_k is not None else ""


def _excluded_line(excluded: list[RunSummary]) -> str:
    """A one-line note naming off-cohort runs that were not ranked, or '' when there are none."""
    if not excluded:
        return ""
    listed = ", ".join(
        f"{_short(s.model)} n={s.result.n_cases}"
        for s in sorted(excluded, key=lambda s: s.result.n_cases, reverse=True)
    )
    return (
        f"Excluded (off-cohort, not ranked): {listed} "
        "-- different split/case count; raise --min-cases or re-run them at the cohort's case count."
    )


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
    return f"- Higher accuracy but below the {floor:.0f} tok/s floor (traded away for speed): {listed}."


def _qpw(summary: RunSummary | None) -> str:
    if summary is None or summary.quality_per_watt is None:
        return "n/a"
    return f"{summary.quality_per_watt:.3f}"


def format_summary_md(rec: Recommendation) -> str:
    """Render the recommendation as a Markdown report (host-adaptive, plain-language picks)."""
    host = rec.host
    host_line = (
        f"{host.gpu_name or 'GPU'} -- {host.tier_gb} GiB tier ({host.total_mb} MiB)"
        if host.detected
        else f"{host.tier_gb} GiB tier (no GPU detected; planning budget only)"
    )
    bq, be, fast, rec_host = (
        rec.best_quality,
        rec.best_efficiency,
        rec.fastest,
        rec.recommended_for_host,
    )
    eff_model = _short(be.model) if be else "n/a"
    eff_note = "" if be is None else " -- best accuracy you can buy per watt on this host."
    recall_str = f"{rec.recall_at_k:.3f}" if rec.recall_at_k is not None else "n/a"
    mrr_str = f"{rec.mrr:.3f}" if rec.mrr is not None else "n/a"
    config_str = ", ".join(
        f"{k}={rec.rag_config.get(k)}" for k in RAG_CONFIG_KEYS if rec.rag_config.get(k) is not None
    )
    # The ranked set is one cohort, so every row shares this case count (apples-to-apples).
    n_cases = rec.summaries[0].result.n_cases
    excluded_line = _excluded_line(rec.excluded)
    floor = rec.min_tokens_per_s
    fit_clause = f"fits the {host.tier_gb} GiB VRAM budget with headroom"
    if floor > 0:
        fit_clause += f" and clears the {floor:.0f} tok/s performance floor"
    too_slow_line = _too_slow_note(rec)
    lines = [
        "# loc-lm-bench recommendation summary",
        "",
        f"Host: {host_line}",
        f"Models compared: {len(rec.summaries)} (final split, {n_cases} cases)",
        *([excluded_line] if excluded_line else []),
        "",
        "## Recommendations",
        "",
        f"- Recommended for this host: **{_short(rec_host.model)}** "
        f"({rec_host.result.backend}) -- highest-accuracy model that is Pareto-optimal and "
        f"{fit_clause} "
        f"(objective {rec_host.result.objective_score:.3f}, "
        f"{rec_host.result.tokens_per_s:.1f} tok/s, "
        f"peak VRAM {_vram(rec_host)}{_top_k_note(rec_host)}).",
        *([too_slow_line] if too_slow_line else []),
        f"- Best RAG accuracy: **{_short(bq.model)}** "
        f"(objective {bq.result.objective_score:.3f}, quality/W {_qpw(bq)}).",
        f"- Best efficiency (quality per watt): **{eff_model}** (quality/W {_qpw(be)}).{eff_note}",
        f"- Fastest: **{_short(fast.model)}** ({fast.result.tokens_per_s:.1f} tok/s).",
        "",
        "## Retrieval (RAG) health",
        "",
        f"- recall@{rec.top_k or '?'}: {recall_str}, MRR: {mrr_str} "
        "(shared FAISS index; retrieval is not the bottleneck when this is high).",
        f"- Best RAG config (from the top run): {config_str}",
        "",
        "## Model comparison",
        "",
        _comparison_table(rec),
        "",
        rec.policy,
        "Quality is the ranking axis; efficiency (quality/W) is the host-cost axis. A model can win "
        "accuracy yet lose on this host if it is slow or VRAM-bound.",
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
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])
