"""Load run bundles into `RunSummary`s and pick the operator-facing winners.

Selection is pure and testable: enrich each bundle with host-efficiency + retrieval fields, keep the
dominant `(split, n_cases)` cohort, then choose the best accuracy / efficiency / speed and the model
recommended for THIS host (highest accuracy that is Pareto-optimal, VRAM-fitting, and fast enough).
"""

import json
from pathlib import Path

from llb.board.recommend.model import (
    RAG_CONFIG_KEYS,
    SAFE_VRAM_FRACTION,
    HostInfo,
    Recommendation,
    RunSummary,
)
from llb.board.runs import RunRecord, best_per_model, load_run_records
from llb.scoring.aggregate import pareto_front, rank_board, ranking_policy_note


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
