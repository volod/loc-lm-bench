"""Rendering for the reranker bake-off: the ranked table, the cost columns, and the keep-or-swap.

The shared paired/verdict/gate/boundary sections come from `llb.rag.bakeoff_report_sections`; what
this module owns is the part that is specific to choosing a RERANKER -- first-hit rank beside
recall, the per-query rerank latency and VRAM footprint a swap is actually paid in, and the
reranker-off row that says whether the second model is worth running at all.

Both adoption bars are printed as delta columns because both are on by default in this lane
(`llb.rag.rerank_bakeoff.lane`): a cross-encoder can only re-sort what it is handed, so rank is the
quantity it moves, and a report that showed only recall would hide the effect being bought.
"""

from llb.rag.bakeoff_report_sections import (
    NO_PAIRED_CELL,
    boundary_section,
    gate_summary,
    paired_cells,
    skipped_section,
    verdict_lines,
)
from llb.rag.embedding_bakeoff_uncertainty import (
    BAR_FIRST_HIT,
    BAR_RECALL,
    DEFAULT_CONFIDENCE,
)
from llb.rag.rerank_bakeoff.models import (
    KIND_RETRIEVAL_ORDER,
    ROW_NO_RERANK,
    RerankBakeoffReport,
    RerankCandidateResult,
)

_HEADLINE = "[compare-rerankers]"


def _rank_key(row: RerankCandidateResult) -> tuple[float, float, float, str]:
    """Rank rows the way the recommendation reads them: recall, then MRR, then cheaper."""
    return (-row["recall_at_k"], -row["mrr"], row["rerank_ms_per_query"], row["model"])


def _cell(value: float | None, spec: str = ".1f") -> str:
    return format(value, spec) if value is not None else NO_PAIRED_CELL


def _fit_cell(row: RerankCandidateResult) -> str:
    """Whether the measured peak footprint fits the declared budget (`-` when none was declared)."""
    fits = row.get("fits_headroom")
    return NO_PAIRED_CELL if fits is None else ("yes" if fits else "NO")


def format_report(report: RerankBakeoffReport) -> str:
    """ASCII summary for the terminal (AGENTS.md: ASCII-only, no box-drawing)."""
    rows = report["candidates"]
    lines = [
        f"{_HEADLINE} n={report['n']} k={report['k']} pool={report['pool_depth']} "
        f"encoder={report['embedding_model']} chunking={report['chunking']}"
    ]
    if not rows:
        lines.append("  (no candidates)")
        return "\n".join(lines)
    width = max(len(row["model"]) for row in rows)
    lines.append(
        f"  {'model'.ljust(width)}   recall@k     mrr   1st-hit   ms/query   VRAM_MB"
        "   d_recall vs baseline"
    )
    for row in sorted(rows, key=_rank_key):
        delta, _ledger, _sign_p, _rand_p, _reading = paired_cells(row, BAR_RECALL)
        lines.append(
            f"  {row['model'].ljust(width)}   {row['recall_at_k']:8.3f} {row['mrr']:7.3f} "
            f"{_cell(row['first_hit_rank_mean'], '.2f'):>9} "
            f"{row['rerank_ms_per_query']:10.1f} {_cell(row.get('vram_mb'), '.0f'):>9}"
            f"   {delta:>22}"
        )
    lines.append(
        f"  best (recall@k): {report['best_recall']}; best (MRR): {report['best_first_hit']}"
    )
    for skipped in report.get("skipped") or []:
        lines.append(f"  skipped: {skipped['model']} -- {skipped['detail']}")
    lines.extend(verdict_lines(report, prefix="  ", call_word="SWAP TO"))
    floor = report.get("noise_floor")
    if floor is not None:
        from llb.rag.noise_floor_report import format_noise_floor

        lines.extend(format_noise_floor(floor))
    return "\n".join(lines)


def _header_lines(report: RerankBakeoffReport) -> list[str]:
    """What the run pinned, so two reports are comparable without opening the config."""
    settings = report.get("uncertainty")
    baseline = settings["baseline"] if settings else None
    lines = [
        "# Reranker bake-off (Ukrainian RAG)",
        "",
        f"- corpus: `{report['corpus_root']}`",
        f"- items scored: {report['n']}",
        f"- cutoff: recall@{report['k']} / MRR / first-hit rank",
        f"- fixed retrieval: encoder `{report['embedding_model']}`, chunking `{report['chunking']}`,"
        f" candidate pool {report['pool_depth']} (every candidate re-sorts the SAME pool)",
        f"- rerank batch size: {report['batch_size']}",
    ]
    if (headroom := report.get("headroom")) is not None:
        lines.append(
            "- VRAM budget: "
            + (
                f"{_cell(headroom['headroom_mb'], '.0f')} MB left for the reranker "
                f"(device {_cell(headroom['total_mb'], '.0f')} MB - generator "
                f"{_cell(headroom['generator_mb'], '.0f')} MB - {headroom['reserve_mb']:.0f} MB "
                "reserve)"
                if headroom["headroom_mb"] is not None
                else "no generator residency declared, so footprints are reported and the fit gate "
                "did not run"
            )
        )
    if settings is not None:
        lines.append(
            f"- paired uncertainty: baseline `{baseline}`, {settings['resamples']} resamples, "
            f"{settings['confidence']:.0%} percentile bootstrap, seed {settings['seed']}"
        )
        lines.append(f"- adoption bar(s): {', '.join(settings.get('bars') or [BAR_RECALL])}")
    return lines


def _table(report: RerankBakeoffReport, baseline: str | None) -> list[str]:
    lines = [
        "",
        "| model | family | recall@k | MRR | first-hit rank | items hit | rerank ms/query "
        "| pairs/s | load s | VRAM (MB) | peak VRAM (MB) | fits budget "
        f"| recall delta vs {baseline or 'baseline'} | w/l/t | rand p | recall reading "
        "| MRR delta | MRR reading |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: | ---: "
        "| :-: | ---: | :-: | ---: | :-: |",
    ]
    for row in sorted(report["candidates"], key=_rank_key):
        delta, ledger, _sign_p, rand_p, reading = paired_cells(row, BAR_RECALL)
        mrr_delta, _l, _s, _r, mrr_reading = paired_cells(row, BAR_FIRST_HIT)
        label = "off (retrieval order)" if row["kind"] == KIND_RETRIEVAL_ORDER else row["model"]
        lines.append(
            f"| `{label}` | {row['family']} | {row['recall_at_k']:.3f} | {row['mrr']:.3f} "
            f"| {_cell(row['first_hit_rank_mean'], '.2f')} | {row['hit_items']} "
            f"| {row['rerank_ms_per_query']:.1f} | {row['pairs_per_second']:.1f} "
            f"| {_cell(row.get('load_seconds'), '.1f')} | {_cell(row.get('vram_mb'), '.0f')} "
            f"| {_cell(row.get('vram_peak_mb'), '.0f')} | {_fit_cell(row)} "
            f"| {delta} | {ledger} | {rand_p} | {reading} | {mrr_delta} | {mrr_reading} |"
        )
    return lines


def _recommendation(report: RerankBakeoffReport) -> list[str]:
    """The sentence an operator acts on: what to run, and what the swap costs per query."""
    rows = {row["model"]: row for row in report["candidates"]}
    verdict = report.get("verdict")
    named = verdict["model"] if verdict else report["best_recall"]
    chosen = rows.get(named or "")
    off = rows.get(ROW_NO_RERANK)
    lines = [
        f"Point-estimate leaders: recall@{report['k']} `{report['best_recall']}`, "
        f"MRR `{report['best_first_hit']}`. Apply a chosen reranker with "
        "`run-eval RERANKER=<model>` / `RunConfig.reranker`; `none` means leave reranking off.",
    ]
    if chosen is not None and off is not None:
        lines.append(
            f"Cost of running `{named}`: {chosen['rerank_ms_per_query']:.1f} ms per query and "
            f"{_cell(chosen.get('vram_peak_mb') or chosen.get('vram_mb'), '.0f')} MB of VRAM beside "
            f"the generator, for {chosen['recall_at_k'] - off['recall_at_k']:+.3f} recall@"
            f"{report['k']} and {chosen['mrr'] - off['mrr']:+.3f} MRR against no reranking."
        )
    return ["", *lines, ""]


def render_markdown(report: RerankBakeoffReport) -> str:
    """Durable `report.md`: the ranked table, the cost of each row, and the keep-or-swap call."""
    settings = report.get("uncertainty")
    baseline = settings["baseline"] if settings else None
    confidence = settings["confidence"] if settings else DEFAULT_CONFIDENCE
    lines = _header_lines(report)
    lines += _table(report, baseline)
    lines += ["", *verdict_lines(report, call_word="SWAP TO")]
    lines += _recommendation(report)
    lines += skipped_section(report.get("skipped") or [], title="Candidates not scored")
    lines += gate_summary(report["candidates"], confidence)
    lines += boundary_section(
        report["candidates"],
        baseline,
        confidence,
        title="How close each candidate sits to the adoption cut",
        key_header="reranker / bar",
        subject="the reranker",
    )
    lines += _floor_section(report)
    return "\n".join(lines)


def _floor_section(report: RerankBakeoffReport) -> list[str]:
    """The measurement floor the recommendation has to clear, when it was measured.

    The floor is read on each lane's OWN ranking scores at a scale-matched jitter, because two
    cross-encoder heads do not share a score scale (see `llb.rag.rerank_bakeoff.scoring`).
    """
    floor = report.get("noise_floor")
    if floor is None:
        return [
            "The measurement floor was not measured for this run; re-run with `--noise-floor` to",
            "state whether the recommended gap is larger than numeric noise.",
            "",
        ]
    from llb.rag.noise_floor_report import render_noise_floor_markdown

    return render_noise_floor_markdown(floor)
