"""Focused embedding bakeoff report implementation."""

from llb.rag.embedding_bakeoff_models import BYTES_PER_MB, BakeoffReport, CandidateResult
from llb.rag.embedding_bakeoff_uncertainty import (
    BARS,
    BAR_RECALL,
    bar_stability,
    recall_delta,
)
from llb.rag.embedding_bakeoff_verdict import (
    DECISION_ADOPT,
)
from llb.rag.fusion_evidence.stability import boundary_table, format_reading
from llb.rag.fusion_evidence.stats import format_interval

_NO_PAIRED_CELL = "-"


def _throughput(row: CandidateResult) -> float:
    """Indexed chunks embedded per second (0.0 when unmeasured)."""
    return row["n_indexed"] / row["embed_seconds"] if row["embed_seconds"] > 0 else 0.0


def _paired_cells(row: CandidateResult) -> tuple[str, str, str, str]:
    """`delta [lo, hi]`, `w/l/t`, `sign p`, reading for one row (dashes when there is no baseline).

    The reading column is what keeps a `flat` that missed by a mile from printing exactly like one
    that missed by nothing -- the whole point of the borderline annotation.
    """
    paired = row.get("paired_vs_baseline")
    if paired is None:
        return (_NO_PAIRED_CELL,) * 4
    delta = recall_delta(paired)
    stability = bar_stability(paired, BAR_RECALL)
    return (
        format_interval(delta["delta"]),
        f"{delta['wins']}/{delta['losses']}/{delta['ties']}",
        f"{delta['sign_test_p']:.3f}",
        format_reading(stability, stability["reading"]) if stability else _NO_PAIRED_CELL,
    )


def _boundary_section(report: BakeoffReport) -> list[str]:
    """How close each candidate's bar reading sits to the cut that produced it."""
    settings = report.get("uncertainty")
    baseline = settings["baseline"] if settings else None
    rows = []
    for row in sorted(report["candidates"], key=lambda c: c["model"]):
        paired = row.get("paired_vs_baseline")
        if paired is None or row["model"] == baseline:
            continue
        for bar in BARS:
            stability = bar_stability(paired, bar)
            if stability is not None:
                rows.append((f"`{row['model']}` {bar}", stability))
    return boundary_table(
        rows,
        title="How close each candidate sits to the adoption cut",
        key_header="candidate / bar",
        subject="the candidate",
        confidence=settings["confidence"] if settings else 0.95,
    )


def format_report(report: BakeoffReport) -> str:
    """ASCII summary table (AGENTS.md: ASCII-only, no box-drawing)."""
    rows = report["candidates"]
    lines = [f"[compare-embeddings] n={report['n']} k={report['k']}"]
    if not rows:
        lines.append("  (no candidates)")
        return "\n".join(lines)
    width = max(len(r["model"]) for r in rows)
    header = (
        f"  {'model'.ljust(width)}   recall@k     mrr    dim   chunks/s   size_MB"
        "   d_recall vs baseline        w/l/t"
    )
    lines.append(header)
    for row in sorted(rows, key=lambda c: (-c["recall_at_k"], -c["mrr"], c["model"])):
        size_mb = row["index_bytes"] / BYTES_PER_MB
        delta, ledger, _p, _reading = _paired_cells(row)
        lines.append(
            f"  {row['model'].ljust(width)}   {row['recall_at_k']:8.3f} {row['mrr']:7.3f} "
            f"{row['dim']:6d} {_throughput(row):9.1f} {size_mb:9.2f}   {delta:>22} {ledger:>12}"
        )
    lines.append(f"  best (recall@k): {report['best_recall']}")
    lines.extend(_verdict_lines(report, prefix="  "))
    floor = report.get("noise_floor")
    if floor is not None:
        from llb.rag.noise_floor_report import format_noise_floor

        lines.extend(format_noise_floor(floor))
    return "\n".join(lines)


def render_markdown(report: BakeoffReport) -> str:
    """Durable `report.md`: the ranked table, the paired intervals, and the adopt-or-retain call."""
    settings = report.get("uncertainty")
    baseline = settings["baseline"] if settings else None
    lines = [
        "# Embedding bake-off (Ukrainian RAG)",
        "",
        f"- corpus: `{report['corpus_root']}`",
        f"- items scored: {report['n']}",
        f"- cutoff: recall@{report['k']} / MRR",
    ]
    if settings is not None:
        lines.append(
            f"- paired uncertainty: baseline `{baseline}`, {settings['resamples']} resamples, "
            f"{settings['confidence']:.0%} percentile bootstrap, seed {settings['seed']}"
        )
        lines.append(f"- adoption bar(s): {', '.join(settings.get('bars') or [BAR_RECALL])}")
    lines += [
        "",
        "| model | kind | recall@k | MRR | dim | indexed | chunks/s | size (MB) | cost (USD) "
        f"| recall delta vs {baseline or 'baseline'} | w/l/t | sign p | recall reading |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: | ---: | :-: |",
    ]
    for row in sorted(
        report["candidates"], key=lambda c: (-c["recall_at_k"], -c["mrr"], c["model"])
    ):
        cost = f"{row['cost_usd']:.4f}" if "cost_usd" in row else "-"
        size_mb = row["index_bytes"] / BYTES_PER_MB
        delta, ledger, sign_p, reading = _paired_cells(row)
        lines.append(
            f"| `{row['model']}` | {row['kind']} | {row['recall_at_k']:.3f} | {row['mrr']:.3f} "
            f"| {row['dim']} | {row['n_indexed']} | {_throughput(row):.1f} | {size_mb:.2f} | {cost} "
            f"| {delta} | {ledger} | {sign_p} | {reading} |"
        )
    lines += ["", *_verdict_lines(report), ""]
    lines += [
        f"Point-estimate leader (recall@{report['k']}; ties break by MRR then embed throughput): "
        f"`{report['best_recall']}`. Apply an ADOPTED embedder with "
        f"`build-index --embedding-model <model>` and set `RunConfig.embedding_model` to match.",
        "",
    ]
    lines += _boundary_section(report)
    lines += _floor_section(report)
    return "\n".join(lines)


def _verdict_lines(report: BakeoffReport, prefix: str = "") -> list[str]:
    """The adopt-or-retain sentence, or a note that the run carries no paired reading."""
    verdict = report.get("verdict")
    if verdict is None:
        return [
            f"{prefix}No paired uncertainty was computed for this run, so the ranking above is a "
            "point estimate only."
        ]
    call = "ADOPT" if verdict["decision"] == DECISION_ADOPT else verdict["decision"].upper()
    named = f" `{verdict['model']}`" if verdict["model"] else ""
    return [f"{prefix}Verdict: {call}{named} -- {verdict['reason']}."]


def _floor_section(report: BakeoffReport) -> list[str]:
    """The measurement floor the recommendation above has to clear, when it was measured.

    A bake-off ranks four candidates on ONE corpus, so the gap between the winner and the runner-up
    is routinely worth a single item; without the floor beside it there is no way to tell a real
    ranking from tie order.
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
