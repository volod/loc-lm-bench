"""Focused embedding bakeoff report implementation."""

from llb.rag.bakeoff_report_sections import (
    NO_PAIRED_CELL as _NO_PAIRED_CELL,
)
from llb.rag.bakeoff_report_sections import (
    boundary_section,
    card_parity_section,
    gate_summary,
    paired_cells,
    skipped_section,
    verdict_lines,
)
from llb.rag.embedding_bakeoff_models import BYTES_PER_MB, BakeoffReport, CandidateResult
from llb.rag.embedding_bakeoff_uncertainty import (
    BAR_RECALL,
    DEFAULT_CONFIDENCE,
)
from llb.rag.fusion_evidence.power_report import power_summary


def _throughput(row: CandidateResult) -> float:
    """Indexed chunks embedded per second (0.0 when unmeasured)."""
    return row["n_indexed"] / row["embed_seconds"] if row["embed_seconds"] > 0 else 0.0


def _peak_vram_mb(row: CandidateResult) -> str:
    """Peak encoder VRAM from the warm decomposition, when `--encoder-throughput` measured it."""
    profile = row.get("throughput_profile")
    peak = profile.get("peak_vram_mb") if profile else None
    return f"{peak:.0f}" if peak else _NO_PAIRED_CELL


def _dtype_cell(row: CandidateResult) -> str:
    """The precision the row was MEASURED at, marked when it differs from the published one.

    Without it the chunks/s column compares checkpoints: a half-precision upload outruns a float32
    one at identical parameter count and dimension, which is a packaging fact, not a model fact.
    """
    measured = row.get("dtype")
    published = row.get("published_dtype")
    if measured is None:
        return published or _NO_PAIRED_CELL
    if published and published != measured:
        return f"{measured} (card {published})"
    return measured


def _family_cell(row: CandidateResult) -> str:
    """The convention the row was scored under, marked when it ran repo-supplied model code."""
    family = row.get("family", "-")
    return f"{family} (remote-code)" if row.get("trust_remote_code") else family


def _paired_cells(row: CandidateResult) -> tuple[str, str, str, str, str]:
    """The recall@k paired cells -- the metric this lane's unconditional adoption bar is read on."""
    return paired_cells(row, BAR_RECALL)


def _gate_summary(report: BakeoffReport) -> list[str]:
    """How many of the bake-off's per-bar paired readings the minimum-evidence gate relabeled."""
    settings = report.get("uncertainty")
    return gate_summary(
        report["candidates"], settings["confidence"] if settings else DEFAULT_CONFIDENCE
    )


def _boundary_section(report: BakeoffReport) -> list[str]:
    """How close each candidate's bar reading sits to the cut that produced it."""
    settings = report.get("uncertainty")
    return boundary_section(
        report["candidates"],
        settings["baseline"] if settings else None,
        settings["confidence"] if settings else DEFAULT_CONFIDENCE,
        title="How close each candidate sits to the adoption cut",
        key_header="candidate / bar",
        subject="the candidate",
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
        delta, ledger, _sign_p, _randomization_p, _reading = _paired_cells(row)
        lines.append(
            f"  {row['model'].ljust(width)}   {row['recall_at_k']:8.3f} {row['mrr']:7.3f} "
            f"{row['dim']:6d} {_throughput(row):9.1f} {size_mb:9.2f}   {delta:>22} {ledger:>12}"
        )
    lines.append(f"  best (recall@k): {report['best_recall']}")
    for skipped in report.get("skipped") or []:
        lines.append(f"  skipped: {skipped['model']} -- {skipped['detail']}")
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
    if analysis := report.get("power_analysis"):
        lines.append("")
        lines += power_summary(analysis, title="Predeclared paired-power contract")
    if settings is not None:
        lines.append(
            f"- paired uncertainty: baseline `{baseline}`, {settings['resamples']} resamples, "
            f"{settings['confidence']:.0%} percentile bootstrap, seed {settings['seed']}"
        )
        lines.append(f"- adoption bar(s): {', '.join(settings.get('bars') or [BAR_RECALL])}")
    lines += [
        "",
        "| model | family | kind | dtype | recall@k | MRR | dim | indexed | chunks/s | size (MB) "
        "| peak VRAM (MB) | cost (USD) "
        f"| recall delta vs {baseline or 'baseline'} | w/l/t | sign p | rand p "
        "| recall reading |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| :-: | ---: | ---: | :-: |",
    ]
    for row in sorted(
        report["candidates"], key=lambda c: (-c["recall_at_k"], -c["mrr"], c["model"])
    ):
        cost = f"{row['cost_usd']:.4f}" if "cost_usd" in row else "-"
        size_mb = row["index_bytes"] / BYTES_PER_MB
        delta, ledger, sign_p, randomization_p, reading = _paired_cells(row)
        lines.append(
            f"| `{row['model']}` | {_family_cell(row)} | {row['kind']} | {_dtype_cell(row)} "
            f"| {row['recall_at_k']:.3f} | {row['mrr']:.3f} "
            f"| {row['dim']} | {row['n_indexed']} | {_throughput(row):.1f} | {size_mb:.2f} "
            f"| {_peak_vram_mb(row)} | {cost} "
            f"| {delta} | {ledger} | {sign_p} | {randomization_p} | {reading} |"
        )
    lines += ["", *_verdict_lines(report), ""]
    lines += [
        f"Point-estimate leader (recall@{report['k']}; ties break by MRR then embed throughput): "
        f"`{report['best_recall']}`. Apply an ADOPTED embedder with "
        f"`build-index --embedding-model <model>` and set `RunConfig.embedding_model` to match.",
        "",
    ]
    lines += _skipped_section(report)
    lines += card_parity_section(report["candidates"])
    lines += _gate_summary(report)
    lines += _boundary_section(report)
    lines += _floor_section(report)
    lines += _throughput_section(report)
    return "\n".join(lines)


def _skipped_section(report: BakeoffReport) -> list[str]:
    """Roster entries that produced no row (declined `trust_remote_code`, unregistered id)."""
    return skipped_section(report.get("skipped") or [])


def _throughput_section(report: BakeoffReport) -> list[str]:
    """Cold/warm encoder decomposition appendix when `--encoder-throughput` ran."""
    summary = report.get("encoder_throughput")
    if not summary:
        return []
    from llb.rag.encoder_throughput_report import format_host_summary, render_host_markdown

    # Reuse the markdown table body (skip the H1) under an H2 in the bake-off report.
    md = render_host_markdown(summary)
    body = "\n".join(line for line in md.splitlines() if not line.startswith("# "))
    return ["", "## Encoder throughput decomposition", "", body, "", format_host_summary(summary)]


def _verdict_lines(report: BakeoffReport, prefix: str = "") -> list[str]:
    """The adopt-or-retain sentence, or a note that the run carries no paired reading."""
    return verdict_lines(report, prefix)


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
