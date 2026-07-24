"""Focused embedding bakeoff report implementation."""

from llb.rag.embedding_bakeoff import BakeoffReport, CandidateResult, _BYTES_PER_MB


def _throughput(row: CandidateResult) -> float:
    """Indexed chunks embedded per second (0.0 when unmeasured)."""
    return row["n_indexed"] / row["embed_seconds"] if row["embed_seconds"] > 0 else 0.0


def format_report(report: BakeoffReport) -> str:
    """ASCII summary table (AGENTS.md: ASCII-only, no box-drawing)."""
    rows = report["candidates"]
    lines = [f"[compare-embeddings] n={report['n']} k={report['k']}"]
    if not rows:
        lines.append("  (no candidates)")
        return "\n".join(lines)
    width = max(len(r["model"]) for r in rows)
    header = f"  {'model'.ljust(width)}   recall@k     mrr    dim   chunks/s   size_MB"
    lines.append(header)
    for row in sorted(rows, key=lambda c: (-c["recall_at_k"], -c["mrr"], c["model"])):
        size_mb = row["index_bytes"] / _BYTES_PER_MB
        lines.append(
            f"  {row['model'].ljust(width)}   {row['recall_at_k']:8.3f} {row['mrr']:7.3f} "
            f"{row['dim']:6d} {_throughput(row):9.1f} {size_mb:9.2f}"
        )
    lines.append(f"  best (recall@k): {report['best_recall']}")
    floor = report.get("noise_floor")
    if floor is not None:
        from llb.rag.noise_floor_report import format_noise_floor

        lines.extend(format_noise_floor(floor))
    return "\n".join(lines)


def render_markdown(report: BakeoffReport) -> str:
    """Durable `report.md`: a ranked table plus the applied-recommendation line."""
    lines = [
        "# Embedding bake-off (Ukrainian RAG)",
        "",
        f"- corpus: `{report['corpus_root']}`",
        f"- items scored: {report['n']}",
        f"- cutoff: recall@{report['k']} / MRR",
        "",
        "| model | kind | recall@k | MRR | dim | indexed | chunks/s | size (MB) | cost (USD) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        report["candidates"], key=lambda c: (-c["recall_at_k"], -c["mrr"], c["model"])
    ):
        cost = f"{row['cost_usd']:.4f}" if "cost_usd" in row else "-"
        size_mb = row["index_bytes"] / _BYTES_PER_MB
        lines.append(
            f"| `{row['model']}` | {row['kind']} | {row['recall_at_k']:.3f} | {row['mrr']:.3f} "
            f"| {row['dim']} | {row['n_indexed']} | {_throughput(row):.1f} | {size_mb:.2f} | {cost} |"
        )
    lines += [
        "",
        f"**Recommended embedder:** `{report['best_recall']}` (highest recall@{report['k']}; "
        "ties break by MRR then embed throughput). Apply it with "
        f"`build-index --embedding-model {report['best_recall']}` and set "
        "`RunConfig.embedding_model` to match.",
        "",
    ]
    lines += _floor_section(report)
    return "\n".join(lines)


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
