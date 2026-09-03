"""Validation, consent, and report persistence for the embedding bake-off CLI."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from llb.rag.embedding_bakeoff.verdict import resolve_bars
from llb.core.contracts.retrieval.comparison import SIDECAR_KIND_COMPARISON
from llb.rag.comparison.sidecar import write_sidecar

if TYPE_CHECKING:
    from llb.rag.encoders.throughput import ThroughputProfile


def resolved_bars(adoption_bars: str) -> Sequence[str]:
    try:
        return resolve_bars(adoption_bars)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None


def egress_consent(cfg: Any, api_model: str | None, assumed: bool) -> bool:
    if assumed:
        return True
    return typer.confirm(
        f"[compare-embeddings] embed the corpus at {cfg.corpus_root} through {api_model} "
        "(full corpus egress to a hosted API). Proceed?"
    )


def write_throughput_summary(
    profiles: list["ThroughputProfile"],
    *,
    baseline: str | None,
    data_dir: Path,
    run_ts: str,
) -> dict[str, Any]:
    from llb.rag.encoders.throughput_report import format_host_summary, render_host_markdown
    from llb.rag.encoders.throughput_summary import build_host_summary

    summary = build_host_summary(
        profiles,
        corpus_n_texts=max((profile["n_texts"] for profile in profiles), default=0),
        baseline_model=baseline,
    )
    throughput_dir = data_dir / "encoder-throughput" / run_ts
    throughput_dir.mkdir(parents=True, exist_ok=True)
    (throughput_dir / "report.md").write_text(render_host_markdown(summary), encoding="utf-8")
    write_sidecar(
        throughput_dir / "report.json", SIDECAR_KIND_COMPARISON, "encoder-throughput", summary
    )
    typer.echo(format_host_summary(summary))
    typer.echo(f"[encoder-throughput] wrote summary -> {throughput_dir}")
    return cast(dict[str, Any], summary)


def write_bakeoff_report(report: Any, report_path: Path) -> None:
    from llb.rag.embedding_bakeoff.report import format_report, render_markdown

    typer.echo(format_report(report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(report), encoding="utf-8")
    json_path = report_path.with_suffix(".json")
    write_sidecar(json_path, SIDECAR_KIND_COMPARISON, "compare-embeddings", report)
    typer.echo(f"[compare-embeddings] wrote report -> {report_path} ; {json_path}")
