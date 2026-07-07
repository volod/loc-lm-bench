"""Streamlit board and MLflow UI commands."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("board")
def board_cmd(
    host: str = typer.Option("127.0.0.1", help="network interface for the Streamlit board"),
    port: int = typer.Option(8501, min=1, max=65535, help="port for the Streamlit board"),
) -> None:
    """Serve the thin Streamlit leaderboard (rank + best-config-per-model + CIs)."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        typer.echo(
            '[error] the board needs the [board] extra. uv pip install -e ".[board]"', err=True
        )
        raise typer.Exit(code=2) from None
    from llb.board import app as board_app

    app_path = Path(board_app.__file__)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    raise typer.Exit(subprocess.call(cmd))


@app.command("recommend")
def recommend_cmd(
    run_root: Optional[Path] = typer.Option(
        None, help="run-eval bundle root (default: $DATA_DIR/run-eval)"
    ),
    out: Optional[Path] = typer.Option(
        None, help="Markdown summary path (default: $DATA_DIR/recommend/summary.md)"
    ),
    chart: Optional[Path] = typer.Option(
        None, help="comparison chart PNG path (default: $DATA_DIR/recommend/comparison.png)"
    ),
    json_out: Optional[Path] = typer.Option(None, help="machine-readable recommendation JSON path"),
    min_cases: int = typer.Option(
        1, help="drop bundles with fewer scored cases (filters partial/smoke runs)"
    ),
    gpu_gb: Optional[int] = typer.Option(
        None, help="host GPU tier override (12/16/24/32); default detects the host"
    ),
    min_tokens_per_s: float = typer.Option(
        0.0,
        "--min-tokens-per-s",
        help="good-enough-performance floor (tok/s) the host pick must clear on top of VRAM fit; "
        "0 = off",
    ),
    no_chart: bool = typer.Option(False, "--no-chart", help="skip rendering the comparison chart"),
) -> None:
    """Summarize a sweep into operator picks: best RAG accuracy, best efficiency, best for this host.

    Reads the final-split run bundles, ranks them, and writes a host-adaptive Markdown summary plus a
    model-comparison chart (needs the [viz] extra). The recommended-for-host pick is the
    highest-accuracy model that is Pareto-optimal and fits the GPU tier's VRAM budget with headroom.
    """
    from llb.board.recommend import (
        HostInfo,
        build_recommendation,
        format_config_detail_md,
        format_summary_md,
        load_config_cells,
        load_run_summaries,
        recommendation_payload,
    )
    from llb.inference.generate import resolve_tier
    from llb.core.paths import resolve_data_dir

    data_dir = resolve_data_dir()
    run_root = run_root or (data_dir / "run-eval")
    out = out or (data_dir / "recommend" / "summary.md")
    chart = chart or (data_dir / "recommend" / "comparison.png")

    summaries = load_run_summaries(run_root, min_cases=min_cases)
    if not summaries:
        typer.echo(
            f"[recommend] no final-split run bundles (>= {min_cases} cases) under {run_root}; "
            "run a sweep first",
            err=True,
        )
        raise typer.Exit(code=1)

    tier = resolve_tier(gpu_gb)
    # VRAM budget for the fit check: the measured card when detected, else the nominal tier size.
    # An explicit --gpu-gb override simulates that tier's budget, so the same bundles can be
    # re-recommended for a bigger/smaller CUDA host (e.g. would a 24 GiB box pick the 27B?).
    budget_mb = gpu_gb * 1024 if gpu_gb is not None else (tier.total_mb or tier.tier_gb * 1024)
    host = HostInfo(tier.tier_gb, budget_mb, tier.gpu_name, tier.detected)
    rec = build_recommendation(summaries, host, min_tokens_per_s=min_tokens_per_s)
    summary_md = format_summary_md(rec)
    # The per-configuration (model x top_k) proof: every config cell, not just best-per-model.
    detail_md = format_config_detail_md(load_config_cells(run_root, min_cases=min_cases))
    full_md = summary_md + ("\n\n" + detail_md if detail_md else "")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full_md + "\n", encoding="utf-8")
    typer.echo(full_md)
    typer.echo(f"\n[recommend] summary -> {out}")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(recommendation_payload(rec), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        typer.echo(f"[recommend] json -> {json_out}")

    if not no_chart:
        from llb.board.charts import render_comparison_chart

        rendered = render_comparison_chart(rec, chart)
        if rendered is not None:
            typer.echo(f"[recommend] comparison chart -> {rendered}")
        else:
            typer.echo("[recommend] chart skipped (install the [viz] extra for matplotlib)")


@app.command("mlflow-ui")
def mlflow_ui_cmd(
    host: str = typer.Option("127.0.0.1", help="network interface for the local MLflow UI"),
    port: int = typer.Option(5000, min=1, max=65535, help="port for the local MLflow UI"),
) -> None:
    """Serve the shared local MLflow experiment store."""
    from llb.tracking.server import run_mlflow_ui

    exit_code = run_mlflow_ui(host, port)
    if exit_code:
        raise typer.Exit(exit_code)
