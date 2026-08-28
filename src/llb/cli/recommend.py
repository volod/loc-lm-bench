"""Operator recommendation commands: the sweep summary and the composed agent operating profile.

`llb recommend` turns the run bundles into host-adaptive picks; `--agent-profile` additionally
composes every lane that has run on this host into ONE agent operating profile.
"""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


def _extra_sections_md(data_dir: Path, run_root: Path, min_cases: int) -> str:
    """The optional report sections, each rendering only when its flow has persisted an artifact.

    Every section is '' for an operator who never ran that flow, so the summary of a plain sweep is
    byte-identical to what it was before any of them existed.
    """
    from llb.board.miss_analysis.report import latest_analysis
    from llb.board.recommend.build import load_config_cells
    from llb.board.recommend.render import format_config_detail_md
    from llb.board.recommend.sections import (
        format_chain_context_section_md,
        format_finetune_campaign_section_md,
        format_miss_section_md,
        format_self_improvement_section_md,
        latest_chain_context,
        latest_finetune_campaign,
        latest_self_improvement,
    )

    sections = [
        # The per-configuration (model x top_k) proof: every config cell, not just best-per-model.
        format_config_detail_md(load_config_cells(run_root, min_cases=min_cases)),
        format_miss_section_md(latest_analysis(data_dir)),
        format_self_improvement_section_md(latest_self_improvement(data_dir)),
        format_finetune_campaign_section_md(latest_finetune_campaign(data_dir)),
        format_chain_context_section_md(latest_chain_context(data_dir)),
    ]
    return "".join("\n\n" + section for section in sections if section)


def _write_agent_profile(data_dir: Path, rec: object) -> None:
    """Compose and persist the agent operating profile beside the recommendation summary.

    A field whose lane never ran is reported as a gap, so this always writes an artifact -- the
    counts it echoes are the operator's cue for which lane to run next.
    """
    from llb.board.agent_profile.compose import build_agent_profile
    from llb.board.agent_profile.persist import write_profile
    from llb.board.agent_profile.render import STATE_ORDER

    profile = build_agent_profile(data_dir, rec)  # type: ignore[arg-type]
    paths = write_profile(profile, data_dir)
    counts = {state: 0 for state in STATE_ORDER}
    for item in profile.fields:
        counts[item.state] = counts.get(item.state, 0) + 1
    typer.echo(
        "[recommend] agent profile: "
        + " ".join(f"{state}={counts[state]}" for state in STATE_ORDER)
    )
    typer.echo(f"[recommend] agent profile -> {paths['json']}")
    typer.echo(f"[recommend] agent profile rationale -> {paths['markdown']}")


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
    agent_profile: bool = typer.Option(
        False,
        "--agent-profile",
        help="also compose ONE agent operating profile from every lane that has run on this host "
        "(model/backend, prompt system, adapter, context policy and order, retrieval knobs, loop "
        "policy) into $DATA_DIR/agent-profile/<run>/",
    ),
) -> None:
    """Summarize a sweep into operator picks: best RAG accuracy, best efficiency, best for this host.

    Reads the final-split run bundles, ranks them, and writes a host-adaptive Markdown summary plus a
    model-comparison chart (needs the [viz] extra). The recommended-for-host pick is the
    highest-accuracy model that is Pareto-optimal and fits the GPU tier's VRAM budget with headroom.
    """
    from llb.board.recommend.build import build_recommendation, load_run_summaries
    from llb.board.recommend.model import HostInfo
    from llb.board.recommend.render import format_summary_md, recommendation_payload
    from llb.core.paths import resolve_data_dir
    from llb.inference.generate import resolve_tier

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
    full_md = format_summary_md(rec) + _extra_sections_md(data_dir, run_root, min_cases)

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

    if agent_profile:
        _write_agent_profile(data_dir, rec)

    if not no_chart:
        from llb.board.charts import render_comparison_chart

        rendered = render_comparison_chart(rec, chart)
        if rendered is not None:
            typer.echo(f"[recommend] comparison chart -> {rendered}")
        else:
            typer.echo("[recommend] chart skipped (install the [viz] extra for matplotlib)")
