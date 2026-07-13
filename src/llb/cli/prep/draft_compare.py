"""Shared-seed local-vs-frontier ontology drafting comparison command."""

from pathlib import Path
from typing import Optional, cast

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error
from llb.cli.prep.draft_endpoints import (
    _VllmLaunchOptions,
    _confirm_frontier_egress,
    _endpoint_config_setup,
)


def _comparison_report(comparison_root: Path) -> Path:
    report = comparison_root / "comparison.json"
    if not report.is_file():
        cli_error(f"comparison report not found: {report}")
    return report


@app.command("draft-compare-review")
def draft_compare_review_cmd(
    comparison_root: Path = typer.Option(..., help="comparison root containing comparison.json"),
    order: str = typer.Option("worksheet", help="worksheet | confidence"),
) -> None:
    """Interactively review both comparison lanes with the shared human verifier."""
    from llb.goldset.verify_session.loop import run_session
    from llb.prep.ontology.compare_gate import comparison_worksheets, worksheet_progress

    if order not in ("worksheet", "confidence"):
        cli_error("review order must be worksheet or confidence")
    report = _comparison_report(comparison_root)
    worksheets = comparison_worksheets(report)
    for number, (lane, worksheet) in enumerate(worksheets.items(), start=1):
        decided, total = worksheet_progress(worksheet)
        typer.echo(
            f"[draft-compare-review] session {number}/{len(worksheets)}: {lane} "
            f"({decided}/{total} decided)"
        )
        if total == 0:
            cli_error(f"{lane} worksheet is empty; inspect the lane calibration artifacts")
        if decided < total:
            run_session(worksheet, order=order)
        decided, total = worksheet_progress(worksheet)
        if decided < total:
            typer.echo(
                f"[draft-compare-review] saved {lane} at {decided}/{total}; "
                "re-run this command to resume"
            )
            raise typer.Exit(code=1)
    typer.echo(
        "[draft-compare-review] both worksheets complete; run make draft-compare-finalize "
        f"DRAFT_COMPARE_OUT_DIR={comparison_root}"
    )


@app.command("draft-compare-finalize")
def draft_compare_finalize_cmd(
    comparison_root: Path = typer.Option(..., help="comparison root containing comparison.json"),
) -> None:
    """Refresh reviewed metrics and mechanically check every comparison acceptance gate."""
    from llb.prep.ontology.compare_gate import finalize_comparison

    report_path = _comparison_report(comparison_root)
    try:
        report = finalize_comparison(report_path)
    except (KeyError, OSError, ValueError) as exc:
        cli_error(f"cannot finalize comparison: {exc}")
    finalization = cast(dict[str, object], report["finalization"])
    checks = cast(dict[str, bool], finalization["checks"])
    for name, passed in checks.items():
        typer.echo(f"[{'ok' if passed else 'fail'}] {name}")
    typer.echo(f"[draft-compare-finalize] report -> {report_path}")
    if not finalization["passed"]:
        raise typer.Exit(code=1)


@app.command("draft-compare-report")
def draft_compare_report_cmd(
    report: Path = typer.Option(..., help="existing comparison.json to update in place"),
    local_verification: Path = typer.Option(..., help="reviewed local verify_sample.csv"),
    frontier_verification: Path = typer.Option(..., help="reviewed frontier verify_sample.csv"),
) -> None:
    """Refresh reviewed accept rates without re-running either model lane."""
    from llb.prep.ontology.compare import refresh_comparison_acceptance

    for path in (report, local_verification, frontier_verification):
        if not path.is_file():
            cli_error(f"comparison artifact not found: {path}")
    result = refresh_comparison_acceptance(report, local_verification, frontier_verification)
    rankings = cast(dict[str, object], result["rankings"])
    typer.echo(f"[draft-compare-report] accept-rate={rankings['accept_rate']} -> {report}")


@app.command("draft-compare")
def draft_compare_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source documents"),
    seeds: int = typer.Option(..., min=1, help="number of exact shared seeds to draft per lane"),
    frontier_model: str = typer.Option(..., help="Litellm route for the frontier drafting lane"),
    local_model: str = typer.Option(..., help="local extraction and drafting model id"),
    local_backend: str = typer.Option("ollama", help="ollama | vllm | openai"),
    local_base_url: Optional[str] = typer.Option(None, help="running local endpoint base URL"),
    max_usd: Optional[float] = typer.Option(None, min=0.000001, help="measured frontier spend cap"),
    max_calls: int = typer.Option(100, min=1, help="hard frontier call cap"),
    seed: int = typer.Option(13, help="deterministic seed selection and split seed"),
    out_dir: Optional[Path] = typer.Option(
        None, help="report root (default: $DATA_DIR/draft-compare/<timestamp>/)"
    ),
    local_verification: Optional[Path] = typer.Option(
        None, help="reviewed local verify_sample.csv to include its accept rate"
    ),
    frontier_verification: Optional[Path] = typer.Option(
        None, help="reviewed frontier verify_sample.csv to include its accept rate"
    ),
    max_tokens: int = typer.Option(4096, min=1, help="completion token budget per call"),
    temperature: float = typer.Option(0.0, min=0.0),
    timeout: float = typer.Option(300.0, min=1.0),
    no_think: bool = typer.Option(
        True, "--no-think/--think", help="disable local hidden reasoning"
    ),
    num_ctx: Optional[int] = typer.Option(None, min=1),
    vllm_port: int = typer.Option(8000, min=1, max=65535),
) -> None:
    """Draft identical local-extracted seeds locally and through a consented frontier route."""
    from llb.prep.frontier_telemetry import DraftBudgetExceeded
    from llb.prep.ontology.compare import compare_drafters
    from llb.prep.ontology.endpoint_config import EndpointConfig

    if not corpus_root.is_dir():
        cli_error(f"corpus root not found: {corpus_root}")
    for worksheet in (local_verification, frontier_verification):
        if worksheet is not None and not worksheet.is_file():
            cli_error(f"verification worksheet not found: {worksheet}")
    _confirm_frontier_egress(corpus_root, frontier_model)
    options = _VllmLaunchOptions(
        port=vllm_port,
        gpu_memory_utilization=0.85,
        max_model_len=num_ctx,
        cpu_offload_gb=None,
        kv_offloading_size_gb=None,
        dtype="auto",
        quantization=None,
        startup_timeout=600.0,
    )
    local, launcher, resolved_out = _endpoint_config_setup(
        local_model,
        "local",
        local_backend,
        local_base_url,
        out_dir,
        num_ctx,
        options,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        no_think=no_think,
    )
    frontier = EndpointConfig(
        kind="frontier",
        model=frontier_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        egress_consent=True,
        max_usd=max_usd,
        max_calls=max_calls,
    )
    try:
        report = compare_drafters(
            corpus_root,
            local,
            frontier,
            seeds=seeds,
            seed=seed,
            out_dir=resolved_out,
            local_verification=local_verification,
            frontier_verification=frontier_verification,
        )
    except DraftBudgetExceeded as exc:
        cli_error(f"{exc.reason}; comparison artifacts remain inspectable", code=1)
    finally:
        if launcher is not None:
            launcher.stop()
    rankings = cast(dict[str, object], report["rankings"])
    typer.echo(
        f"[draft-compare] kept-yield={rankings['kept_yield']} "
        f"accept-rate={rankings['accept_rate']} -> {report['out_dir']}"
    )
