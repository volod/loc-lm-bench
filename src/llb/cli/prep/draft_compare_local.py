"""Sequential local draft comparison and comparison.json analytics commands."""

import json
from pathlib import Path
from typing import Optional, cast

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error


@app.command("draft-compare-local")
def draft_compare_local_cmd(
    corpus_root: Path = typer.Option(..., help="directory of .md/.txt source documents"),
    seeds: int = typer.Option(12, min=1, help="exact shared seeds drafted by each model"),
    baseline_model: Optional[str] = typer.Option(None, help="Qwen override; default detects GPU"),
    probe_model: Optional[str] = typer.Option(None, help="Gemma override; default detects GPU"),
    out_dir: Optional[Path] = typer.Option(None, help="comparison artifact root"),
    base_url: Optional[str] = typer.Option(None, help="Ollama OpenAI-compatible base URL"),
    seed: int = typer.Option(13, help="deterministic seed selection and split seed"),
    max_tokens: int = typer.Option(4096, min=1),
    timeout: float = typer.Option(900.0, min=1.0),
) -> None:
    """Compare GPU-adaptive Qwen and Gemma models sequentially on one Ollama host."""
    from llb.prep.ontology.endpoint_builder import EndpointConfigBuilder
    from llb.prep.ontology.endpoint_config import DEFAULT_LOCAL_BASE_URL, EndpointConfig
    from llb.prep.ontology.local_compare import compare_local_drafters
    from llb.prep.ontology.local_compare_models import select_local_compare_models

    if not corpus_root.is_dir():
        cli_error(f"corpus root not found: {corpus_root}")
    resolved_url = base_url or DEFAULT_LOCAL_BASE_URL
    try:
        baseline, probe, num_ctx, selection = select_local_compare_models(
            resolved_url,
            baseline_model=baseline_model,
            probe_model=probe_model,
        )

        def endpoint(model: str) -> EndpointConfig:
            return EndpointConfigBuilder(
                kind="local",
                model=model,
                backend="ollama",
                base_url=resolved_url,
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=timeout,
                think=False,
                num_ctx=num_ctx,
            ).build()

        report = compare_local_drafters(
            corpus_root,
            endpoint(baseline),
            endpoint(probe),
            seeds=seeds,
            seed=seed,
            out_dir=out_dir,
            resource_selection=selection,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        cli_error(str(exc))
    typer.echo(
        f"[draft-compare-local] sequential models={baseline} -> {probe} "
        f"rankings={report['rankings']} -> {report['out_dir']}"
    )


@app.command("draft-compare-analyze")
def draft_compare_analyze_cmd(
    report: Path = typer.Option(..., help="comparison.json to analyze"),
    as_json: bool = typer.Option(False, "--json", help="emit normalized statistics as JSON"),
    require_passed_gates: bool = typer.Option(
        False, help="exit nonzero when any lane calibration gate failed"
    ),
) -> None:
    """Print lane metrics, deltas, execution order, and human-review progress."""
    from llb.prep.ontology.compare_analysis import (
        comparison_statistics,
        format_comparison_statistics,
        load_comparison,
    )

    if not report.is_file():
        cli_error(f"comparison report not found: {report}")
    try:
        stats = comparison_statistics(load_comparison(report))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        cli_error(f"cannot analyze comparison: {exc}")
    if as_json:
        typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        typer.echo(format_comparison_statistics(stats))
    lanes = cast(dict[str, dict[str, object]], stats["lanes"])
    if require_passed_gates and not all(
        bool(lane["calibration_passed"]) for lane in lanes.values()
    ):
        raise typer.Exit(code=1)
