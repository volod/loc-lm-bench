"""Local judge calibration experiment + smoke commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app


@app.command("judge-experiment")
def judge_experiment_cmd(
    judge_model: str = typer.Option(..., help="served local judge model id"),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
    ),
    data_dir: Optional[Path] = typer.Option(None, help="artifact root (default: DATA_DIR)"),
) -> None:
    """Run fixed Ukrainian judge sanity cases and record prompts plus scores."""
    from llb.judge.experiment import run_judge_experiment

    report, out_path = run_judge_experiment(
        judge_model,
        base_url=judge_base_url,
        data_dir=data_dir,
    )
    typer.echo(
        f"[judge-experiment] model={report['judge']['model']} "
        f"cases={len(report['cases'])} -> {out_path}"
    )


@app.command("judge-smoke")
def judge_smoke_cmd(
    judge_model: str = typer.Option(..., help="served local judge model id"),
    judge_base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
    ),
) -> None:
    """judge diagnostics: strict-JSON judge precheck. Run ONE grounded case and confirm the local judge returns
    a well-formed, non-zero score BEFORE a long judged run; exits non-zero (naming the reason) when
    the judge cannot emit strict JSON or its endpoint is unreachable."""
    from llb.judge.experiment import judge_smoke_check

    result = judge_smoke_check(judge_model, base_url=judge_base_url)
    if result.ok and result.score is not None:
        typer.echo(
            f"[judge-smoke] ok model={judge_model} "
            f"faithfulness={result.score['faithfulness']:.3f} "
            f"answer_relevancy={result.score['answer_relevancy']:.3f}"
        )
        return
    typer.echo(f"[judge-smoke] FAILED model={judge_model}: {result.reason}", err=True)
    raise typer.Exit(code=2)
