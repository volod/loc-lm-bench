"""Distillation, local self-improvement, and campaign commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.finetune.options import TRAINER_OPTION_HELP
from llb.cli.helpers import load_config, planning_models


@app.command("distill")
def distill_cmd(
    teacher: str = typer.Option(..., "--teacher", help="local teacher model id"),
    student: str = typer.Option(..., "--student", help="student base model id to fine-tune"),
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL"),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", help="corpus root used to build the retrieval index"
    ),
    gate: float = typer.Option(0.8, "--gate", min=0.0, max=1.0, help="teacher answer F1 gate"),
    limit: Optional[int] = typer.Option(None, help="cap tuning teacher items for smoke runs"),
    compare_split: str = typer.Option(
        "final", "--compare-split", help="split used for distilled-vs-reference comparison"
    ),
    compare_limit: Optional[int] = typer.Option(
        None, "--compare-limit", help="cap comparison items for smoke runs"
    ),
    out_dir: Optional[Path] = typer.Option(None, help="distillation output dir"),
    trainer: str = typer.Option("auto", "--trainer", help=TRAINER_OPTION_HELP),
) -> None:
    """Distill tuning-split teacher answers into a student LoRA adapter."""
    from llb.finetune.distill.run import run_distillation

    cfg = load_config(
        config, model=student, backend=backend, goldset_path=goldset, corpus_root=corpus
    )
    result = run_distillation(
        cfg,
        teacher=teacher,
        student=student,
        gate=gate,
        out_dir=out_dir,
        trainer=trainer,
        limit=limit,
        compare_split=compare_split,
        compare_limit=compare_limit,
    )
    typer.echo(
        f"[distill] accepted={result.accepted} rejected={result.rejected} "
        f"delta={result.comparison.delta:.4f}"
    )
    typer.echo(f"[distill] adapter -> {result.adapter_dir}")
    typer.echo(f"[distill] report -> {result.report_path}")


@app.command("self-improve")
def self_improve_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL"),
    rounds: int = typer.Option(2, min=1, help="maximum adapter rounds"),
    limit: Optional[int] = typer.Option(None, help="cap eval items per split for smoke runs"),
    out_dir: Optional[Path] = typer.Option(None, help="campaign output dir"),
    resume: Optional[Path] = typer.Option(None, help="resume a self-improve campaign dir"),
    trainer: str = typer.Option("auto", "--trainer", help=TRAINER_OPTION_HELP),
    min_gain: float = typer.Option(
        0.0, "--min-gain", help="minimum final-split objective delta before accepting a round"
    ),
) -> None:
    """Chain tuning eval -> miss analysis -> export -> fine-tune -> final eval per round."""
    from llb.finetune.loop import run_self_improve

    cfg = load_config(config, model=model, backend=backend, goldset_path=goldset)
    result = run_self_improve(
        cfg,
        rounds=rounds,
        out_dir=out_dir,
        resume=resume,
        trainer=trainer,
        limit=limit,
        min_gain=min_gain,
    )
    typer.echo(
        f"[self-improve] verdict={result.verdict} rounds={len(result.rounds)} "
        f"base_final={result.base_final_run_dir}"
    )
    typer.echo(f"[self-improve] report -> {result.out_dir / 'report.md'}")


@app.command("finetune-campaign")
def finetune_campaign_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    models: str = typer.Option(..., "--models", help="comma-separated local model roster"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="gold set JSONL"),
    corpus: Optional[Path] = typer.Option(
        None, "--corpus", help="corpus root used to build RAG index"
    ),
    rounds: int = typer.Option(1, min=1, help="adapter rounds per feasible roster entry"),
    limit: Optional[int] = typer.Option(None, help="cap eval items per split for smoke runs"),
    out_dir: Optional[Path] = typer.Option(None, help="campaign output dir"),
    resume: Optional[Path] = typer.Option(None, help="resume a finetune-campaign dir"),
    trainer: str = typer.Option("auto", "--trainer", help=TRAINER_OPTION_HELP),
    manifest: Optional[Path] = typer.Option(
        None, "--manifest", help="optional model manifest for feasibility planning"
    ),
) -> None:
    """Run the local self-improvement loop across a roster and rank tunability."""
    from llb.finetune.campaign.run import run_finetune_campaign

    cfg = load_config(config, backend=backend, goldset_path=goldset, corpus_root=corpus)
    specs = planning_models(manifest) if manifest is not None else None
    result = run_finetune_campaign(
        cfg,
        models=[models],
        rounds=rounds,
        out_dir=out_dir,
        resume=resume,
        trainer=trainer,
        limit=limit,
        model_specs=specs,
    )
    completed = sum(1 for entry in result.entries if entry.status == "completed")
    skipped = sum(1 for entry in result.entries if entry.status == "skipped")
    typer.echo(
        f"[finetune-campaign] completed={completed} skipped={skipped} entries={len(result.entries)}"
    )
    typer.echo(f"[finetune-campaign] report -> {result.out_dir / 'report.md'}")
