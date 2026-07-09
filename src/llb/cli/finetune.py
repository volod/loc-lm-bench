"""Fine-tuning and local self-improvement commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config


@app.command("export-finetune-set")
def export_finetune_set_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", help="finalized tuning run bundle"),
    goldset: Path = typer.Option(..., "--goldset", help="goldset JSONL used by the run"),
    out: Path = typer.Option(..., "--out", help="output dataset directory"),
    misses: Optional[Path] = typer.Option(
        None,
        "--misses",
        help="optional miss-analysis misses.jsonl; targeted misses get exported/weighted",
    ),
) -> None:
    """Export contamination-guarded SFT/DPO records from the tuning split."""
    from llb.finetune.dataset import export_finetune_set

    manifest = export_finetune_set(
        run_dir=run_dir,
        goldset_path=goldset,
        out_dir=out,
        misses_path=misses,
    )
    typer.echo(
        f"[export-finetune-set] sft={manifest['n_sft']} dpo={manifest['n_dpo']} "
        f"digest={manifest['dataset_digest']}"
    )
    typer.echo(f"[export-finetune-set] manifest -> {out / 'dataset_manifest.json'}")


@app.command("finetune-adapter")
def finetune_adapter_cmd(
    dataset: Path = typer.Option(
        ..., "--dataset", help="dataset directory from export-finetune-set"
    ),
    model: str = typer.Option(..., "--model", help="base local model id"),
    out: Optional[Path] = typer.Option(None, "--out", help="adapter output dir"),
    seed: int = typer.Option(13, "--seed", help="training seed recorded in adapter manifest"),
    trainer: str = typer.Option(
        "auto", "--trainer", help="auto (PEFT/TRL) | fake (CI/control-plane smoke)"
    ),
) -> None:
    """Fine-tune a LoRA/QLoRA adapter behind the training seam."""
    from llb.finetune.trainer import train_adapter

    out_dir = out or (dataset.parent / "adapter")
    manifest = train_adapter(
        dataset_dir=dataset,
        model=model,
        out_dir=out_dir,
        seed=seed,
        trainer=trainer,
    )
    typer.echo(
        f"[finetune-adapter] adapter={manifest['adapter_label']} "
        f"digest={manifest['adapter_digest']}"
    )
    typer.echo(f"[finetune-adapter] manifest -> {out_dir / 'adapter_manifest.json'}")


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
    trainer: str = typer.Option(
        "auto", "--trainer", help="auto (PEFT/TRL) | fake (CI/control-plane smoke)"
    ),
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
