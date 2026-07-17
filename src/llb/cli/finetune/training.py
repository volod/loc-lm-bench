"""Fine-tuning dataset export, adapter training, and hparam/compat commands."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.finetune.options import TRAINER_OPTION_HELP
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
    config: Optional[Path] = typer.Option(None, help="YAML run config (locates DATA_DIR)"),
    out: Optional[Path] = typer.Option(None, "--out", help="adapter output dir"),
    seed: int = typer.Option(13, "--seed", help="training seed recorded in adapter manifest"),
    trainer: str = typer.Option("auto", "--trainer", help=TRAINER_OPTION_HELP),
    searched_hparams: bool = typer.Option(
        True,
        "--searched-hparams/--default-hparams",
        help="use this model's recorded finetune-hparams best config as the trainer defaults",
    ),
) -> None:
    """Fine-tune a LoRA/QLoRA adapter behind the training seam."""
    from llb.finetune.hparam_search.manifest_io import trainer_defaults
    from llb.finetune.trainer import train_adapter

    cfg = load_config(config)
    defaults = trainer_defaults(cfg.data_dir, model) if searched_hparams else {}
    out_dir = out or (dataset.parent / "adapter")
    manifest = train_adapter(
        dataset_dir=dataset,
        model=model,
        out_dir=out_dir,
        seed=seed,
        trainer=trainer,
        **defaults,
    )
    if defaults:
        typer.echo(f"[finetune-adapter] hyperparameters <- {defaults['hparams_manifest']}")
    typer.echo(
        f"[finetune-adapter] adapter={manifest['adapter_label']} "
        f"digest={manifest['adapter_digest']}"
    )
    typer.echo(f"[finetune-adapter] manifest -> {out_dir / 'adapter_manifest.json'}")


@app.command("finetune-hparams")
def finetune_hparams_cmd(
    model: str = typer.Option(..., "--model", help="base local model id"),
    dataset: Path = typer.Option(
        ..., "--dataset", help="tuning-split dataset directory from export-finetune-set"
    ),
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(
        None, help="goldset the dev slice is scored against (and the split guard checks)"
    ),
    max_trials: int = typer.Option(8, "--max-trials", min=1, help="trial budget for the study"),
    max_hours: Optional[float] = typer.Option(
        None, "--max-hours", help="wall-clock budget; checked between trials, never mid-training"
    ),
    seed: int = typer.Option(13, "--seed", help="study + dev-slice seed"),
    dev_fraction: float = typer.Option(
        0.25, "--dev-fraction", help="share of the tuning split held out to score trials"
    ),
    trainer: str = typer.Option("auto", "--trainer", help=TRAINER_OPTION_HELP),
    out_dir: Optional[Path] = typer.Option(None, help="study output dir"),
    resume: Optional[Path] = typer.Option(None, help="resume a finetune-hparams study dir"),
    stratify_by_base_score: Optional[Path] = typer.Option(
        None,
        "--stratify-by-base-score",
        help="scored base-model run bundle (scores.jsonl); the dev slice is drawn "
        "proportionally per base-score bucket so it holds answerable items",
    ),
    vram_headroom_mib: Optional[float] = typer.Option(
        None,
        "--vram-headroom-mib",
        help="VRAM left beside the base model during training; a trial whose estimated "
        "adapter footprint exceeds it is pruned BEFORE the fine-tune runs",
    ),
) -> None:
    """Search the LoRA space for one model on a held-out dev slice of the tuning split."""
    from llb.finetune.hparam_search.search import search_hyperparameters

    cfg = load_config(config, model=model, backend=backend, goldset_path=goldset)
    result = search_hyperparameters(
        cfg,
        model=model,
        dataset_dir=dataset,
        max_trials=max_trials,
        max_hours=max_hours,
        seed=seed,
        dev_fraction=dev_fraction,
        trainer=trainer,
        out_dir=out_dir,
        resume=resume,
        goldset_path=goldset,
        stratify_by_base_score=stratify_by_base_score,
        vram_headroom_mib=vram_headroom_mib,
    )
    typer.echo(
        f"[finetune-hparams] trials={len(result.trials)}/{max_trials} "
        f"complete={result.n_complete} budget_exhausted={result.budget_exhausted}"
    )
    typer.echo(
        f"[finetune-hparams] dev slice: {len(result.dev_slice.train_ids)} train / "
        f"{len(result.dev_slice.dev_ids)} dev items (seed {result.dev_slice.seed})"
    )
    typer.echo(f"[finetune-hparams] manifest -> {result.manifest_path}")
    if result.best_hyperparameters is None:
        typer.echo(
            f"[finetune-hparams] no trial completed; resume with --resume {result.out_dir}",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(
        f"[finetune-hparams] best trial {result.best_trial} objective="
        f"{result.best_objective:.4f} config={json.dumps(result.best_hyperparameters, sort_keys=True)}"
    )


@app.command("finetune-compat")
def finetune_compat_cmd(
    model: str = typer.Option(..., "--model", help="checkpoint id to probe for trainability"),
    config: Optional[Path] = typer.Option(None, help="YAML run config (locates DATA_DIR)"),
    config_only: bool = typer.Option(
        False,
        "--config-only",
        help="stop after the cheap config-introspection stage (no weights are loaded)",
    ),
) -> None:
    """Probe whether a (possibly compressed-QAT) checkpoint can take a LoRA adapter on this host.

    Stage 1 classifies the checkpoint's native quantization scheme against PEFT's dispatch table
    (config only). Stage 2 loads the model, scans its linear classes, selects target modules from
    the modules that exist, attaches a rank-4 LoRA, and runs one forward/backward micro-step. The
    verdict + evidence land in `$DATA_DIR/finetune-compat/<model>/<timestamp>/compat_report.json`.
    """
    from llb.finetune.compat import (
        VERDICT_TRAINABLE,
        config_compat_probe,
        probe_trainability,
    )

    cfg = load_config(config, model=model)
    if config_only:
        verdict = config_compat_probe(model, local_only=False)
        typer.echo(
            f"[finetune-compat] {model}: {verdict['verdict']}"
            + (f" -- {verdict['blocker']}" if verdict.get("blocker") else "")
        )
        raise typer.Exit(code=0 if verdict["verdict"] == VERDICT_TRAINABLE else 1)
    report = probe_trainability(model, out_root=cfg.data_dir)
    typer.echo(
        f"[finetune-compat] {model}: {report['verdict']}"
        + (f" -- {report['blocker']}" if report.get("blocker") else "")
    )
    typer.echo(f"[finetune-compat] report -> {report['report_path']}")
    if report["verdict"] != VERDICT_TRAINABLE:
        raise typer.Exit(code=1)
