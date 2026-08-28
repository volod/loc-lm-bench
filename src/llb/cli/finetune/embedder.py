"""Embedder fine-tuning command (`finetune-embedder`): adapt the pinned encoder to a corpus."""

import json
from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.finetune.embedder.negatives import DEFAULT_NEGATIVES
from llb.finetune.embedder.pairs import DEFAULT_MAX_POSITIVES
from llb.finetune.embedder.trainer import KNOWN_TRAINERS, TRAINER_AUTO

TRAINER_HELP = (
    "auto (sentence-transformers contrastive training on this host) | fake (CI/control-plane "
    "smoke: validates the rows and writes a manifest, trains nothing)"
)


@app.command("finetune-embedder")
def finetune_embedder_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(
        None, "--goldset", help="gold set JSONL; only its verified TUNING items are read"
    ),
    corpus_root: Optional[Path] = typer.Option(
        None, "--corpus-root", help="corpus the positives and hard negatives are chunked from"
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base-model",
        help="encoder to fine-tune; defaults to the run config's embedding_model. Its registered "
        "query/passage convention is what the tuned model inherits",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", help="run dir (default: $DATA_DIR/finetune-embedder/<slug>/<ts>/)"
    ),
    seed: int = typer.Option(13, "--seed", help="export + training seed, recorded in the manifest"),
    trainer: str = typer.Option(TRAINER_AUTO, "--trainer", help=TRAINER_HELP),
    negatives: int = typer.Option(
        DEFAULT_NEGATIVES, "--negatives", min=1, help="hard negatives per training row"
    ),
    max_positives: int = typer.Option(
        DEFAULT_MAX_POSITIVES, "--max-positives", min=1, help="gold chunks trained per gold item"
    ),
    epochs: Optional[float] = typer.Option(None, "--epochs", help="override training epochs"),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="override the training batch size. In-batch negatives come from the batch, so a "
        "smaller value weakens the objective as well as the throughput",
    ),
    learning_rate: Optional[float] = typer.Option(
        None, "--learning-rate", help="override the training learning rate"
    ),
    mini_batch_size: Optional[int] = typer.Option(
        None,
        "--mini-batch-size",
        min=1,
        help="GradCache chunk for the cached loss: a pure VRAM knob, since the cached gradients "
        "are the uncached ones. Lower it when a host runs out of memory; --batch-size is what "
        "changes the objective",
    ),
) -> None:
    """Fine-tune the pinned embedder on this corpus, contrastively, from tuning-split gold only.

    Exports (question, gold-chunk, hard-negative) rows from the verified TUNING split -- positives
    are the chunks overlapping an item's source spans, hard negatives are BM25 rows for the same
    question that carry none of its evidence -- then trains the base encoder on them and writes a
    tuned model directory whose manifest records the base model, the dataset digest, the item ids,
    and the split counts. A calibration or final id anywhere in that dataset refuses the run.

    Measure the result with `compare-embeddings`, naming the tuned directory as a candidate beside
    the base encoder on the held-out FINAL split; the tuned row must clear zero against the base in
    the paired verdict to be worth adopting.
    """
    from llb.finetune.embedder.run import run_embedder_finetune

    cfg = load_config(config, goldset_path=goldset, corpus_root=corpus_root)
    model = base_model or cfg.embedding_model
    overrides = {
        key: value
        for key, value in (
            ("num_train_epochs", epochs),
            ("per_device_train_batch_size", batch_size),
            ("learning_rate", learning_rate),
            ("mini_batch_size", mini_batch_size),
        )
        if value is not None
    }
    try:
        result = run_embedder_finetune(
            cfg,
            base_model=model,
            out_dir=out_dir,
            seed=seed,
            trainer=trainer,
            negatives=negatives,
            max_positives=max_positives,
            hyperparameters=overrides or None,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    pairs, tuned = result.pairs_manifest, result.tuned_manifest
    typer.echo(
        f"[finetune-embedder] pairs={pairs['n_pairs']} items={len(pairs['item_ids'])} "
        f"skipped_without_positive={len(pairs['items_without_positive'])} "
        f"negatives={pairs['negatives_per_pair']}"
    )
    typer.echo(
        f"[finetune-embedder] trained {tuned['base_model']} ({tuned['convention_family']}) with "
        f"{tuned['trainer']}: {json.dumps(tuned['hyperparameters'], sort_keys=True)}"
    )
    typer.echo(f"[finetune-embedder] tuned embedder -> {result.model_dir}")
    typer.echo(
        "[finetune-embedder] measure it: make compare-embeddings SPLIT=final "
        f'MODELS="{tuned["base_model"]},{result.model_dir}" '
        f'EMBED_BASELINE="{tuned["base_model"]}"'
    )


__all__ = ["finetune_embedder_cmd", "KNOWN_TRAINERS"]
