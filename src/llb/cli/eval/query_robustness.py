"""CLI for the Ukrainian noisy-query robustness benchmark."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config
from llb.eval.query_robustness_variants import (
    APOSTROPHE_MIXED_SCRIPT,
    VARIANT_CLASSES,
    parse_variant_classes,
)


def _inferred_corpus(goldset: Optional[Path], corpus_root: Optional[Path]) -> Optional[Path]:
    if corpus_root is not None or goldset is None:
        return corpus_root
    sibling = goldset.parent / "corpus"
    return sibling if sibling.is_dir() else None


@app.command("bench-query-robustness")
def bench_query_robustness_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    model: Optional[str] = typer.Option(None, help="model name (Ollama tag or HF repo id)"),
    backend: Optional[str] = typer.Option(None, help="ollama | vllm | llamacpp"),
    goldset: Optional[Path] = typer.Option(None, help="verified gold set JSONL"),
    corpus_root: Optional[Path] = typer.Option(
        None,
        help="matching indexed corpus; defaults to the gold set's sibling corpus directory",
    ),
    split: str = typer.Option("final", help="verified gold split to probe"),
    limit: Optional[int] = typer.Option(
        None, help="optional item cap; omitted runs the full split"
    ),
    seed: Optional[int] = typer.Option(None, help="deterministic noise seed"),
    typo_rate: float = typer.Option(
        0.08, help="share of eligible characters replaced in typo and mixed-script lanes"
    ),
    variant_classes: Optional[str] = typer.Option(
        None,
        "--variant-classes",
        help=(
            "comma-separated noise classes; default "
            f"{','.join(VARIANT_CLASSES)} (add {APOSTROPHE_MIXED_SCRIPT} for the combined class)"
        ),
    ),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="retrieved chunks per query"),
    max_tokens: Optional[int] = typer.Option(
        None, help="maximum answer tokens per clean or noisy case"
    ),
) -> None:
    """Measure clean-to-noisy RAG deltas under the off / normalize / normalize,typos lanes."""
    from llb.eval.query_robustness_run import run_query_robustness

    try:
        classes = parse_variant_classes(variant_classes) if variant_classes else None
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    cfg = load_config(
        config,
        model=model,
        backend=backend,
        goldset_path=goldset,
        corpus_root=_inferred_corpus(goldset, corpus_root),
        seed=seed,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    try:
        run = run_query_robustness(
            cfg,
            split=split,
            limit=limit,
            typo_rate=typo_rate,
            variant_classes=classes,
            progress=typer.echo,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"[query-robustness] clean baseline -> {run.clean_run_dir}")
    typer.echo(f"[query-robustness] report -> {run.paths['report']}")
    typer.echo(f"[query-robustness] rows -> {run.paths['robustness']}")
