"""Adapter registry, serving, and lifecycle (gc) commands."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config

if TYPE_CHECKING:
    from llb.finetune.serving.model import ServeResult


@app.command("register-adapter")
def register_adapter_cmd(
    adapter_dir: Path = typer.Option(..., "--adapter-dir", help="directory holding the adapter"),
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="goldset the adapter was trained against"),
    corpus: Optional[Path] = typer.Option(None, "--corpus", help="corpus root used for training"),
    index_dir: Optional[Path] = typer.Option(
        None,
        "--index-dir",
        help="RAG store dir whose store_meta.json produced the training contexts "
        "(default: the config's index dir when it holds a store)",
    ),
    source_run: Optional[Path] = typer.Option(
        None, help="tuning run bundle that produced the data"
    ),
) -> None:
    """Register an adapter trained outside the loop, so the board can cite it.

    `self-improve` and `finetune-campaign` register their adapters automatically; a bare
    `finetune-adapter` does not, and an unregistered adapter never renders on the board.
    """
    from llb.finetune.registry.io import registry_path
    from llb.finetune.registry.register import register_adapter

    cfg = load_config(config, goldset_path=goldset, corpus_root=corpus)
    registry = registry_path(cfg.data_dir)
    resolved_index = index_dir if index_dir is not None else cfg.index_dir()
    try:
        entry = register_adapter(
            registry=registry,
            adapter_dir=adapter_dir,
            goldset_path=cfg.goldset_path if cfg.goldset_path.is_file() else None,
            corpus_root=cfg.corpus_root if cfg.corpus_root.is_dir() else None,
            index_dir=resolved_index if resolved_index.is_dir() else None,
            source_run=source_run,
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"[register-adapter] {entry.short_id} base={entry.base_model}")
    typer.echo(f"[register-adapter] registry -> {registry}")


@app.command("list-adapters")
def list_adapters_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config (locates DATA_DIR)"),
    json_out: bool = typer.Option(False, "--json", help="emit the rows as JSON"),
) -> None:
    """List registered adapters with base model, evidence, and staleness verdict."""
    from llb.finetune.registry.io import load_registry, registry_path
    from llb.finetune.registry.rows import adapter_rows

    cfg = load_config(config)
    rows = adapter_rows(load_registry(registry_path(cfg.data_dir)))
    if json_out:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        typer.echo(f"[list-adapters] no adapters registered under {registry_path(cfg.data_dir)}")
        return
    typer.echo(f"{'adapter':<14} {'staleness':<10} {'objective':<10} base model")
    for row in rows:
        objective = row.get("objective_score")
        score = f"{float(objective):.4f}" if isinstance(objective, int | float) else "n/a"
        typer.echo(
            f"{row['adapter_id']:<14} {row['staleness']:<10} {score:<10} {row['base_model']}"
        )
        for reason in row.get("reasons") or []:
            typer.echo(f"{'':<14} - {reason}")


@app.command("serve-adapter")
def serve_adapter_cmd(
    adapter: str = typer.Option(..., "--adapter", help="registered adapter id, prefix, or label"),
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    backend: Optional[str] = typer.Option(None, help="vllm | ollama | llamacpp"),
    smoke: bool = typer.Option(
        False, "--smoke", help="probe the endpoint once and exit instead of holding it open"
    ),
) -> None:
    """Serve a registered adapter: vLLM loads the LoRA directly, GGUF backends serve a merge."""
    from llb.finetune.serving.run import serve_adapter

    cfg = load_config(config, backend=backend)
    try:
        result = serve_adapter(
            cfg,
            adapter=adapter,
            backend=backend,
            hold=not smoke,
            on_ready=lambda ready: _echo_serving(ready, holding=not smoke),
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2) from None
    if result.probe_error:
        typer.echo(f"[serve-adapter] probe failed: {result.probe_error}", err=True)
        raise typer.Exit(code=1)


def _echo_serving(result: "ServeResult", *, holding: bool) -> None:
    """Report the live endpoint while the backend is still up (called before `--hold` blocks)."""
    typer.echo(
        f"[serve-adapter] adapter={result.adapter_id[:12]} backend={result.backend} "
        f"staleness={result.staleness.verdict}"
    )
    for reason in result.staleness.reasons:
        typer.echo(f"[serve-adapter] - {reason}")
    typer.echo(f"[serve-adapter] endpoint={result.endpoint} request-model={result.request_model}")
    if result.merged is not None:
        typer.echo(f"[serve-adapter] merged -> {result.merged.merged_dir}")
    if holding and result.probe_error is None:
        typer.echo("[serve-adapter] probe ok; serving in the foreground -- Ctrl-C to stop")


@app.command("gc-adapters")
def gc_adapters_cmd(
    config: Optional[Path] = typer.Option(None, help="YAML run config (locates DATA_DIR)"),
    force: bool = typer.Option(
        False, "--force", help="delete superseded adapters even when a run bundle cites them"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print decisions without deleting"),
) -> None:
    """Delete superseded adapters, never one a run bundle still cites (unless --force)."""
    from llb.finetune.lifecycle import gc_adapters, gc_rows

    cfg = load_config(config)
    plan = gc_adapters(data_dir=cfg.data_dir, force=force, dry_run=dry_run)
    for row in gc_rows(plan):
        typer.echo(f"[gc-adapters] {row['action']:<7} {row['adapter_id']:<14} {row['reason']}")
    verb = "would delete" if dry_run else "deleted"
    typer.echo(
        f"[gc-adapters] {verb}={len(plan.deleted)} refused={len(plan.refused)} "
        f"kept={len(plan.kept)}"
    )
