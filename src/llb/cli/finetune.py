"""Fine-tuning, adapter lifecycle, and local self-improvement commands."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config, planning_models

if TYPE_CHECKING:
    from llb.finetune.serving import ServeResult


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
    trainer: str = typer.Option(
        "auto", "--trainer", help="auto (PEFT/TRL) | fake (CI/control-plane smoke)"
    ),
    manifest: Optional[Path] = typer.Option(
        None, "--manifest", help="optional model manifest for feasibility planning"
    ),
) -> None:
    """Run the local self-improvement loop across a roster and rank tunability."""
    from llb.finetune.campaign import run_finetune_campaign

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


@app.command("register-adapter")
def register_adapter_cmd(
    adapter_dir: Path = typer.Option(..., "--adapter-dir", help="directory holding the adapter"),
    config: Optional[Path] = typer.Option(None, help="YAML run config"),
    goldset: Optional[Path] = typer.Option(None, help="goldset the adapter was trained against"),
    corpus: Optional[Path] = typer.Option(None, "--corpus", help="corpus root used for training"),
    source_run: Optional[Path] = typer.Option(
        None, help="tuning run bundle that produced the data"
    ),
) -> None:
    """Register an adapter trained outside the loop, so the board can cite it.

    `self-improve` and `finetune-campaign` register their adapters automatically; a bare
    `finetune-adapter` does not, and an unregistered adapter never renders on the board.
    """
    from llb.finetune.registry import register_adapter, registry_path

    cfg = load_config(config, goldset_path=goldset, corpus_root=corpus)
    registry = registry_path(cfg.data_dir)
    try:
        entry = register_adapter(
            registry=registry,
            adapter_dir=adapter_dir,
            goldset_path=cfg.goldset_path if cfg.goldset_path.is_file() else None,
            corpus_root=cfg.corpus_root if cfg.corpus_root.is_dir() else None,
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
    from llb.finetune.registry import adapter_rows, load_registry, registry_path

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
    from llb.finetune.serving import serve_adapter

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
