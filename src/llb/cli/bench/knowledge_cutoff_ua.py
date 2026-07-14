"""Ukrainian translation review and paired knowledge-cutoff commands."""

from pathlib import Path
from typing import Optional

import typer

from llb.cli.app import app
from llb.cli.bench._shared import _echo_throughput
from llb.cli.helpers import best_effort_gpu_readers, cli_error, load_config


@app.command("knowledge-cutoff-ua-draft")
def knowledge_cutoff_ua_draft_cmd(
    translator_model: str = typer.Option(..., help="local model used only to draft translations"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(None, help="running local OpenAI-compatible endpoint"),
    events: Optional[Path] = typer.Option(None, help="offline/local source event JSONL"),
    dataset_id: str = typer.Option("apoorvumang/knowledge-cutoff-benchmark"),
    dataset_revision: str = typer.Option("main", help="HF commit strongly recommended"),
    out_dir: Optional[Path] = typer.Option(None, help="translation bundle output directory"),
    limit: Optional[int] = typer.Option(None, min=1, help="smoke-only event cap"),
    max_model_len: Optional[int] = typer.Option(None, help="served context window"),
    gpu_memory_utilization: float = typer.Option(0.95, min=0.1, max=1.0),
) -> None:
    """Draft/resume source-aligned Ukrainian translations and emit a review worksheet."""
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.bench.knowledge_cutoff.data import LoadedEvents, load_events, select_events
    from llb.bench.knowledge_cutoff.translation import (
        TRANSLATION_MAX_TOKENS,
        draft_translation_bundle,
        translation_progress,
    )

    cfg = load_config(
        None,
        model=translator_model,
        backend=backend,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    try:
        loaded = load_events(
            path=events,
            dataset_id=dataset_id,
            revision=dataset_revision,
            cache_dir=cfg.data_dir / "cache" / "huggingface" / "datasets",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        cli_error(str(exc))
    loaded = LoadedEvents(select_events(loaded.events, limit), loaded.source)
    destination = out_dir or cfg.data_dir / "knowledge-cutoff-ua" / loaded.source.resolved_revision
    try:
        drafted, total = translation_progress(loaded, destination)
    except ValueError as exc:
        cli_error(str(exc))
    if drafted == total:

        def unexpected_complete(_prompt: str) -> str:
            raise AssertionError("complete draft must not call the translator")

        worksheet = draft_translation_bundle(
            loaded,
            complete=unexpected_complete,
            out_dir=destination,
            translator=translator_model,
        )
        typer.echo(f"[knowledge-cutoff-ua-draft] complete draft reused -> {worksheet}")
        return
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> Path:
        return draft_translation_bundle(
            loaded, complete=complete, out_dir=destination, translator=translator_model
        )

    try:
        worksheet = drive_with_backend(
            cfg,
            run,
            base_url=base_url,
            max_tokens=TRANSLATION_MAX_TOKENS,
            vram_reader=vram_reader,
            pid_usage_reader=pid_reader,
            meter=meter,
        )
    except (RuntimeError, ValueError) as exc:
        cli_error(str(exc))
    typer.echo(f"[knowledge-cutoff-ua-draft] review worksheet -> {worksheet}")
    _echo_throughput("knowledge-cutoff-ua-draft", meter)


@app.command("knowledge-cutoff-ua-review")
def knowledge_cutoff_ua_review_cmd(
    bundle: Path = typer.Option(..., help="translation bundle directory"),
    start: Optional[int] = typer.Option(None, min=1, help="one-based row to open"),
) -> None:
    """Open the shared resumable terminal review over every translated row."""
    from llb.bench.knowledge_cutoff.translation import WORKSHEET_FILENAME
    from llb.goldset.verify_session.loop import run_session

    worksheet = bundle / WORKSHEET_FILENAME
    if not worksheet.is_file():
        cli_error(f"translation worksheet not found: {worksheet}")
    decided = run_session(worksheet, start=start)
    typer.echo(f"[knowledge-cutoff-ua-review] decided={decided} -> {worksheet}")


@app.command("knowledge-cutoff-ua-revise")
def knowledge_cutoff_ua_revise_cmd(
    bundle: Path = typer.Option(..., help="translation bundle directory"),
    revisions: Path = typer.Option(..., help="JSONL replacement question/choices by item_id"),
) -> None:
    """Apply explicit draft corrections and rerun language/numeric/source gates."""
    from llb.bench.knowledge_cutoff.translation_revision import apply_translation_revisions

    try:
        changed = apply_translation_revisions(bundle, revisions)
    except (OSError, ValueError) as exc:
        cli_error(str(exc), code=1)
    typer.echo(f"[knowledge-cutoff-ua-revise] revised={changed} -> {bundle}")


@app.command("knowledge-cutoff-ua-freeze")
def knowledge_cutoff_ua_freeze_cmd(
    bundle: Path = typer.Option(..., help="translation bundle directory"),
    reviewer: str = typer.Option(..., help="bilingual reviewer's sign-off name or stable id"),
) -> None:
    """Validate all decisions/checks and freeze the accepted aligned language lanes."""
    from llb.bench.knowledge_cutoff.translation_review import freeze_reviewed_bundle

    try:
        summary = freeze_reviewed_bundle(bundle, reviewer=reviewer)
    except (OSError, RuntimeError, ValueError) as exc:
        cli_error(str(exc), code=1)
    typer.echo(
        f"[knowledge-cutoff-ua-freeze] accepted={summary['accepted_rows']} "
        f"excluded={summary['excluded_rows']} -> {bundle}"
    )


@app.command("knowledge-cutoff-ua-confirm-accepted")
def knowledge_cutoff_ua_confirm_accepted_cmd(
    bundle: Path = typer.Option(..., help="translation bundle directory"),
) -> None:
    """Confirm that prior aggregate accept decisions imply all four passing checks."""
    from llb.bench.knowledge_cutoff.translation_review import (
        confirm_accepted_translation_checks,
    )

    try:
        changed = confirm_accepted_translation_checks(bundle)
    except (OSError, ValueError) as exc:
        cli_error(str(exc), code=1)
    typer.echo(f"[knowledge-cutoff-ua-confirm-accepted] confirmed={changed} -> {bundle}")


@app.command("knowledge-cutoff-ua-validate")
def knowledge_cutoff_ua_validate_cmd(
    bundle: Path = typer.Option(..., help="translation bundle directory"),
) -> None:
    """Mechanically validate all draft assets and report review-gate progress."""
    from llb.bench.knowledge_cutoff.translation_review import review_bundle_status

    try:
        status = review_bundle_status(bundle)
    except (OSError, ValueError) as exc:
        cli_error(str(exc), code=1)
    typer.echo(
        f"[knowledge-cutoff-ua-validate] drafts={status['draft_rows']}/"
        f"{status['source_rows']} accepted={status['accepted_rows']} "
        f"excluded={status['excluded_rows']} undecided={status['undecided_rows']} "
        f"incomplete-accepted={status['incomplete_accepted_rows']}"
    )
    if not status["ready_to_freeze"]:
        raise typer.Exit(code=1)


@app.command("bench-knowledge-cutoff-bilingual")
def bench_knowledge_cutoff_bilingual_cmd(
    bundle: Path = typer.Option(..., help="frozen reviewed translation bundle"),
    model: str = typer.Option(..., help="candidate local model id"),
    backend: str = typer.Option("ollama", help="ollama | vllm | llamacpp"),
    base_url: Optional[str] = typer.Option(None, help="running local OpenAI-compatible endpoint"),
    threshold: float = typer.Option(0.5, min=0.01, max=1.0),
    optuna_trials: int = typer.Option(200, min=1),
    seed: int = typer.Option(42, help="fit and paired-bootstrap seed"),
    max_model_len: Optional[int] = typer.Option(None, help="served context window"),
    gpu_memory_utilization: float = typer.Option(0.9, min=0.1, max=1.0),
) -> None:
    """Run aligned English/Ukrainian lanes and emit one paired cutoff report."""
    from llb.bench.common import LLMComplete
    from llb.bench.common_backend import ThroughputMeter, drive_with_backend
    from llb.bench.knowledge_cutoff.paired import BilingualCutoffRun, run_bilingual_cutoff

    cfg = load_config(
        None,
        model=model,
        backend=backend,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    vram_reader, pid_reader = best_effort_gpu_readers()
    meter = ThroughputMeter()

    def run(complete: LLMComplete) -> BilingualCutoffRun:
        return run_bilingual_cutoff(
            bundle,
            model=model,
            backend=backend,
            complete=complete,
            data_dir=cfg.data_dir,
            threshold=threshold,
            optuna_trials=optuna_trials,
            seed=seed,
            meter=meter,
        )

    try:
        result = drive_with_backend(
            cfg,
            run,
            base_url=base_url,
            max_tokens=16,
            vram_reader=vram_reader,
            pid_usage_reader=pid_reader,
            meter=meter,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        cli_error(str(exc), code=1)
    typer.echo(
        f"[bench-knowledge-cutoff-bilingual] delta={result.paired['accuracy_delta']:.3f} "
        f"en-cutoff={result.english.fit.effective_cutoff or 'unavailable'} "
        f"uk-cutoff={result.ukrainian.fit.effective_cutoff or 'unavailable'}"
    )
    _echo_throughput("bench-knowledge-cutoff-bilingual", meter)
    if result.paths is not None:
        typer.echo(
            f"[bench-knowledge-cutoff-bilingual] report -> "
            f"{Path(result.paths['manifest']).parent / 'report.md'}"
        )
