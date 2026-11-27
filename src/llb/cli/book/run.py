"""`llb book-ocr run`: start, dry-run, or resume a scanned-book recovery run."""

import logging
from pathlib import Path
from typing import Any, Optional

import typer

from llb.cli.book import book_app

_LOG = logging.getLogger(__name__)
_EXIT_REFUSED = 2
_EXIT_VALIDATE = 1


def _validate_vaults(run_dir: Path) -> None:
    """Re-check a vault this run already holds. A run with no vault is unchanged."""
    from llb.prep.book.render.constants import RENDER_MANIFEST_FILENAME, VAULT_DIRNAME
    from llb.prep.book.validate import validate_book

    root = run_dir / VAULT_DIRNAME
    if not root.is_dir():
        return
    failed = False
    for vault in sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    ):
        if not (vault / "metadata" / RENDER_MANIFEST_FILENAME).is_file():
            continue
        try:
            report = validate_book(vault, run_dir)
        except (OSError, ValueError) as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(code=_EXIT_VALIDATE) from exc
        state = "ok" if report.ok else "failed"
        typer.echo(f"[book-ocr] validate {state} -> {vault}")
        failed = failed or not report.ok
        for finding in report.findings:
            typer.echo(f"[book-ocr] {finding.file}: {finding.check}: {finding.detail}", err=True)
    if failed:
        raise typer.Exit(code=_EXIT_VALIDATE)


def _stage_overrides(raw: str | None, models: list[str], file_stages: Any) -> Any:
    """Stage specs from `--stages` (else the config file), with `--stage-model` applied."""
    from llb.core.contracts.data_prep.book import BookStageSpec
    from llb.prep.book.config import stage_specs

    specs = list(
        stage_specs([n.strip() for n in raw.split(",") if n.strip()]) if raw else file_stages or []
    )
    for item in models:
        name, sep, model = item.partition("=")
        if not sep or not model:
            raise ValueError(f"--stage-model {item!r} must be <stage>=<backend>:<name>")
        matches = [i for i, spec in enumerate(specs) if spec.name == name]
        if not matches:
            raise ValueError(f"--stage-model names stage {name!r}, which is not configured")
        specs[matches[0]] = BookStageSpec(name=name, params=specs[matches[0]].params, model=model)
    return tuple(specs) if specs else None


def _request(pdf: Path, config: Optional[Path], cli: dict[str, Any], models: list[str]) -> Any:
    from llb.prep.book.config import RunRequest, load_request_file

    values: dict[str, Any] = load_request_file(config) if config else {}
    stages = _stage_overrides(cli.pop("stages"), models, values.pop("stages", None))
    values.update({key: value for key, value in cli.items() if value is not None})
    if stages:
        values["stages"] = stages
    return RunRequest(source=pdf, **values)


def _report_preflight(plan: Any) -> None:
    if plan.sha256:
        typer.echo(f"[book-ocr] source sha256 {plan.sha256}, {plan.page_count} PDF pages")
    if plan.diagnostics is not None:
        diag = plan.diagnostics
        typer.echo(
            f"[book-ocr] embedded text chars {diag.embedded_text_chars}, "
            f"image-only pages {diag.image_only_pages}/{diag.page_count}"
        )
    if plan.pages:
        typer.echo(
            f"[book-ocr] {len(plan.pages)} page(s) selected in {len(plan.batches)} batch(es)"
        )
    for error in plan.errors:
        typer.echo(f"[error] {error}", err=True)


def _resume(run_dir: Path, pdf: Optional[Path], dry_run: bool, shaping: dict[str, Any]) -> None:
    from llb.prep.book.config import read_config, stage_errors
    from llb.prep.book.runner import resume_run
    from llb.prep.book.source import verify_source
    from llb.prep.book.stages import default_registry

    given = sorted(name for name, value in shaping.items() if value)
    if given:
        raise ValueError(
            f"--resume reads its configuration from {run_dir}/config.json; drop {given}"
        )
    if dry_run:
        config = read_config(run_dir)
        verify_source(run_dir, config.source, pdf)
        errors = stage_errors(config.stages, default_registry(), None)
        for error in errors:
            typer.echo(f"[error] {error}", err=True)
        typer.echo(f"[book-ocr] dry run: resume of {run_dir} {'refused' if errors else 'ok'}")
        if errors:
            raise typer.Exit(code=_EXIT_REFUSED)
        return
    _summarize(resume_run(run_dir, pdf))


def _summarize(outcome: Any) -> None:
    counts = ", ".join(f"{key} {value}" for key, value in sorted(outcome.outcomes.items()))
    typer.echo(f"[book-ocr] stage outcomes: {counts or 'nothing to do'}")
    for name, summary in outcome.manifest.summary.items():
        typer.echo(f"[book-ocr]   {name}: " + ", ".join(f"{k} {v}" for k, v in summary.items()))
    typer.echo(f"[book-ocr] run -> {outcome.run_dir}")
    failed = sum(summary.get("failed", 0) for summary in outcome.manifest.summary.values())
    if failed:
        typer.echo(f"[book-ocr] {failed} page stage(s) failed; see `llb book-ocr status`", err=True)
    _validate_vaults(outcome.run_dir)


@book_app.command("run")
def run_cmd(
    pdf: Optional[Path] = typer.Argument(
        None,
        help="scanned source PDF (optional with --resume, where it must match the recorded one)",
    ),
    pages: Optional[str] = typer.Option(
        None, help="1-based inclusive PDF pages, e.g. 1-20,37,900- (default: all)"
    ),
    batch_pages: Optional[int] = typer.Option(None, help="pages per batch (default 25)"),
    max_attempts: Optional[int] = typer.Option(
        None, help="attempts per page stage before it is recorded failed (default 2)"
    ),
    stages: Optional[str] = typer.Option(
        None, help="comma-separated stage names, in order (default: render)"
    ),
    stage_model: list[str] = typer.Option(
        [], "--stage-model", help="<stage>=<backend>:<name>, repeatable"
    ),
    config: Optional[Path] = typer.Option(
        None, help="YAML/JSON file: pages, batch_pages, max_attempts, source_policy, stages"
    ),
    source_policy: Optional[str] = typer.Option(None, help="copy | symlink (default) | reference"),
    reuse_from: Optional[Path] = typer.Option(
        None, help="earlier run whose stages are taken wherever the cache key matches"
    ),
    resume: Optional[Path] = typer.Option(
        None, help="continue this run directory under its recorded config"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, help="parent of the run directory (default: $DATA_DIR/book-ocr)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="validate source, ranges, config, and models; run nothing"
    ),
) -> None:
    """Inventory a scanned PDF, render its pages, and drive the configured stages page by page."""
    from llb.core.paths import resolve_data_dir
    from llb.prep.book.constants import METHOD_DIRNAME
    from llb.prep.book.runner import BookRunError, check_request, start_run

    shaping = {
        "pages": pages, "batch_pages": batch_pages, "max_attempts": max_attempts, "stages": stages,
        "stage_model": stage_model, "config": config, "source_policy": source_policy,
        "reuse_from": reuse_from, "out_dir": out_dir,
    }  # fmt: skip
    try:
        if resume is not None:
            _resume(resume, pdf, dry_run, shaping)
            return
        if pdf is None:
            raise ValueError("a source PDF is required unless --resume names a run")
        cli = {
            key: shaping[key]
            for key in (
                "pages",
                "batch_pages",
                "max_attempts",
                "stages",
                "source_policy",
                "reuse_from",
            )
        }
        request = _request(pdf, config, cli, stage_model)
        if dry_run:
            plan = check_request(request)
            _report_preflight(plan)
            typer.echo(f"[book-ocr] dry run {'ok' if plan.ok else 'refused'}; nothing was written")
            if not plan.ok:
                raise typer.Exit(code=_EXIT_REFUSED)
            return
        _summarize(start_run(request, out_dir or resolve_data_dir() / METHOD_DIRNAME))
    except BookRunError as exc:
        for error in exc.errors:
            typer.echo(f"[error] {error}", err=True)
        raise typer.Exit(code=_EXIT_REFUSED)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=_EXIT_REFUSED)
