"""`llb book-ocr pilot-run`: read the pilot with one engine at a time."""

from pathlib import Path

import typer
import yaml

from llb.cli.book import book_app
from llb.core.fsutil import atomic_write_text
from llb.core.paths import resolve_data_dir
from llb.prep.book.classes.records import read_selection
from llb.prep.book.compare.campaign import (
    campaign_lines,
    execute_campaign,
    prepare_campaign,
    roster_entry,
)
from llb.prep.book.compare.campaign_record import (
    CAMPAIGN_FILENAME,
    CampaignRecord,
    comparison_payload,
    read_campaign,
    write_campaign,
)
from llb.prep.book.compare.lane_config import (
    lane_request,
    pilot_pages,
    primary_counts,
    require_source,
)
from llb.prep.book.constants import METHOD_DIRNAME
from llb.prep.book.runner import BookRunError, check_request, new_run_dir

_EXIT_REFUSED = 2


@book_app.command("pilot-run")
def pilot_run_cmd(
    pdf: Path | None = typer.Argument(None, help="scanned PDF; required unless --resume"),
    pilot: Path | None = typer.Option(None, "--pilot", help="frozen pilot selection"),
    reuse_from: Path | None = typer.Option(
        None, "--reuse-from", help="classified run whose render and classes are reused"
    ),
    language: str = typer.Option("eng", "--language", help="book OCR language"),
    out_root: Path | None = typer.Option(None, "--out-root", help="parent of the run directories"),
    resume: Path | None = typer.Option(None, "--resume", help="campaign directory to continue"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the plan and write nothing"),
    reference: Path | None = typer.Option(None, "--reference", help="accepted reference directory"),
    predraft: Path | None = typer.Option(None, "--predraft", help="unedited pre-draft directory"),
    accepts: Path | None = typer.Option(None, "--accepts", help="accepts.json"),
    manifest_out: Path | None = typer.Option(
        None, "--manifest-out", help="write the compare manifest here"
    ),
) -> None:
    """Read each fitting engine over the pilot pages. Scoring is ``book-ocr compare``."""
    try:
        root = out_root or (resolve_data_dir() / METHOD_DIRNAME)
        bundle, record = _open(pdf, pilot, reuse_from, language, root, resume)
        errors = _preflight(record)
    except (OSError, ValueError, BookRunError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=_EXIT_REFUSED) from exc
    for line in campaign_lines(record):
        typer.echo(line)
    for error in errors:
        typer.echo(f"[error] {error}", err=True)
    if errors:
        raise typer.Exit(code=_EXIT_REFUSED)
    if dry_run:
        typer.echo("[book-ocr] dry run: pilot plan ok")
        return
    bundle.mkdir(parents=True, exist_ok=True)
    write_campaign(bundle, record)
    try:
        finished = execute_campaign(record, bundle, root)
    except (OSError, ValueError, BookRunError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=_EXIT_REFUSED) from exc
    typer.echo(f"[book-ocr] pilot -> {bundle / CAMPAIGN_FILENAME}")
    for line in campaign_lines(finished):
        typer.echo(line)
    _write_manifest(finished, reference, predraft, accepts, manifest_out)


def _open(
    pdf: Path | None,
    pilot: Path | None,
    reuse_from: Path | None,
    language: str,
    root: Path,
    resume: Path | None,
) -> tuple[Path, CampaignRecord]:
    if resume is not None:
        record = read_campaign(resume / CAMPAIGN_FILENAME)
        if pdf is not None:
            require_source(pdf, read_selection(Path(record.pilot)))
        return resume, record
    if pdf is None or pilot is None:
        raise ValueError("a new pilot run needs the PDF and --pilot")
    selection = read_selection(pilot)
    counts = primary_counts(reuse_from) if reuse_from is not None else {}
    from llb.prep.book.survey.measure import device_total_mb

    record = prepare_campaign(
        pdf=pdf,
        pilot=pilot,
        digest=require_source(pdf, selection),
        pages=pilot_pages(selection),
        language=language,
        reuse_from=reuse_from,
        host_mb=device_total_mb(),
        class_counts=counts,
    )
    _run_id, bundle = new_run_dir(root)
    return bundle, record


def _preflight(record: CampaignRecord) -> list[str]:
    errors: list[str] = []
    reuse = None if record.reuse_from is None else Path(record.reuse_from)
    source = Path(record.source)
    for lane in record.lanes:
        if lane.status != "pending":
            continue
        request = lane_request(source, record.pages, roster_entry(lane), record.language, reuse)
        plan = check_request(request)
        errors.extend(f"{lane.lane}: {item}" for item in plan.errors)
    return errors


def _write_manifest(
    record: CampaignRecord,
    reference: Path | None,
    predraft: Path | None,
    accepts: Path | None,
    manifest_out: Path | None,
) -> None:
    if reference is None and predraft is None and accepts is None and manifest_out is None:
        return
    if reference is None or predraft is None or accepts is None or manifest_out is None:
        typer.echo(
            "[error] manifest needs --reference, --predraft, --accepts, and --manifest-out",
            err=True,
        )
        raise typer.Exit(code=_EXIT_REFUSED)
    try:
        payload = comparison_payload(
            record, reference=str(reference), predraft=str(predraft), accepts=str(accepts)
        )
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=_EXIT_REFUSED) from exc
    atomic_write_text(manifest_out, yaml.safe_dump(payload, sort_keys=False))
    typer.echo(f"[book-ocr] manifest -> {manifest_out}")
