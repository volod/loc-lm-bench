"""`llb book-ocr survey`: list, install, and smoke the recovery-tool candidates."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from llb.cli.book import book_app

if TYPE_CHECKING:
    from llb.prep.book.survey.fixture import SmokePage

survey_app = typer.Typer(
    add_completion=False,
    rich_markup_mode=None,
    no_args_is_help=True,
    help="Recovery-tool survey: candidates per stage, isolated installs, one-page smokes.",
)
book_app.add_typer(survey_app, name="survey")

SYNTHETIC = "synthetic"
_LOG = logging.getLogger(__name__)


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@survey_app.command("list")
def list_cmd(
    stage: Optional[str] = typer.Option(None, help="only candidates serving this stage"),
) -> None:
    """Print every surveyed candidate: stages, version, licence, route, environment."""
    from llb.prep.book.survey.candidates import CANDIDATES

    for candidate in CANDIDATES:
        if stage and stage not in candidate.stages:
            continue
        where = candidate.env or candidate.route
        typer.echo(
            f"{candidate.id:32} {','.join(candidate.stages):40} {candidate.route:7} {where:8} "
            f"{candidate.version} [{candidate.licence}]"
        )


@survey_app.command("install")
def install_cmd(
    envs: str = typer.Option("all", "--envs", help="comma-separated environment names, or all"),
) -> None:
    """Create or refresh isolated survey environments under $DATA_DIR/venvs/book-<env>/."""
    from llb.core.paths import resolve_data_dir
    from llb.prep.book.survey.candidates import ENVS
    from llb.prep.book.survey.envs import SurveyEnvError, install_env

    names = sorted(ENVS) if envs == "all" else _split(envs)
    failed: list[str] = []
    for name in names:
        try:
            installed = install_env(resolve_data_dir(), name)
            typer.echo(f"[survey] {name}: ready ({installed.python}; freeze {installed.freeze})")
        except (SurveyEnvError, OSError) as exc:
            typer.echo(f"[survey] {name}: FAILED -- {exc}", err=True)
            failed.append(name)
    if failed:
        raise typer.Exit(code=1)


def _pages(specs: list[str], pages_dir: Path) -> list["SmokePage"]:
    from llb.prep.book.survey.fixture import write_fixture, write_real_page

    pages: list[SmokePage] = []
    for spec in specs:
        if spec == SYNTHETIC:
            pages.append(write_fixture(pages_dir / SYNTHETIC))
            continue
        pdf, sep, number = spec.rpartition(":")
        if not sep or not number.isdigit():
            raise typer.BadParameter(f"--page must be 'synthetic' or <pdf>:<page>, got {spec!r}")
        pages.append(write_real_page(Path(pdf), int(number), pages_dir / f"pdf-p{number}"))
    return pages


@survey_app.command("smoke")
def smoke_cmd(
    candidates: str = typer.Option("all", "--candidates", help="comma-separated ids, or all"),
    page: list[str] = typer.Option(
        [SYNTHETIC], "--page", help="'synthetic' or <pdf>:<page> (repeatable)"
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", help="run root (default $DATA_DIR/book-ocr)"
    ),
) -> None:
    """Smoke the candidates one at a time and write survey.json / survey.md."""
    from llb.core.paths import resolve_data_dir
    from llb.prep.book.constants import METHOD_DIRNAME
    from llb.prep.book.runner import new_run_dir
    from llb.prep.book.survey.candidates import CANDIDATES
    from llb.prep.book.survey.report import PAGES_DIRNAME, SUMMARY_MD, SURVEY_DIRNAME, run_survey

    data_dir = resolve_data_dir()
    known = {candidate.id: candidate for candidate in CANDIDATES}
    wanted = list(known) if candidates == "all" else _split(candidates)
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise typer.BadParameter(f"unknown candidate(s) {unknown}; see `llb book-ocr survey list`")
    _, run_dir = new_run_dir(out_dir or data_dir / METHOD_DIRNAME)
    survey_dir = run_dir / SURVEY_DIRNAME
    pages = _pages(page, survey_dir / PAGES_DIRNAME)
    results = run_survey([known[name] for name in wanted], pages, survey_dir, data_dir)
    for result in results:
        typer.echo(f"[survey] {result.candidate:32} {result.page:12} {result.status}")
    typer.echo(f"[survey] summary: {survey_dir / SUMMARY_MD}")


@survey_app.command("rescore")
def rescore_cmd(
    survey_dir: Path = typer.Argument(..., help="a run's survey/ directory"),
) -> None:
    """Recompute the synthetic-page smoke scores from the saved readings (no GPU, no re-run)."""
    from llb.prep.book.survey.report import SUMMARY_MD, rescore_survey

    results = rescore_survey(survey_dir)
    typer.echo(f"[survey] rescored {sum(1 for r in results if r.score)} reading(s)")
    typer.echo(f"[survey] summary: {survey_dir / SUMMARY_MD}")
