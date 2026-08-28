"""Upstream currency command: is each registered family's carried generation still the newest one?"""

from pathlib import Path

import typer

from llb.cli.app import app
from llb.cli.helpers import cli_error
from llb.cli.models.families import DEFAULT_MANIFEST


@app.command("check-model-currency")
def check_model_currency_cmd(
    manifest: Path = typer.Option(DEFAULT_MANIFEST, help="candidate-models YAML manifest"),
    family: str = typer.Option("", help="report only these family ids (comma-separated)"),
    replay: Path = typer.Option(
        None, help="replay recorded registry responses instead of reading upstream (offline)"
    ),
    record: Path = typer.Option(
        None, help="read upstream and record every response to this file for later replay"
    ),
    detail: bool = typer.Option(True, help="print what each registry answered under every row"),
    as_json: bool = typer.Option(False, "--json", help="print the report as JSON"),
    strict: bool = typer.Option(
        False, help="exit non-zero when any family is behind or unknown (for automation)"
    ),
) -> None:
    """Report each family's carried generation against the newest one upstream offers."""
    from llb.backends.currency import Cassette, live_fetch, probe_register, render_json, render_text
    from llb.backends.currency.report import CURRENT
    from llb.backends.roster import Register, load_register

    try:
        register = load_register(manifest)
    except ValueError as exc:
        cli_error(str(exc))

    wanted = {name.strip() for name in family.split(",") if name.strip()}
    if wanted:
        unknown_ids = wanted - {entry.id for entry in register.families}
        if unknown_ids:
            cli_error(f"no such family in the register: {', '.join(sorted(unknown_ids))}")
        register = Register(
            families=tuple(entry for entry in register.families if entry.id in wanted),
            models=register.models,
        )

    cassette = Cassette()
    if replay is not None:
        try:
            cassette = Cassette.load(replay)
        except (OSError, TypeError, ValueError) as exc:
            cli_error(f"{replay}: unreadable recorded responses -- {exc}")
        fetch = cassette.fetch
    elif record is not None:
        fetch = cassette.recording(live_fetch)
    else:
        fetch = live_fetch

    rows = probe_register(register, fetch)
    if record is not None:
        cassette.save(record)
        typer.echo(
            f"[currency] recorded {len(cassette.responses)} response(s) to {record}", err=True
        )

    typer.echo(render_json(rows) if as_json else render_text(rows, detail=detail))
    if strict and any(row.verdict != CURRENT for row in rows):
        raise typer.Exit(code=1)
