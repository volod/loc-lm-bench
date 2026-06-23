"""Streamlit board and MLflow UI commands."""

import subprocess
import sys
from pathlib import Path

import typer

from llb.cli.app import app


@app.command("board")
def board_cmd(
    host: str = typer.Option("127.0.0.1", help="network interface for the Streamlit board"),
    port: int = typer.Option(8501, min=1, max=65535, help="port for the Streamlit board"),
) -> None:
    """Serve the thin Streamlit leaderboard (rank + best-config-per-model + CIs)."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        typer.echo(
            '[error] the board needs the [board] extra. uv pip install -e ".[board]"', err=True
        )
        raise typer.Exit(code=2) from None
    from llb.board import app as board_app

    app_path = Path(board_app.__file__)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    raise typer.Exit(subprocess.call(cmd))


@app.command("mlflow-ui")
def mlflow_ui_cmd(
    host: str = typer.Option("127.0.0.1", help="network interface for the local MLflow UI"),
    port: int = typer.Option(5000, min=1, max=65535, help="port for the local MLflow UI"),
) -> None:
    """Serve the shared local MLflow experiment store."""
    from llb.tracking.server import run_mlflow_ui

    exit_code = run_mlflow_ui(host, port)
    if exit_code:
        raise typer.Exit(exit_code)
