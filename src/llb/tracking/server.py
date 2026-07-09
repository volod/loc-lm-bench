"""Launch the local MLflow UI for the shared experiment store."""

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from llb.core.paths import resolve_data_dir
from llb.tracking.mlflow import MLFLOW_STORE_DIR, sync_mlflow_runs

_LOG = logging.getLogger(__name__)

MLFLOW_DATABASE = "mlflow.db"
MLFLOW_ARTIFACTS = "artifacts"
SHUTDOWN_TIMEOUT_S = 10.0


def build_mlflow_command(
    executable: Path,
    database: Path,
    artifact_root: Path,
    host: str,
    port: int,
) -> list[str]:
    """Build the local-only MLflow UI command."""
    return [
        str(executable),
        "ui",
        "--backend-store-uri",
        f"sqlite:///{database}",
        "--default-artifact-root",
        artifact_root.resolve().as_uri(),
        "--no-serve-artifacts",
        "--workers",
        "1",
        "--host",
        host,
        "--port",
        str(port),
    ]


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop MLflow and any descendants, escalating only after a bounded wait."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=SHUTDOWN_TIMEOUT_S)


def run_mlflow_ui(host: str, port: int) -> int:
    """Validate and supervise the shared-store MLflow UI process."""
    tracking_root = resolve_data_dir() / MLFLOW_STORE_DIR
    database = tracking_root / MLFLOW_DATABASE
    artifact_root = tracking_root / MLFLOW_ARTIFACTS
    executable = Path(sys.executable).with_name("mlflow")

    if not executable.is_file():
        raise SystemExit("MLflow CLI not found in .venv; run `make venv` with the track extra")
    report = sync_mlflow_runs()
    _LOG.info(
        "[mlflow] canonical sync: %d created, %d updated, %d current, %d failed",
        report["created"],
        report["updated"],
        report["current"],
        report["failed"],
    )
    if not database.is_file():
        raise SystemExit(f"MLflow database not found: {database}; run `make demo-eval` first")

    command = build_mlflow_command(executable, database, artifact_root, host, port)
    _LOG.info("[mlflow] experiment store: %s", database)
    _LOG.info("[mlflow] UI: http://%s:%d", host, port)
    os.environ.setdefault(
        "PYTHONWARNINGS",
        "ignore:starlette.middleware.wsgi is deprecated:UserWarning",
    )
    # The local review UI does not need MLflow's background GenAI job scheduler.
    os.environ.setdefault("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        _LOG.info("[mlflow] stopping UI")
        _stop_process_group(process)
        raise
