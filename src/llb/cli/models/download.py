"""CLI for bounded, resumable open-model downloads."""

from pathlib import Path
from typing import Optional

import typer

from llb.backends.model_download import DownloadConfig, download_model
from llb.backends.model_download.contracts import BYTES_PER_GIB, BYTES_PER_MIB, DownloadError
from llb.cli.app import app


def _bytes(value: float | None, unit: int) -> int | None:
    return None if value is None or value == 0 else int(value * unit)


def _human_bytes(value: int) -> str:
    amount = float(value)
    suffixes = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for suffix in suffixes:
        if amount < 1024 or suffix == suffixes[-1]:
            return f"{amount:.2f} {suffix}"
        amount /= 1024
    raise AssertionError("unit ladder is non-empty")


@app.command("download-model")
def download_model_cmd(
    model: str = typer.Argument(..., help="provider model id"),
    target: Path = typer.Argument(..., help="local directory that will contain the snapshot"),
    provider: str = typer.Option(
        "huggingface",
        help="model provider: huggingface | ollama | github-release",
    ),
    revision: Optional[str] = typer.Option(
        None,
        help="HF revision, Ollama tag, or GitHub release tag; provider default when omitted",
    ),
    token: Optional[str] = typer.Option(
        None,
        help="provider token; prefer HF_TOKEN/GITHUB_TOKEN in .env",
    ),
    chunk_mib: float = typer.Option(64, min=0.001, help="verified range-chunk size in MiB"),
    session_gib: float = typer.Option(
        64,
        min=0,
        help="maximum bytes this invocation may download in GiB; 0 is unlimited",
    ),
    max_mib_per_second: Optional[float] = typer.Option(
        None,
        min=0.001,
        help="hard average transfer ceiling in MiB/s",
    ),
    bandwidth_fraction: float = typer.Option(
        0.8,
        min=0,
        max=1,
        help="fraction of measured raw bandwidth to use; 0 disables adaptive pacing",
    ),
    timeout_seconds: float = typer.Option(60, min=1, help="per-request timeout"),
    retries: int = typer.Option(5, min=0, help="retries per chunk after transient failures"),
    max_rate_wait_seconds: float = typer.Option(
        900,
        min=0,
        help="maximum wait for one provider rate-limit response",
    ),
    min_free_gib: float = typer.Option(
        1,
        min=0,
        help="free-space reserve retained after each chunk",
    ),
    min_free_percent: float = typer.Option(
        5,
        min=0,
        max=99,
        help="filesystem capacity kept free after every chunk",
    ),
    verify_completed: bool = typer.Option(
        False,
        "--verify-completed",
        help="rehash completed files before resuming (detects later disk corruption)",
    ),
    verify_only: bool = typer.Option(
        False,
        "--verify-only",
        help="rehash the complete cached snapshot and download nothing",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="resolve metadata and show size without creating or downloading files",
    ),
) -> None:
    """Download a pinned model with checkpoints, throttling, and checksums."""
    config = DownloadConfig(
        repo_id=model,
        target_dir=target,
        provider=provider,
        revision=revision,
        token=token,
        chunk_bytes=int(chunk_mib * BYTES_PER_MIB),
        session_bytes=_bytes(session_gib, BYTES_PER_GIB),
        max_bytes_per_second=_bytes(max_mib_per_second, BYTES_PER_MIB),
        bandwidth_fraction=bandwidth_fraction or None,
        timeout_seconds=timeout_seconds,
        retries=retries,
        max_rate_limit_wait_seconds=max_rate_wait_seconds,
        min_free_bytes=int(min_free_gib * BYTES_PER_GIB),
        min_free_fraction=min_free_percent / 100,
        verify_completed=verify_completed,
        verify_only=verify_only,
        dry_run=dry_run,
    )
    try:
        report = download_model(
            config,
            progress=lambda detail: typer.echo(f"[download-model] {detail}"),
        )
    except (DownloadError, ValueError) as exc:
        typer.echo(f"[download-model] ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"[download-model] {report.status}: {report.complete_files}/{report.total_files} files, "
        f"{_human_bytes(report.completed_bytes)}/{_human_bytes(report.total_bytes)} complete, "
        f"{_human_bytes(report.session_downloaded_bytes)} downloaded this session"
    )
    typer.echo(
        f"[download-model] provider={report.provider} revision={report.resolved_revision} "
        f"target={report.target_dir}"
    )
