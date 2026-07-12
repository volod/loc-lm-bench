"""Shared helpers for the benchmark category CLI commands."""

from typing import Any

import typer


def _echo_throughput(label: str, meter: Any) -> None:
    """Echo the run's real generation throughput (mirrors bench-security); silent for a run whose
    endpoint recorded no successful calls (e.g. the native tooling transport that bypasses the
    metered `complete`)."""
    if meter.calls:
        typer.echo(
            f"[{label}] throughput={meter.tokens_per_s:.1f} tok/s over {meter.calls} calls "
            f"({meter.completion_tokens} completion tokens)"
        )
