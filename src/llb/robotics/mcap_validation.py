"""Standard-MCAP validation for temporal HFlow evidence references."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.robotics.digests import file_digest


@dataclass(frozen=True)
class McapWindow:
    channels: tuple[str, ...]
    message_start_ns: int
    message_end_ns: int
    message_count: int
    log_times: tuple[int, ...]


def _make_reader(stream: Any) -> Any:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise RuntimeError(
            "MCAP validation requires the robotics extra: uv pip install -e '.[robotics]'"
        ) from exc
    return make_reader(stream)


def inspect_mcap_channels(path: Path, channels: tuple[str, ...]) -> McapWindow:
    """Read channel coverage from a canonical file using only the stock MCAP package."""
    with path.open("rb") as stream:
        reader = _make_reader(stream)
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError(f"{path}: canonical MCAP requires a complete summary and statistics")
        available = {channel.topic for channel in summary.channels.values()}
        missing = set(channels) - available
        if missing:
            raise ValueError(f"{path}: evidence channels are absent from MCAP: {sorted(missing)}")
        times = [
            message.log_time
            for _schema, channel, message in reader.iter_messages(
                topics=list(channels), log_time_order=True
            )
            if channel.topic in channels
        ]
    if not times:
        raise ValueError(f"{path}: evidence channels contain no messages")
    return McapWindow(
        channels=tuple(sorted(channels)),
        message_start_ns=min(times),
        message_end_ns=max(times) + 1,
        message_count=len(times),
        log_times=tuple(times),
    )


def validate_mcap_window(
    path: Path,
    *,
    expected_sha256: str,
    episode_id: str,
    channels: tuple[str, ...],
    start_ns: int,
    end_ns: int,
) -> McapWindow:
    """Open one canonical file with the stock reader and validate a half-open reference."""
    observed_digest = file_digest(path)
    if observed_digest != expected_sha256:
        raise ValueError(
            f"{path}: MCAP digest mismatch, expected {expected_sha256}, observed {observed_digest}"
        )
    if observed_digest.removeprefix("sha256:")[:16] != episode_id:
        raise ValueError(f"{path}: episode id is not the HFlow content address")

    window = inspect_mcap_channels(path, channels)
    if start_ns < window.message_start_ns or end_ns > window.message_end_ns:
        raise ValueError(
            f"{path}: evidence interval [{start_ns}, {end_ns}) is outside "
            f"[{window.message_start_ns}, {window.message_end_ns})"
        )
    if not any(start_ns <= timestamp < end_ns for timestamp in window.log_times):
        raise ValueError(f"{path}: evidence interval contains no messages")
    return window
