"""Snapshot transfer loop, free-space checks, pacing, and final-file verification."""

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from huggingface_hub.utils import get_session

from llb.backends.model_download.contracts import (
    ChunkState,
    DownloadConfig,
    DownloadError,
    DownloadIntegrityError,
    FileState,
    ProgressCallback,
    SnapshotState,
)
from llb.backends.model_download.http_range import download_chunk
from llb.backends.model_download.integrity import repair_part, whole_checksum
from llb.backends.model_download.state import (
    part_path,
    safe_final_path,
    save_state,
)

_EWMA_WEIGHT = 0.25


@dataclass
class _Throttle:
    explicit_limit: float | None
    fraction: float | None
    observed_raw: float | None

    def observe(self, byte_count: int, transfer_seconds: float) -> tuple[float, float]:
        raw_rate = byte_count / max(transfer_seconds, 0.001)
        if self.observed_raw is None:
            self.observed_raw = raw_rate
        else:
            self.observed_raw = _EWMA_WEIGHT * raw_rate + (1 - _EWMA_WEIGHT) * self.observed_raw
        limits = [value for value in (self.explicit_limit,) if value is not None]
        if self.fraction is not None:
            limits.append(self.observed_raw * self.fraction)
        delay = 0.0
        if limits:
            delay = max(0.0, byte_count / min(limits) - transfer_seconds)
        return self.observed_raw, delay


@dataclass
class _TransferSession:
    budget: int | None
    throttle: _Throttle
    downloaded: int = 0

    def next_chunk_size(self, file: FileState, preferred: int) -> int | None:
        remaining_budget = None if self.budget is None else self.budget - self.downloaded
        if remaining_budget is not None and remaining_budget <= 0:
            return None
        size = min(preferred, file.size - file.downloaded_bytes)
        return min(size, remaining_budget) if remaining_budget is not None else size


def _disk_usage(target_dir: Path) -> tuple[int, int]:
    probe = target_dir
    while not probe.exists():
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return usage.total, usage.free


def _require_free_space(config: DownloadConfig, chunk_size: int) -> None:
    total, free = _disk_usage(config.target_dir)
    capacity_reserve = int(total * config.min_free_fraction)
    reserve = max(config.min_free_bytes, capacity_reserve)
    if free - chunk_size < reserve:
        raise DownloadError(
            f"insufficient free space: need {chunk_size} bytes plus {reserve} bytes reserve "
            f"({config.min_free_fraction:.1%} of filesystem or configured floor), "
            f"have {free} bytes"
        )


def _finalize_file(config: DownloadConfig, file: FileState) -> None:
    part = part_path(config.target_dir, file.path)
    part.parent.mkdir(parents=True, exist_ok=True)
    part.touch(exist_ok=True)
    actual = whole_checksum(part, file.checksum_kind, file.size)
    if actual != file.checksum:
        if repair_part(part, file):
            raise DownloadIntegrityError(
                f"local chunk corruption found in {file.path}; restart will resume "
                f"from byte {file.downloaded_bytes}"
            )
        file.chunks.clear()
        with part.open("r+b") as handle:
            handle.truncate(0)
        raise DownloadIntegrityError(
            f"upstream checksum mismatch for {file.path}: expected {file.checksum}, got {actual}"
        )
    final = safe_final_path(config.target_dir, file.path)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, final)
    file.complete = True


def _checkpoint_chunk(
    config: DownloadConfig,
    state: SnapshotState,
    file: FileState,
    session: _TransferSession,
    chunk: ChunkState,
    elapsed: float,
    sleeper: Callable[[float], None],
    progress: ProgressCallback | None,
) -> None:
    file.chunks.append(chunk)
    session.downloaded += chunk.size
    observed, throttle_delay = session.throttle.observe(chunk.size, elapsed)
    state.observed_raw_bytes_per_second = observed
    save_state(config.target_dir, state)
    if progress:
        progress(
            f"{file.path}: {file.downloaded_bytes}/{file.size} bytes "
            f"(session {session.downloaded} bytes)"
        )
    if throttle_delay:
        sleeper(throttle_delay)


def _transfer_file(
    config: DownloadConfig,
    state: SnapshotState,
    file: FileState,
    session: _TransferSession,
    client: Any,
    sleeper: Callable[[float], None],
    progress: ProgressCallback | None,
) -> bool:
    while file.downloaded_bytes < file.size:
        chunk_size = session.next_chunk_size(file, config.chunk_bytes)
        if chunk_size is None:
            return False
        _require_free_space(config, chunk_size)
        chunk, elapsed = download_chunk(
            config,
            file,
            chunk_size,
            client=client,
            sleeper=sleeper,
        )
        _checkpoint_chunk(
            config,
            state,
            file,
            session,
            chunk,
            elapsed,
            sleeper,
            progress,
        )
    return True


def _finalize_and_checkpoint(
    config: DownloadConfig,
    state: SnapshotState,
    file: FileState,
) -> None:
    try:
        _finalize_file(config, file)
    except DownloadIntegrityError:
        save_state(config.target_dir, state)
        raise
    save_state(config.target_dir, state)


def transfer_snapshot(
    config: DownloadConfig,
    state: SnapshotState,
    *,
    client: Any | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    progress: ProgressCallback | None = None,
) -> int:
    """Download no more than the configured session budget and checkpoint each verified chunk."""
    session = _TransferSession(
        budget=config.session_bytes,
        throttle=_Throttle(
            explicit_limit=config.max_bytes_per_second,
            fraction=config.bandwidth_fraction,
            observed_raw=state.observed_raw_bytes_per_second,
        ),
    )
    http_client = client or get_session()

    for file in state.files:
        if file.complete:
            continue
        completed = _transfer_file(
            config,
            state,
            file,
            session,
            http_client,
            sleeper,
            progress,
        )
        if not completed:
            return session.downloaded
        _finalize_and_checkpoint(config, state, file)
    return session.downloaded
