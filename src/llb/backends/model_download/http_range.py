"""One bounded HTTP range request with retry and Hub rate-limit policy."""

import hashlib
import os
import re
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import httpx

from llb.backends.model_download.contracts import (
    ChunkState,
    DownloadAccessError,
    DownloadConfig,
    DownloadError,
    DownloadIntegrityError,
    FileState,
)
from llb.backends.model_download.integrity import hash_range
from llb.backends.model_download.state import part_path

_RATE_LIMIT_RESET = re.compile(r"(?:^|;)\s*t=(\d+)")
_NETWORK_BLOCK_BYTES = 1024 * 1024


class _RateLimitError(DownloadError):
    def __init__(self, headers: Any) -> None:
        super().__init__("provider rate limit reached")
        self.headers = headers


class _RetryableHttpError(DownloadError):
    pass


@dataclass(frozen=True)
class _AttemptOutcome:
    chunk: ChunkState | None = None
    elapsed: float = 0.0
    error: DownloadError | None = None
    retryable: bool = False
    delay: float = 0.0


def _retry_delay(headers: Any, attempt: int, maximum: float) -> float:
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return min(maximum, max(0.0, float(retry_after)))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(retry_after)
                return min(maximum, max(0.0, parsed.timestamp() - time.time()))
            except (TypeError, ValueError):
                pass
    rate_limit = headers.get("ratelimit", "")
    match = _RATE_LIMIT_RESET.search(rate_limit)
    if match:
        return min(maximum, float(match.group(1)))
    return min(maximum, float(2**attempt))


def _validate_response(response: Any, offset: int, size: int, file_size: int) -> None:
    if response.status_code in (401, 403):
        raise DownloadAccessError("provider denied the file request; check access and token")
    if response.status_code == 429:
        raise _RateLimitError(response.headers)
    if response.status_code >= 500:
        raise _RetryableHttpError(f"provider returned HTTP {response.status_code}")
    if response.status_code >= 400:
        raise DownloadError(f"provider returned HTTP {response.status_code}")
    if response.status_code == 206:
        expected = f"bytes {offset}-{offset + size - 1}/{file_size}"
        actual = response.headers.get("content-range")
        if actual != expected:
            raise DownloadIntegrityError(
                f"unexpected Content-Range {actual!r}; expected {expected!r}"
            )
        return
    if response.status_code == 200 and offset == 0 and size == file_size:
        return
    raise DownloadIntegrityError(
        f"provider ignored bounded range {offset}-{offset + size - 1} for a {file_size}-byte file"
    )


def _stream_once(
    client: Any,
    url: str,
    headers: dict[str, str],
    part: Any,
    offset: int,
    size: int,
    file_size: int,
    timeout_seconds: float,
) -> tuple[str, float]:
    digest = hashlib.sha256()
    received = 0
    started = time.monotonic()
    with client.stream(
        "GET",
        url,
        headers=headers,
        timeout=timeout_seconds,
        follow_redirects=True,
    ) as response:
        _validate_response(response, offset, size, file_size)
        with part.open("r+b" if part.exists() else "w+b") as handle:
            handle.seek(offset)
            for block in response.iter_bytes(chunk_size=_NETWORK_BLOCK_BYTES):
                if received + len(block) > size:
                    raise DownloadIntegrityError("Hub sent more bytes than the requested range")
                handle.write(block)
                digest.update(block)
                received += len(block)
            handle.flush()
            os.fsync(handle.fileno())
    elapsed = max(time.monotonic() - started, 0.001)
    if received != size:
        raise DownloadIntegrityError(f"short range response: received {received} of {size} bytes")
    return digest.hexdigest(), elapsed


def _verified_chunk(
    config: DownloadConfig,
    file: FileState,
    size: int,
    client: Any,
    headers: dict[str, str],
) -> tuple[ChunkState, float]:
    offset = file.downloaded_bytes
    part = part_path(config.target_dir, file.path)
    sha256, elapsed = _stream_once(
        client,
        file.source_url,
        headers,
        part,
        offset,
        size,
        file.size,
        config.timeout_seconds,
    )
    if hash_range(part, offset, size) != sha256:
        raise DownloadIntegrityError("chunk changed between network write and disk readback")
    return ChunkState(offset=offset, size=size, sha256=sha256), elapsed


def _attempt_chunk(
    config: DownloadConfig,
    file: FileState,
    size: int,
    client: Any,
    headers: dict[str, str],
    attempt: int,
) -> _AttemptOutcome:
    try:
        chunk, elapsed = _verified_chunk(config, file, size, client, headers)
        return _AttemptOutcome(chunk=chunk, elapsed=elapsed)
    except DownloadAccessError as exc:
        return _AttemptOutcome(error=exc)
    except _RateLimitError as exc:
        delay = _retry_delay(exc.headers, attempt, config.max_rate_limit_wait_seconds)
        return _AttemptOutcome(error=exc, retryable=True, delay=delay)
    except (_RetryableHttpError, DownloadIntegrityError) as exc:
        delay = min(config.max_rate_limit_wait_seconds, float(2**attempt))
        return _AttemptOutcome(error=exc, retryable=True, delay=delay)
    except DownloadError as exc:
        return _AttemptOutcome(error=exc)
    except (httpx.NetworkError, httpx.TimeoutException) as exc:
        error = DownloadError(f"network request failed: {exc}")
        delay = min(config.max_rate_limit_wait_seconds, float(2**attempt))
        return _AttemptOutcome(error=error, retryable=True, delay=delay)


def download_chunk(
    config: DownloadConfig,
    file: FileState,
    size: int,
    *,
    client: Any,
    sleeper: Callable[[float], None],
) -> tuple[ChunkState, float]:
    """Download and disk-readback one range, retrying only transient failures."""
    offset = file.downloaded_bytes
    part = part_path(config.target_dir, file.path)
    part.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "Accept-Encoding": "identity",
        "Range": f"bytes={offset}-{offset + size - 1}",
    }
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"
    if file.source_accept:
        headers["Accept"] = file.source_accept

    for attempt in range(config.retries + 1):
        outcome = _attempt_chunk(config, file, size, client, headers, attempt)
        if outcome.chunk is not None:
            return outcome.chunk, outcome.elapsed
        if outcome.error is None:
            raise DownloadError("download attempt returned neither data nor an error")
        if not outcome.retryable or attempt == config.retries:
            raise outcome.error
        sleeper(outcome.delay)
    raise DownloadError("unreachable retry state")
