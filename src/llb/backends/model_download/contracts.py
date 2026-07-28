"""Data contracts and defaults for bounded open-model downloads."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GIB = 1024 * BYTES_PER_MIB
DEFAULT_CHUNK_BYTES = 64 * BYTES_PER_MIB
DEFAULT_SESSION_BYTES = 64 * BYTES_PER_GIB
DEFAULT_BANDWIDTH_FRACTION = 0.8
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RETRIES = 5
DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS = 15 * 60.0
DEFAULT_MIN_FREE_BYTES = BYTES_PER_GIB
DEFAULT_MIN_FREE_FRACTION = 0.05
STATE_VERSION = 2
METADATA_DIR_NAME = ".llb-model-download"

ProgressCallback = Callable[[str], None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_optional(value: float | int | None) -> bool:
    return value is None or value > 0


class DownloadError(RuntimeError):
    """Base error for a model-cache operation."""


class DownloadAccessError(DownloadError):
    """The provider rejected authentication or gated-repository access."""


class DownloadIntegrityError(DownloadError):
    """Downloaded bytes do not match their expected identity."""


class DownloadStateError(DownloadError):
    """Persisted checkpoint state is incompatible or invalid."""


@dataclass(frozen=True)
class DownloadConfig:
    """Operator controls for one pinned model snapshot."""

    repo_id: str
    target_dir: Path
    provider: str = "huggingface"
    revision: str | None = None
    token: str | None = None
    chunk_bytes: int = DEFAULT_CHUNK_BYTES
    session_bytes: int | None = DEFAULT_SESSION_BYTES
    max_bytes_per_second: float | None = None
    bandwidth_fraction: float | None = DEFAULT_BANDWIDTH_FRACTION
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    retries: int = DEFAULT_RETRIES
    max_rate_limit_wait_seconds: float = DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    min_free_fraction: float = DEFAULT_MIN_FREE_FRACTION
    verify_completed: bool = False
    verify_only: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        _require(bool(self.repo_id.strip()), "repo_id must not be empty")
        _require(
            self.revision is None or bool(self.revision.strip()),
            "revision must not be empty",
        )
        _require(self.chunk_bytes > 0, "chunk_bytes must be positive")
        _require(
            _positive_optional(self.session_bytes),
            "session_bytes must be positive or None",
        )
        _require(
            _positive_optional(self.max_bytes_per_second),
            "max_bytes_per_second must be positive or None",
        )
        _require(
            self.bandwidth_fraction is None or 0 < self.bandwidth_fraction <= 1,
            "bandwidth_fraction must be in (0, 1] or None",
        )
        _require(self.timeout_seconds > 0, "timeout_seconds must be positive")
        _require(self.retries >= 0, "retries must be non-negative")
        _require(
            self.max_rate_limit_wait_seconds >= 0,
            "max_rate_limit_wait_seconds must be non-negative",
        )
        _require(self.min_free_bytes >= 0, "min_free_bytes must be non-negative")
        _require(0 <= self.min_free_fraction < 1, "min_free_fraction must be in [0, 1)")
        _require(
            not (self.verify_only and self.dry_run), "verify_only and dry_run cannot be combined"
        )


@dataclass
class ChunkState:
    offset: int
    size: int
    sha256: str


@dataclass
class FileState:
    path: str
    size: int
    checksum_kind: str
    checksum: str
    source_url: str
    source_accept: str | None = None
    chunks: list[ChunkState] = field(default_factory=list)
    complete: bool = False

    @property
    def downloaded_bytes(self) -> int:
        return sum(chunk.size for chunk in self.chunks)


@dataclass
class SnapshotState:
    provider: str
    repo_id: str
    requested_revision: str
    resolved_revision: str
    files: list[FileState]
    observed_raw_bytes_per_second: float | None = None
    version: int = STATE_VERSION

    @property
    def total_bytes(self) -> int:
        return sum(file.size for file in self.files)

    @property
    def completed_bytes(self) -> int:
        return sum(file.size for file in self.files if file.complete)


@dataclass(frozen=True)
class DownloadReport:
    provider: str
    repo_id: str
    resolved_revision: str
    target_dir: Path
    total_bytes: int
    completed_bytes: int
    session_downloaded_bytes: int
    complete_files: int
    total_files: int
    status: str
