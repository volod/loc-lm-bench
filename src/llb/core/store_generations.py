"""Immutable store-generation directories shared by the RAG and graph stores.

A store base directory (for example ``$DATA_DIR/llb/rag``) historically holds the built store
files directly. `refresh-index` never edits a built store in place: each refresh writes a complete
new store into ``<base>/generations/<utc-timestamp>/`` (staged hidden, then published with one
atomic rename), so every generation directory is immutable and the rollback unit -- deleting the
newest generation resumes the previous one.

`resolve_store_dir` picks the live store among the base directory itself and its generations by
the newest store meta file (mtime, ties broken toward the greater generation name), so a plain
rebuild into the base directory takes over again after a refresh.
"""

from datetime import datetime, timezone
from pathlib import Path

GENERATIONS_DIRNAME = "generations"

_STAGING_PREFIX = "."
_STAGING_SUFFIX = ".tmp"


def generation_timestamp(now: datetime | None = None) -> str:
    """UTC second-resolution timestamp used as a generation directory name."""
    now = now if now is not None else datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_store_dir(base_dir: Path | str, meta_filename: str) -> Path:
    """The live store directory for `base_dir`: the base itself or its newest generation.

    Candidates are the base directory and every ``generations/<name>`` child that contains
    `meta_filename`; the one whose meta file is newest wins (a tie prefers the greater
    generation name, so a just-published refresh beats the base it was derived from). With no
    candidate at all the base is returned unchanged and the caller fails with its normal
    missing-store error.
    """
    base_dir = Path(base_dir)
    candidates: list[tuple[float, str, Path]] = []
    base_meta = base_dir / meta_filename
    if base_meta.is_file():
        candidates.append((base_meta.stat().st_mtime, "", base_dir))
    generations_dir = base_dir / GENERATIONS_DIRNAME
    if generations_dir.is_dir():
        for child in generations_dir.iterdir():
            meta = child / meta_filename
            if child.is_dir() and not child.name.startswith(_STAGING_PREFIX) and meta.is_file():
                candidates.append((meta.stat().st_mtime, child.name, child))
    if not candidates:
        return base_dir
    return max(candidates, key=lambda entry: (entry[0], entry[1]))[2]


def new_generation_paths(base_dir: Path | str, timestamp: str) -> tuple[Path, Path]:
    """(staging_dir, final_dir) for a new generation named `timestamp` under `base_dir`.

    The final name is made unique (``-2``, ``-3``, ...) when the timestamp collides with an
    existing generation, so two refreshes within one second never overwrite each other.
    """
    generations_dir = Path(base_dir) / GENERATIONS_DIRNAME
    name = timestamp
    suffix = 2
    while (generations_dir / name).exists():
        name = f"{timestamp}-{suffix}"
        suffix += 1
    final_dir = generations_dir / name
    staging_dir = generations_dir / f"{_STAGING_PREFIX}{name}{_STAGING_SUFFIX}"
    return staging_dir, final_dir


def publish_generation(staging_dir: Path, final_dir: Path) -> Path:
    """Atomically publish a fully written staging directory as the immutable generation."""
    staging_dir.rename(final_dir)
    return final_dir
