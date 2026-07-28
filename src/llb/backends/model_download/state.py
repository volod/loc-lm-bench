"""Atomic checkpoints, locking, safe paths, and local integrity recovery."""

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from llb.backends.model_download.contracts import (
    METADATA_DIR_NAME,
    STATE_VERSION,
    ChunkState,
    DownloadStateError,
    FileState,
    SnapshotState,
)
from llb.backends.model_download.integrity import hash_range, repair_part, whole_checksum


def metadata_dir(target_dir: Path) -> Path:
    return target_dir / METADATA_DIR_NAME


def state_path(target_dir: Path) -> Path:
    return metadata_dir(target_dir) / "state.json"


def part_path(target_dir: Path, relative_path: str) -> Path:
    name = hashlib.sha256(relative_path.encode("utf-8")).hexdigest() + ".part"
    return metadata_dir(target_dir) / "parts" / name


def safe_final_path(target_dir: Path, relative_path: str) -> Path:
    root = target_dir.resolve()
    candidate = (target_dir / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise DownloadStateError(f"repository path escapes target directory: {relative_path!r}")
    return candidate


def save_state(target_dir: Path, state: SnapshotState) -> None:
    directory = metadata_dir(target_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = state_path(target_dir)
    temporary = destination.with_suffix(".json.tmp")
    payload = json.dumps(asdict(state), indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _decode_file(raw: dict[str, Any]) -> FileState:
    raw_chunks = raw.get("chunks", [])
    if not isinstance(raw_chunks, list):
        raise DownloadStateError("checkpoint chunks must be a list")
    chunks = [ChunkState(**chunk) for chunk in raw_chunks if isinstance(chunk, dict)]
    return FileState(
        path=str(raw["path"]),
        size=int(raw["size"]),
        checksum_kind=str(raw["checksum_kind"]),
        checksum=str(raw["checksum"]),
        source_url=str(raw["source_url"]),
        source_accept=(str(raw["source_accept"]) if raw.get("source_accept") is not None else None),
        chunks=chunks,
        complete=bool(raw.get("complete", False)),
    )


def load_state(target_dir: Path) -> SnapshotState | None:
    path = state_path(target_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if int(raw["version"]) != STATE_VERSION:
            raise DownloadStateError(
                f"unsupported checkpoint version {raw['version']}; expected {STATE_VERSION}"
            )
        files = [_decode_file(item) for item in raw["files"]]
        return SnapshotState(
            provider=str(raw["provider"]),
            repo_id=str(raw["repo_id"]),
            requested_revision=str(raw["requested_revision"]),
            resolved_revision=str(raw["resolved_revision"]),
            files=files,
            observed_raw_bytes_per_second=raw.get("observed_raw_bytes_per_second"),
            version=int(raw["version"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DownloadStateError(f"invalid checkpoint {path}: {exc}") from exc


def validate_identity(
    state: SnapshotState,
    provider: str,
    repo_id: str,
    revision: str,
) -> None:
    if (
        state.provider != provider
        or state.repo_id != repo_id
        or state.requested_revision != revision
    ):
        raise DownloadStateError(
            "target already checkpoints "
            f"{state.provider}:{state.repo_id}@{state.requested_revision}; "
            "choose another target directory"
        )


@contextmanager
def target_lock(target_dir: Path) -> Iterator[None]:
    directory = metadata_dir(target_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / "download.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DownloadStateError(f"another download owns {target_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _well_formed_chunks(chunks: list[ChunkState]) -> bool:
    expected_offset = 0
    for chunk in chunks:
        if chunk.offset != expected_offset or chunk.size <= 0:
            return False
        expected_offset += chunk.size
    return True


def _valid_published_file(final: Path, file: FileState, verify_checksum: bool) -> bool:
    if not final.is_file() or final.stat().st_size != file.size:
        return False
    return (
        whole_checksum(final, file.checksum_kind, file.size) == file.checksum
        if verify_checksum
        else True
    )


def _recover_publication(
    final: Path,
    part: Path,
    file: FileState,
    verify_completed: bool,
) -> tuple[bool, bool]:
    """Return (changed, publication_is_complete)."""
    rename_crashed = not file.complete and file.downloaded_bytes == file.size
    if rename_crashed and _valid_published_file(final, file, True):
        file.complete = True
        return True, True
    if not file.complete:
        return False, False
    if _valid_published_file(final, file, verify_completed):
        return False, True
    if final.exists():
        part.parent.mkdir(parents=True, exist_ok=True)
        os.replace(final, part)
    file.complete = False
    return True, False


def _recover_partial(part: Path, file: FileState, deep_verify: bool) -> bool:
    if not part.exists():
        if file.chunks:
            file.chunks.clear()
            return True
        return False

    downloaded = file.downloaded_bytes
    structurally_valid = _well_formed_chunks(file.chunks) and part.stat().st_size == downloaded
    tail_valid = (
        not file.chunks
        or hash_range(part, file.chunks[-1].offset, file.chunks[-1].size) == file.chunks[-1].sha256
    )
    if structurally_valid and tail_valid and not deep_verify:
        return False
    changed = repair_part(part, file)
    valid_bytes = file.downloaded_bytes
    if part.stat().st_size != valid_bytes:
        with part.open("r+b") as handle:
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        changed = True
    return changed


def recover_file(target_dir: Path, file: FileState, verify_completed: bool) -> bool:
    """Recover to the last checksum-valid chunk. Return True when state changed."""
    final = safe_final_path(target_dir, file.path)
    part = part_path(target_dir, file.path)
    changed, complete = _recover_publication(final, part, file, verify_completed)
    if complete:
        return changed
    return _recover_partial(part, file, verify_completed) or changed
