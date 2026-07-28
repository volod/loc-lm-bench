"""Streaming checksums and chunk-prefix repair for model files."""

import hashlib
import os
from pathlib import Path

from llb.backends.model_download.contracts import (
    ChunkState,
    DownloadIntegrityError,
    DownloadStateError,
    FileState,
)

_HASH_BLOCK_BYTES = 8 * 1024 * 1024
_RANGE_HASH_BLOCK_BYTES = 1024 * 1024


def hash_range(path: Path, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            block = handle.read(min(_RANGE_HASH_BLOCK_BYTES, remaining))
            if not block:
                raise DownloadIntegrityError(f"{path} ended before byte {offset + size}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def whole_checksum(path: Path, kind: str, size: int) -> str:
    if kind == "sha256":
        digest = hashlib.sha256()
    elif kind == "git-sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode())
    else:
        raise DownloadStateError(f"unsupported checksum kind: {kind}")
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _valid_chunk_prefix(path: Path, chunks: list[ChunkState]) -> list[ChunkState]:
    valid: list[ChunkState] = []
    expected_offset = 0
    for chunk in chunks:
        if (
            chunk.offset != expected_offset
            or hash_range(path, chunk.offset, chunk.size) != chunk.sha256
        ):
            break
        valid.append(chunk)
        expected_offset += chunk.size
    return valid


def repair_part(path: Path, file: FileState) -> bool:
    """Truncate a partial file to its longest locally checksum-valid chunk prefix."""
    valid = _valid_chunk_prefix(path, file.chunks)
    if len(valid) == len(file.chunks):
        return False
    valid_bytes = sum(chunk.size for chunk in valid)
    with path.open("r+b") as handle:
        handle.truncate(valid_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    file.chunks = valid
    return True
