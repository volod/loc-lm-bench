"""Describe a file a store needs but whose bytes belong to another library.

The digest is what makes the description a GATE rather than a note: a store records what each
opaque member weighed and hashed when it was published, so a member that was truncated, swapped,
or rebuilt by a different writer is caught before a query runs against it. A directory member --
a vector-store adapter persists one -- hashes its file tree, so the same check covers both shapes.
"""

import hashlib
from pathlib import Path

from llb.artifacts.errors import DatasetReadError
from llb.core.contracts.retrieval_graph.common import OpaqueIndexMember

_CHUNK_BYTES = 1 << 20


def content_digest(path: Path) -> str:
    """`sha256:<hex>` of a file, or of a directory's whole file tree."""
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            _absorb(child, digest)
    else:
        _absorb(path, digest)
    return f"sha256:{digest.hexdigest()}"


def content_size(path: Path) -> int:
    """Bytes of a file, or the summed bytes of a directory's file tree."""
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return path.stat().st_size


def describe_member(
    root: Path,
    member_id: str,
    relative_path: str,
    *,
    owner: str,
    artifact_format: str,
    format_version: str,
    description: str,
) -> OpaqueIndexMember:
    """Describe one present opaque member of the store rooted at `root`."""
    return OpaqueIndexMember(
        member_id=member_id,
        path=relative_path,
        owner=owner,
        format=artifact_format,
        format_version=format_version,
        description=description,
        digest=content_digest(root / relative_path),
        n_bytes=content_size(root / relative_path),
    )


def refuse_changed_members(root: Path, members: list[OpaqueIndexMember]) -> None:
    """Refuse a store whose declared opaque members are missing or no longer hash the same."""
    for member in members:
        path = root / member.path
        if not path.exists():
            raise DatasetReadError(
                f"{path}: {member.owner} {member.format} member '{member.member_id}' is declared "
                "by the store metadata but is missing"
            )
        observed = content_digest(path)
        if observed != member.digest:
            raise DatasetReadError(
                f"{path}: {member.owner} {member.format} member '{member.member_id}' changed "
                f"since publication; declared={member.digest}, observed={observed}"
            )


def _absorb(path: Path, digest: "hashlib._Hash") -> None:
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(block)
