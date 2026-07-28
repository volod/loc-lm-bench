"""Build an immutable file manifest from Hugging Face repository metadata."""

from pathlib import PurePosixPath
from typing import Any

from huggingface_hub import HfApi, hf_hub_url

from llb.backends.model_download.contracts import (
    DownloadAccessError,
    DownloadStateError,
    FileState,
    SnapshotState,
)


def _validate_remote_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts:
        raise DownloadStateError(f"unsafe repository path: {path!r}")
    return candidate.as_posix()


def _file_checksum(sibling: Any) -> tuple[str, str]:
    lfs = getattr(sibling, "lfs", None)
    if lfs is not None:
        sha256 = getattr(lfs, "sha256", None)
        if sha256:
            return "sha256", str(sha256)
    blob_id = getattr(sibling, "blob_id", None)
    if blob_id:
        return "git-sha1", str(blob_id)
    raise DownloadStateError(
        f"Hub returned no content identity for {getattr(sibling, 'rfilename', '<unknown>')}"
    )


def fetch_manifest(
    repo_id: str,
    revision: str | None,
    token: str | None,
    *,
    api: Any | None = None,
) -> SnapshotState:
    """Resolve a moving revision once and retain file sizes plus upstream checksums."""
    requested_revision = revision or "main"
    hub = api or HfApi(token=token)
    try:
        info = hub.model_info(
            repo_id=repo_id,
            revision=requested_revision,
            files_metadata=True,
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if any(code in detail.lower() for code in ("401", "403", "gated", "access")):
            raise DownloadAccessError(
                f"access denied for {repo_id}; accept any repository terms and set HF_TOKEN"
            ) from exc
        raise DownloadStateError(f"could not read Hub metadata for {repo_id}: {detail}") from exc

    resolved_revision = getattr(info, "sha", None)
    siblings = getattr(info, "siblings", None)
    if not resolved_revision or siblings is None:
        raise DownloadStateError("Hub metadata omitted the commit SHA or file list")

    files: list[FileState] = []
    for sibling in siblings:
        size = getattr(sibling, "size", None)
        if size is None or size < 0:
            raise DownloadStateError(
                f"Hub returned no valid size for {getattr(sibling, 'rfilename', '<unknown>')}"
            )
        checksum_kind, checksum = _file_checksum(sibling)
        files.append(
            FileState(
                path=_validate_remote_path(str(sibling.rfilename)),
                size=int(size),
                checksum_kind=checksum_kind,
                checksum=checksum,
                source_url=hf_hub_url(
                    repo_id=repo_id,
                    filename=str(sibling.rfilename),
                    revision=str(resolved_revision),
                ),
            )
        )
    files.sort(key=lambda item: item.path)
    return SnapshotState(
        provider="huggingface",
        repo_id=repo_id,
        requested_revision=requested_revision,
        resolved_revision=str(resolved_revision),
        files=files,
    )
