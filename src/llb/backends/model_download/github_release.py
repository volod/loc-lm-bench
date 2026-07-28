"""Resolve checksum-bearing GitHub release assets as a model snapshot."""

from typing import Any
from urllib.parse import quote

from llb.backends.model_download.contracts import (
    DownloadAccessError,
    DownloadStateError,
    FileState,
    SnapshotState,
)

GITHUB_API = "https://api.github.com"
GITHUB_API_ACCEPT = "application/vnd.github+json"
GITHUB_ASSET_ACCEPT = "application/octet-stream"
GITHUB_API_VERSION = "2022-11-28"


def normalized_github_revision(revision: str | None) -> str:
    return revision or "latest"


def _release_response(
    client: Any,
    repo_id: str,
    revision: str | None,
    token: str | None,
    timeout_seconds: float,
) -> Any:
    parts = repo_id.strip("/").split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        raise DownloadStateError("GitHub release model id must be owner/repository")
    suffix = "latest" if revision is None else f"tags/{quote(revision, safe='')}"
    headers = {
        "Accept": GITHUB_API_ACCEPT,
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{GITHUB_API}/repos/{parts[0]}/{parts[1]}/releases/{suffix}"
    response = client.get(url, headers=headers, timeout=timeout_seconds, follow_redirects=True)
    if response.status_code in (401, 403):
        raise DownloadAccessError("GitHub denied access or the API rate limit was exhausted")
    if response.status_code >= 400:
        raise DownloadStateError(f"GitHub Releases API returned HTTP {response.status_code}")
    return response


def _asset_file(asset: dict[str, Any]) -> FileState:
    digest_text = asset.get("digest")
    if not isinstance(digest_text, str) or not digest_text.startswith("sha256:"):
        raise DownloadStateError(
            f"GitHub asset {asset.get('name')!r} has no server-side SHA-256 digest"
        )
    digest = digest_text.removeprefix("sha256:")
    if len(digest) != 64:
        raise DownloadStateError(f"GitHub asset {asset.get('name')!r} has an invalid digest")
    name = str(asset.get("name", ""))
    if not name or "/" in name or name in (".", ".."):
        raise DownloadStateError(f"GitHub asset has an unsafe name: {name!r}")
    size = int(asset.get("size", -1))
    if size < 0:
        raise DownloadStateError(f"GitHub asset {name!r} has no valid size")
    return FileState(
        path=name,
        size=size,
        checksum_kind="sha256",
        checksum=digest,
        source_url=str(asset["url"]),
        source_accept=GITHUB_ASSET_ACCEPT,
    )


def fetch_manifest(
    repo_id: str,
    revision: str | None,
    token: str | None,
    *,
    client: Any,
    timeout_seconds: float,
) -> SnapshotState:
    """Pin a release and retain only assets with provider-supplied SHA-256 identities."""
    response = _release_response(client, repo_id, revision, token, timeout_seconds)
    try:
        release = response.json()
        assets = [_asset_file(asset) for asset in release["assets"]]
        tag = str(release["tag_name"])
        release_id = int(release["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DownloadStateError(f"invalid GitHub release metadata: {exc}") from exc
    if not assets:
        raise DownloadStateError("GitHub release has no downloadable assets")
    assets.sort(key=lambda file: file.path)
    return SnapshotState(
        provider="github-release",
        repo_id=repo_id,
        requested_revision=normalized_github_revision(revision),
        resolved_revision=f"{tag} (release {release_id})",
        files=assets,
    )
