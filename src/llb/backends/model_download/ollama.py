"""Resolve an Ollama registry model into its content-addressed local store."""

import hashlib
import json
from typing import Any

from llb.backends.model_download.contracts import (
    DownloadAccessError,
    DownloadStateError,
    FileState,
    SnapshotState,
)

OLLAMA_REGISTRY = "https://registry.ollama.ai"
OLLAMA_REGISTRY_HOST = "registry.ollama.ai"
OLLAMA_MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"


def normalized_ollama_revision(repo_id: str, revision: str | None) -> str:
    _repository, tag = _repository_and_tag(repo_id, revision)
    return tag


def _repository_and_tag(repo_id: str, revision: str | None) -> tuple[str, str]:
    value = repo_id.removeprefix("ollama://").strip("/")
    if not value:
        raise DownloadStateError("Ollama model id must not be empty")
    last = value.rsplit("/", 1)[-1]
    embedded_tag = last.rsplit(":", 1)[1] if ":" in last else None
    repository = value.rsplit(":", 1)[0] if embedded_tag else value
    if embedded_tag and revision and embedded_tag != revision:
        raise DownloadStateError(
            f"Ollama id selects tag {embedded_tag!r} but --revision selects {revision!r}"
        )
    tag = revision or embedded_tag or "latest"
    if "/" not in repository:
        repository = f"library/{repository}"
    if any(part in ("", ".", "..") for part in repository.split("/")):
        raise DownloadStateError(f"invalid Ollama repository path: {repository!r}")
    return repository, tag


def _manifest_response(
    client: Any,
    url: str,
    token: str | None,
    timeout_seconds: float,
) -> Any:
    headers = {"Accept": OLLAMA_MANIFEST_ACCEPT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = client.get(url, headers=headers, timeout=timeout_seconds, follow_redirects=True)
    if response.status_code in (401, 403):
        raise DownloadAccessError("Ollama registry denied access to the model manifest")
    if response.status_code >= 400:
        raise DownloadStateError(f"Ollama registry returned HTTP {response.status_code}")
    return response


def _digest(value: object, label: str) -> str:
    text = str(value)
    algorithm, separator, digest = text.partition(":")
    if separator != ":" or algorithm != "sha256" or len(digest) != 64:
        raise DownloadStateError(f"{label} has unsupported digest {text!r}")
    return digest


def fetch_manifest(
    repo_id: str,
    revision: str | None,
    token: str | None,
    *,
    client: Any,
    timeout_seconds: float,
) -> SnapshotState:
    """Fetch and pin the manifest, then enumerate every immutable Ollama blob."""
    repository, tag = _repository_and_tag(repo_id, revision)
    tag_url = f"{OLLAMA_REGISTRY}/v2/{repository}/manifests/{tag}"
    response = _manifest_response(client, tag_url, token, timeout_seconds)
    raw_manifest = bytes(response.content)
    raw_digest = hashlib.sha256(raw_manifest).hexdigest()
    header_digest = response.headers.get("docker-content-digest")
    if header_digest and _digest(header_digest, "manifest") != raw_digest:
        raise DownloadStateError("Ollama manifest content digest does not match its response body")
    try:
        manifest = json.loads(raw_manifest)
        descriptors = [manifest["config"], *manifest["layers"]]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DownloadStateError(f"invalid Ollama image manifest: {exc}") from exc

    digest_ref = f"sha256:{raw_digest}"
    files: list[FileState] = []
    seen = set()
    for descriptor in descriptors:
        digest_text = str(descriptor.get("digest"))
        digest = _digest(digest_text, "blob")
        if digest in seen:
            continue
        seen.add(digest)
        size = int(descriptor.get("size", -1))
        if size < 0:
            raise DownloadStateError(f"Ollama blob {digest_text} has no valid size")
        files.append(
            FileState(
                path=f"blobs/sha256-{digest}",
                size=size,
                checksum_kind="sha256",
                checksum=digest,
                source_url=f"{OLLAMA_REGISTRY}/v2/{repository}/blobs/{digest_text}",
            )
        )
    # Publish the manifest last so an interrupted download is not visible to Ollama as complete.
    files.append(
        FileState(
            path=f"manifests/{OLLAMA_REGISTRY_HOST}/{repository}/{tag}",
            size=len(raw_manifest),
            checksum_kind="sha256",
            checksum=raw_digest,
            source_url=f"{OLLAMA_REGISTRY}/v2/{repository}/manifests/{digest_ref}",
            source_accept=OLLAMA_MANIFEST_ACCEPT,
        )
    )
    return SnapshotState(
        provider="ollama",
        repo_id=repo_id,
        requested_revision=tag,
        resolved_revision=digest_ref,
        files=files,
    )
