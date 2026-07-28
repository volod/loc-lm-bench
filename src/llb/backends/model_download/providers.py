"""Provider registry and provider-specific token/revision normalization."""

import os
from typing import Any

from huggingface_hub import get_token
from huggingface_hub.utils import get_session

from llb.backends.model_download.contracts import DownloadConfig, DownloadStateError, SnapshotState
from llb.backends.model_download.github_release import (
    fetch_manifest as fetch_github_manifest,
)
from llb.backends.model_download.github_release import normalized_github_revision
from llb.backends.model_download.huggingface import fetch_manifest as fetch_huggingface_manifest
from llb.backends.model_download.ollama import fetch_manifest as fetch_ollama_manifest
from llb.backends.model_download.ollama import normalized_ollama_revision
from llb.core import env

_ALIASES = {
    "hf": "huggingface",
    "hugging-face": "huggingface",
    "github": "github-release",
    "github-releases": "github-release",
}
SUPPORTED_PROVIDERS = ("huggingface", "ollama", "github-release")


def normalize_provider(value: str) -> str:
    provider = _ALIASES.get(value.strip().lower(), value.strip().lower())
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise DownloadStateError(f"unsupported provider {value!r}; choose one of: {supported}")
    return provider


def normalized_revision(provider: str, repo_id: str, revision: str | None) -> str:
    if provider == "huggingface":
        return revision or "main"
    if provider == "ollama":
        return normalized_ollama_revision(repo_id, revision)
    return normalized_github_revision(revision)


def provider_token(config: DownloadConfig, provider: str) -> str | None:
    if config.token:
        return config.token
    if provider == "huggingface":
        return os.environ.get(env.HF_TOKEN) or get_token()
    if provider == "github-release":
        return os.environ.get(env.GITHUB_TOKEN)
    return None


def fetch_manifest(
    config: DownloadConfig,
    *,
    api: Any | None = None,
    client: Any | None = None,
) -> SnapshotState:
    provider = normalize_provider(config.provider)
    if provider == "huggingface":
        return fetch_huggingface_manifest(
            config.repo_id,
            config.revision,
            config.token,
            api=api,
        )
    http_client = client or get_session()
    if provider == "ollama":
        return fetch_ollama_manifest(
            config.repo_id,
            config.revision,
            config.token,
            client=http_client,
            timeout_seconds=config.timeout_seconds,
        )
    return fetch_github_manifest(
        config.repo_id,
        config.revision,
        config.token,
        client=http_client,
        timeout_seconds=config.timeout_seconds,
    )
