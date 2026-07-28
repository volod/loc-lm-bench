"""Orchestrate provider resolution, checkpoint recovery, verification, and transfer."""

import time
from dataclasses import replace
from typing import Any, Callable

from llb.backends.model_download.contracts import (
    DownloadConfig,
    DownloadIntegrityError,
    DownloadReport,
    ProgressCallback,
    SnapshotState,
)
from llb.backends.model_download.providers import (
    fetch_manifest,
    normalize_provider,
    normalized_revision,
    provider_token,
)
from llb.backends.model_download.state import (
    load_state,
    recover_file,
    save_state,
    target_lock,
    validate_identity,
)
from llb.backends.model_download.transfer import transfer_snapshot


def _report(
    config: DownloadConfig,
    state: SnapshotState,
    session_downloaded_bytes: int,
    status: str,
) -> DownloadReport:
    return DownloadReport(
        provider=state.provider,
        repo_id=state.repo_id,
        resolved_revision=state.resolved_revision,
        target_dir=config.target_dir,
        total_bytes=state.total_bytes,
        completed_bytes=state.completed_bytes,
        session_downloaded_bytes=session_downloaded_bytes,
        complete_files=sum(file.complete for file in state.files),
        total_files=len(state.files),
        status=status,
    )


def _load_or_fetch(
    config: DownloadConfig,
    *,
    api: Any | None,
    client: Any | None,
) -> tuple[SnapshotState, bool]:
    existing = load_state(config.target_dir)
    if existing is not None:
        validate_identity(
            existing,
            config.provider,
            config.repo_id,
            normalized_revision(config.provider, config.repo_id, config.revision),
        )
        return existing, False
    if config.verify_only:
        raise DownloadIntegrityError(
            f"no checkpoint found under {config.target_dir}; nothing can be verified"
        )
    state = fetch_manifest(
        config,
        api=api,
        client=client,
    )
    return state, True


def _recover(
    config: DownloadConfig,
    state: SnapshotState,
    progress: ProgressCallback | None,
) -> bool:
    changed = False
    for file in state.files:
        file_changed = recover_file(
            config.target_dir,
            file,
            config.verify_completed or config.verify_only,
        )
        changed = changed or file_changed
        if file_changed and progress:
            progress(f"recovered {file.path} to {file.downloaded_bytes} verified bytes")
    return changed


def download_model(
    config: DownloadConfig,
    *,
    api: Any | None = None,
    client: Any | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    progress: ProgressCallback | None = None,
) -> DownloadReport:
    """Download one provider-pinned model snapshot into a directly usable local directory."""
    config.validate()
    provider = normalize_provider(config.provider)
    effective = replace(
        config,
        provider=provider,
        token=provider_token(config, provider),
    )

    if config.dry_run:
        state, _created = _load_or_fetch(effective, api=api, client=client)
        return _report(effective, state, 0, "planned")

    effective.target_dir.mkdir(parents=True, exist_ok=True)
    with target_lock(effective.target_dir):
        state, created = _load_or_fetch(effective, api=api, client=client)
        changed = _recover(effective, state, progress)
        if created or changed:
            save_state(effective.target_dir, state)
        if effective.verify_only:
            incomplete = [file.path for file in state.files if not file.complete]
            if incomplete:
                raise DownloadIntegrityError(
                    f"verification found {len(incomplete)} incomplete or corrupt files"
                )
            return _report(effective, state, 0, "verified")

        session_bytes = transfer_snapshot(
            effective,
            state,
            client=client,
            sleeper=sleeper,
            progress=progress,
        )
        status = "complete" if state.completed_bytes == state.total_bytes else "checkpointed"
        return _report(effective, state, session_bytes, status)
