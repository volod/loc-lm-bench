"""Run-target resolution for the runner: the config fingerprint payload, the run timestamp/id, and
the fresh-or-resumed staging directory.

`run_eval` calls `_eval_config_payload` then `_resolve_run_target` before any backend work, so an
interrupted run resumes its journaled staging dir instead of re-spending model calls.
"""

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llb.core.config import RunConfig
from llb.executor import durability, durability_journal
from llb.goldset.schema import GoldItem

_RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"


def _run_timestamp(run_id: str) -> str:
    now = datetime.now(timezone.utc).strftime(_RUN_TIMESTAMP_FORMAT)
    return f"{now}-{run_id}"


def _eval_config_payload(
    config: RunConfig,
    items: list[GoldItem],
    prompt_system_provenance: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Config fingerprint enriched with the adapter manifest and prompt-system id."""
    adapter_manifest = None
    if config.adapter_path is not None:
        from llb.finetune.guard import validate_adapter_for_eval
        from llb.finetune.registry.io import registry_path

        adapter_manifest = validate_adapter_for_eval(
            adapter_path=config.adapter_path,
            items=items,
            model=config.model,
            judge_model=config.judge_model,
            registry=registry_path(config.data_dir),
        )
    config_payload = config.fingerprint()
    if adapter_manifest is not None:
        config_payload["adapter"] = adapter_manifest
        label = adapter_manifest.get("adapter_label")
        if isinstance(label, str) and label:
            config_payload["model"] = label
    if prompt_system_provenance is not None:
        config_payload["prompt_system"] = prompt_system_provenance["prompt_system_id"]
    return config_payload


def _resolve_run_target(
    config: RunConfig,
    resume: Path | str | None,
    config_payload: dict[str, Any],
    items: list[GoldItem],
    split: str,
) -> tuple[str, str, Path, Path]:
    """Resume an interrupted run's staging dir, or start a fresh journaled one.

    Returns (run_timestamp, run_id, run_dir, staging_dir).
    """
    if resume is not None:
        run_timestamp, run_id, run_dir, staging_dir = durability.resume_target(
            config.run_dir, config.run_staging_dir, resume
        )
        if run_dir.exists():
            raise SystemExit(f"[run-eval] {run_dir} is already finalized; nothing to resume")
        if not staging_dir.exists():
            raise SystemExit(f"[run-eval] no interrupted run to resume at {staging_dir}")
        durability_journal.verify_resume_meta(
            staging_dir, config_fingerprint=config_payload, items=items, split=split
        )
        return run_timestamp, run_id, run_dir, staging_dir
    run_id = uuid.uuid4().hex[:12]
    run_timestamp = _run_timestamp(run_id)
    run_dir = config.run_dir(run_timestamp)
    staging_dir = config.run_staging_dir(run_timestamp)
    staging_dir.mkdir(parents=True, exist_ok=True)
    durability_journal.write_journal_meta(
        staging_dir,
        config_fingerprint=config_payload,
        items=items,
        run_id=run_id,
        split=split,
    )
    return run_timestamp, run_id, run_dir, staging_dir
