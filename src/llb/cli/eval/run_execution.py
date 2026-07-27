"""Prompt and adapter resolution for the run-eval command."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from llb.cli.helpers import resolve_registered_adapter

if TYPE_CHECKING:
    from llb.core.config import RunConfig


def execute_eval(
    cfg: "RunConfig",
    *,
    adapter: Optional[str],
    prompt_system: Optional[str],
    prompt_package: Optional[Path],
    split: str,
    limit: Optional[int],
    judge_rho: Optional[float],
    worksheet: Optional[Path],
    evict: bool,
    wait: bool,
    resume: Optional[Path],
    max_case_retries: int,
    retry_backoff_s: float,
) -> None:
    """Resolve optional run inputs and invoke the evaluator."""
    from llb.executor.runner import run_eval
    from llb.prompt_system.selection import (
        prompt_system_id_from_package_path,
        resolve_prompt_package,
    )

    if adapter is not None:
        cfg = cfg.with_overrides(adapter_path=resolve_registered_adapter(cfg.data_dir, adapter))
    selected_prompt = None
    prompt_id = prompt_system or prompt_system_id_from_package_path(prompt_package)
    if prompt_id is not None:
        selected_prompt = resolve_prompt_package(cfg.data_dir, prompt_id, prompt_package)
    run_eval(
        cfg,
        split=split,
        limit=limit,
        judge_rho=judge_rho,
        worksheet=worksheet,
        evict=evict,
        wait=wait,
        resume=resume,
        max_case_retries=max_case_retries,
        retry_backoff_s=retry_backoff_s,
        prompt_package=selected_prompt.package if selected_prompt is not None else None,
        prompt_system_provenance=(
            selected_prompt.provenance if selected_prompt is not None else None
        ),
    )
