"""Rendering and persistence for Gemma controller-authority feedback transfer."""

from pathlib import Path

from llb.bench.loop_feedback.transfer_report import (
    format_feedback_transfer_table,
    persist_feedback_transfer,
)
from llb.bench.common import Mirror
from llb.core.contracts.runs import RunPaths

format_feedback_authority_table = format_feedback_transfer_table


def persist_feedback_authority(
    design: dict[str, object],
    analysis: dict[str, object],
    *,
    data_dir: Path | str,
    task_digest: str,
    table: str,
    mirror: Mirror | None = None,
) -> RunPaths:
    """Persist the authority decision beside all source policy-cell manifests."""
    return persist_feedback_transfer(
        design,
        analysis,
        data_dir=data_dir,
        task_digest=task_digest,
        table=table,
        mirror=mirror,
        artifact_stem="controller-authority-transfer",
        report_title="Gemma controller-authority feedback transfer",
    )
