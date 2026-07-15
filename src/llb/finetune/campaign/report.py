"""Campaign report rendering and the `latest_campaign` lookup for the recommendation summary."""

from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.fsutil import atomic_write_text
from llb.finetune.campaign.coerce import _fmt
from llb.finetune.campaign.model import (
    COMPLETE_VERDICT,
    PROGRESS_FILENAME,
    REPORT_FILENAME,
    CampaignEntry,
)
from llb.finetune.campaign.state import _read_completed_entries


def _write_report(
    root: Path, entries: list[CampaignEntry], shared_dataset_dir: Path | None
) -> None:
    ranked = sorted(
        [entry for entry in entries if entry.status == COMPLETE_VERDICT],
        key=lambda entry: (
            entry.delta if entry.delta is not None else float("-inf"),
            -(entry.train_wall_clock_s or 0.0),
            -(entry.peak_vram_mb or 0.0),
        ),
        reverse=True,
    )
    lines = [
        "# Fine-tune campaign report",
        "",
        f"Shared dataset: `{shared_dataset_dir}`" if shared_dataset_dir else "Shared dataset: n/a",
        "",
        "| rank | model | base objective | tuned objective | delta | train s | peak VRAM | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rank_by_model = {entry.model: idx for idx, entry in enumerate(ranked, 1)}
    ordered = sorted(entries, key=lambda entry: rank_by_model.get(entry.model, 10_000))
    for entry in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank_by_model.get(entry.model, "")),
                    entry.model,
                    _fmt(entry.base_objective),
                    _fmt(entry.tuned_objective),
                    _fmt(entry.delta),
                    _fmt(entry.train_wall_clock_s),
                    _fmt(entry.peak_vram_mb),
                    entry.status if entry.reason is None else f"{entry.status}: {entry.reason}",
                ]
            )
            + " |"
        )
    atomic_write_text(root / REPORT_FILENAME, "\n".join(lines) + "\n")


def latest_campaign(data_dir: Path | str) -> JsonObject | None:
    """Newest `$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` with report path attached."""
    root = Path(data_dir) / "finetune-campaign"
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        progress = candidate / PROGRESS_FILENAME
        if not progress.is_file():
            continue
        entries = [entry.as_dict() for entry in _read_completed_entries(candidate).values()]
        if not entries:
            continue
        return {
            "campaign_dir": str(candidate),
            "report_path": str(candidate / REPORT_FILENAME),
            "entries": entries,
        }
    return None
