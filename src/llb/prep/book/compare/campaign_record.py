"""The campaign index. It records which lanes ran. It is not a comparison report.

Recovered page text stays in the lane run directories. This file is how a killed
campaign resumes, and how a later compare manifest is filled in once a reviewer
has accepted the pilot reference.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from llb.core.contracts.data_prep.corpus import DataPrepRow
from llb.core.fsutil import atomic_write_text
from llb.prep.book.compare.constants import BASELINE_VRAM_MIB

CAMPAIGN_FILENAME = "campaign.json"
CAMPAIGN_KIND: Literal["book-ocr-pilot-campaign"] = "book-ocr-pilot-campaign"
LaneStatus = Literal["pending", "running", "measured", "over-vram", "unavailable"]
BaselineStatus = Literal["pending", "measured", "unavailable"]


def _widen(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, int):
        return value
    return float(value)


class CampaignBaseline(DataPrepRow):
    """The existing Docling lane, with its language recorded on the sidecar."""

    status: BaselineStatus
    detail: str = ""
    citations: str | None = None
    seconds_per_page: float | None = None
    vram_mb: int = Field(default=BASELINE_VRAM_MIB, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _seconds(cls, data: object) -> object:
        if not isinstance(data, dict) or "seconds_per_page" not in data:
            return data
        return {**data, "seconds_per_page": _widen(data["seconds_per_page"])}


class CampaignLane(DataPrepRow):
    """One roster entry and what this host did with it."""

    lane: str = Field(min_length=1)
    engine: str = Field(min_length=1)
    role: Literal["page-reader", "variant-policy"]
    vram_mb: int = Field(ge=0)
    variant: str = "original"
    dewarp: str | None = None
    status: LaneStatus
    detail: str = ""
    run: str | None = None
    seconds_per_page: float | None = None
    failed_pages: list[int] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _seconds(cls, data: object) -> object:
        if not isinstance(data, dict) or "seconds_per_page" not in data:
            return data
        return {**data, "seconds_per_page": _widen(data["seconds_per_page"])}


class CampaignRecord(DataPrepRow):
    """Everything ``pilot-run`` needs to resume, and no recovered page text."""

    kind: Literal["book-ocr-pilot-campaign"] = CAMPAIGN_KIND
    language: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pilot: str = Field(min_length=1)
    reuse_from: str | None = None
    host_vram_mb: int | None = None
    class_counts: dict[str, int] = Field(default_factory=dict)
    pages: list[int] = Field(min_length=1)
    baseline: CampaignBaseline
    lanes: list[CampaignLane] = Field(default_factory=list)


def read_campaign(path: Path) -> CampaignRecord:
    """The campaign index, refusing a file that is not one."""
    try:
        return CampaignRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not a pilot campaign: {exc}") from exc


def write_campaign(bundle: Path, record: CampaignRecord) -> None:
    """Replace the campaign index. A crash keeps the previous complete file."""
    payload = record.model_dump_json(indent=2) + "\n"
    atomic_write_text(bundle / CAMPAIGN_FILENAME, payload)


def comparison_payload(
    record: CampaignRecord, *, reference: str, predraft: str, accepts: str
) -> dict[str, object]:
    """The compare manifest. Refuses a baseline that was not measured."""
    citations = record.baseline.citations
    if record.baseline.status != "measured" or not citations:
        raise ValueError("the existing lane has no citation sidecar")
    lanes = []
    for lane in record.lanes:
        if lane.status in {"pending", "running"}:
            raise ValueError(f"lane {lane.lane} has not finished")
        lanes.append(
            {
                "lane": lane.lane,
                "engine": lane.engine,
                "role": lane.role,
                "vram_mb": lane.vram_mb,
                "run": lane.run,
                "variant": lane.variant,
                "dewarp": lane.dewarp,
                "seconds_per_page": lane.seconds_per_page,
                "status": _manifest_status(lane.status),
                "detail": lane.detail,
            }
        )
    return {
        "language": record.language,
        "reference": reference,
        "predraft": predraft,
        "accepts": accepts,
        "citations": citations,
        "baseline_vram_mb": record.baseline.vram_mb,
        "baseline_seconds_per_page": record.baseline.seconds_per_page,
        "class_counts": dict(record.class_counts),
        "source_sha256": record.source_sha256,
        "lanes": lanes,
    }


def _manifest_status(status: str) -> str:
    if status == "measured":
        return "measured"
    if status == "over-vram":
        return "over-vram"
    return "unavailable"
