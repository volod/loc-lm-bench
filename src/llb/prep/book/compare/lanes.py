"""One candidate a comparison manifest names."""

from dataclasses import dataclass

from llb.prep.book.compare.constants import STATUS_MEASURED


@dataclass(frozen=True)
class LaneInput:
    """One candidate the manifest names. Only a measured lane is scored."""

    lane: str
    engine: str
    role: str
    vram_mb: int
    variant: str = "original"
    dewarp: str | None = None
    seconds_per_page: float | None = None
    status: str = STATUS_MEASURED
    detail: str = ""
