"""The pilot roster and the rule that runs one engine at a time.

Declared VRAM is the survey peak, not a measurement from this comparison. An entry
above the host budget is not started. The ones that fit are handed to ``run_one``
in roster order, and the next entry is not handed over until that call returns.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llb.prep.book.compare.constants import ROLE_PAGE, ROLE_VARIANT


def _mib(gigabytes: float) -> int:
    return round(gigabytes * 1024)


@dataclass(frozen=True)
class RosterEntry:
    """One candidate the pilot may run."""

    lane: str
    engine: str
    role: str
    declared_vram_mb: int
    variant: str = "original"
    dewarp: str | None = None


def _page(engine: str, gigabytes: float) -> RosterEntry:
    return RosterEntry(f"book-ocr/{engine}", engine, ROLE_PAGE, _mib(gigabytes), "original", None)


def _policy(suffix: str, variant: str, dewarp: str | None, gigabytes: float) -> RosterEntry:
    lane = "book-ocr/rapidocr" if suffix == "original" else f"book-ocr/rapidocr@{suffix}"
    return RosterEntry(lane, "rapidocr", ROLE_VARIANT, _mib(gigabytes), variant, dewarp)


# Page readers, in the survey's reading order, then the line-engine variant policies.
PILOT_ROSTER: tuple[RosterEntry, ...] = (
    _page("marker-balanced", 10.3),
    _page("mineru-standard", 2.8),
    _page("docling-vlm-mineru2-pro", 3.5),
    _page("dots-ocr", 11.3),
    _page("paddleocr-vl", 3.2),
    _page("deepseek-ocr-2", 11.2),
    _page("olmocr-2", 11.7),
    _policy("original", "original", None, 0),
    _policy("deskew", "deskew", None, 0),
    _policy("illumination", "illumination", None, 0),
    _policy("dewarp-opencv", "dewarp", "opencv", 0),
    _policy("dewarp-page", "dewarp", "page-dewarp", 0),
    _policy("dewarp-uvdoc", "dewarp", "uvdoc", 0.4),
)


@dataclass(frozen=True)
class Skipped:
    """A roster entry the host was not asked to start."""

    entry: RosterEntry
    reason: str
    detail: str


def partition(
    entries: Sequence[RosterEntry], host_mb: int | None
) -> tuple[list[RosterEntry], list[Skipped]]:
    """Split the roster into what this host can start and what it must skip."""
    budget = 0 if host_mb is None else host_mb
    runnable: list[RosterEntry] = []
    skipped: list[Skipped] = []
    for entry in entries:
        if entry.declared_vram_mb <= budget:
            runnable.append(entry)
            continue
        have = "no CUDA device" if host_mb is None else f"{host_mb} MB"
        skipped.append(
            Skipped(
                entry,
                "over-vram",
                f"needs {entry.declared_vram_mb} MB VRAM and this host has {have}",
            )
        )
    return runnable, skipped


def run_roster(
    entries: Sequence[RosterEntry],
    host_mb: int | None,
    run_one: Callable[[RosterEntry], None],
) -> list[Skipped]:
    """Call ``run_one`` for each entry that fits, one after another."""
    runnable, skipped = partition(entries, host_mb)
    for entry in runnable:
        run_one(entry)
    return skipped
