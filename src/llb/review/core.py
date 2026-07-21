"""Backend-independent record, ledger, progress, and navigation contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SectionRole = Literal["data", "evidence", "metadata"]
ActionTone = Literal["positive", "warning", "negative", "neutral"]


@dataclass(frozen=True, slots=True)
class ReviewSection:
    """One visually distinct part of a review record."""

    title: str
    text: str
    role: SectionRole


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """A persistence-neutral record shown by the workbench."""

    key: str
    title: str
    sections: tuple[ReviewSection, ...]
    stratum: str = "all"
    verdict: str = ""


@dataclass(frozen=True, slots=True)
class ReviewAction:
    """One adapter-specific verdict mutation exposed as a key and button."""

    key: str
    label: str
    value: str
    tone: ActionTone = "neutral"


@dataclass(frozen=True, slots=True)
class ReviewProgress:
    """Dataset, current-record, and current-stratum progress."""

    position: int
    total: int
    reviewed: int
    stratum: str
    stratum_total: int
    stratum_reviewed: int


class ReviewAdapter(ABC):
    """Thin ledger adapter; implementations must save through the legacy writer."""

    kind: str
    path: Path

    @property
    @abstractmethod
    def actions(self) -> tuple[ReviewAction, ...]:
        """Actions valid for the current ledger type."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of records across all ledgers represented by this adapter."""

    @abstractmethod
    def record(self, index: int) -> ReviewRecord:
        """Build the current neutral record from live ledger state."""

    @abstractmethod
    def apply(self, index: int, action: str) -> None:
        """Apply and immediately persist one action."""

    def first_pending(self) -> int | None:
        for index in range(len(self)):
            if not self.record(index).verdict:
                return index
        return None

    def progress(self, index: int) -> ReviewProgress:
        current = self.record(index)
        records = [self.record(i) for i in range(len(self))]
        stratum = [record for record in records if record.stratum == current.stratum]
        return ReviewProgress(
            position=index + 1,
            total=len(records),
            reviewed=sum(bool(record.verdict) for record in records),
            stratum=current.stratum,
            stratum_total=len(stratum),
            stratum_reviewed=sum(bool(record.verdict) for record in stratum),
        )

    def finish(self) -> None:
        """Optional post-review finalization hook."""


class ReviewNavigator:
    """Consistent bounded navigation and resume behavior for every adapter."""

    def __init__(self, adapter: ReviewAdapter, start: int | None = None) -> None:
        if len(adapter) == 0:
            raise ValueError(f"{adapter.path}: review ledger is empty")
        self.adapter = adapter
        pending = adapter.first_pending()
        self.index = min(max((start or 1) - 1, 0), len(adapter) - 1)
        if start is None and pending is not None:
            self.index = pending

    def next(self) -> int:
        self.index = min(self.index + 1, len(self.adapter) - 1)
        return self.index

    def previous(self) -> int:
        self.index = max(self.index - 1, 0)
        return self.index

    def next_pending(self) -> int:
        for offset in range(1, len(self.adapter) + 1):
            candidate = (self.index + offset) % len(self.adapter)
            if not self.adapter.record(candidate).verdict:
                self.index = candidate
                break
        return self.index

    def advance_after_verdict(self) -> int:
        if self.index < len(self.adapter) - 1:
            self.index += 1
        elif self.adapter.first_pending() is not None:
            self.index = self.adapter.first_pending() or 0
        return self.index
