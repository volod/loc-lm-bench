"""The records a vault check passes between its steps."""

from dataclasses import dataclass, field
from pathlib import Path

from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookPageManifest
from llb.core.contracts.data_prep.book_structure import DocumentStructure


@dataclass(frozen=True, slots=True)
class Finding:
    """One refused check. `file` is the note or asset that failed it."""

    check: str
    file: str
    detail: str


@dataclass(frozen=True, slots=True)
class PageRow:
    """How one PDF page was accounted for."""

    pdf_page: int
    page_id: str
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class VaultDoc:
    """One Markdown file in the vault, front matter already split off."""

    rel: str
    path: Path
    matter: dict[str, str]
    body: str


@dataclass(frozen=True, slots=True)
class VaultLink:
    """One link a reader can follow."""

    source: str
    source_path: Path
    kind: str
    label: str
    target: str
    fragment: str
    line: int
    located: Path
    exists: bool


@dataclass(frozen=True, slots=True)
class OpenMark:
    """One open review entry, from the queue or from the ledger."""

    kind: str
    block_id: str
    start: int
    end: int
    pdf_page: int
    block_kind: str
    classes: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str, int, int, int]:
        return (self.kind, self.block_id, self.start, self.end, self.pdf_page)


@dataclass
class EngineScores:
    """Confidence counts for one engine. The scores themselves are not kept."""

    count: int = 0
    missing: int = 0
    minimum: float | None = None
    maximum: float | None = None
    buckets: dict[str, int] = field(default_factory=dict)

    def add(self, score: float | None, bucket: str) -> None:
        self.buckets.setdefault(bucket, 0)
        if score is None:
            self.missing += 1
            return
        self.count += 1
        self.buckets[bucket] = self.buckets.get(bucket, 0) + 1
        self.minimum = score if self.minimum is None else min(self.minimum, score)
        self.maximum = score if self.maximum is None else max(self.maximum, score)


@dataclass
class RunFacts:
    """What the run contributes, apart from the vault files."""

    marks: list[OpenMark]
    queue_marks: list[OpenMark] | None
    orphans: int
    orphaned_by_reason: dict[str, int]
    scores: dict[str, EngineScores]
    readings: list[tuple[str, str, str, int]]
    findings: list[Finding]
    failed_pages: tuple[str, ...] = ()


@dataclass
class BookContext:
    """A vault and the run that rendered it."""

    book_dir: Path
    run_dir: Path
    config: BookOcrRunConfig
    structure: DocumentStructure
    manifest: BookPageManifest | None
    fragment_name: str
    docs: list[VaultDoc]
    toc: VaultDoc | None
    content: list[VaultDoc]
    links: list[VaultLink]


@dataclass
class ValidationReport:
    """The outcome written into the book directory."""

    ok: bool
    findings: tuple[Finding, ...]
    pages: tuple[PageRow, ...]
    warnings: tuple[str, ...]
    facts: RunFacts
    context: BookContext
