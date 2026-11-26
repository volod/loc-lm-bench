"""Page revisions, scan-bound human acceptances, and immutable score inputs."""

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import Field

from llb.core.contracts.data_prep.book_pilot_review import (
    PILOT_PAGE_REVISION,
    PILOT_REFERENCE_ACCEPTANCE,
    PILOT_REVIEW_VERSION,
    PilotPageAcceptance,
    PilotPageRevision,
    PilotReferenceAcceptance,
    PilotReviewManifest,
)
from llb.core.contracts.data_prep.corpus import DataPrepRow
from llb.prep.book.cache import digest_json
from llb.prep.book.metrics.reference import RefPage, digest_pages, require_formulas
from llb.prep.book.pilot_review.prepare import write_json
from llb.prep.book.source import sha256_file


class SaveRequest(DataPrepRow):
    page: dict[str, Any]
    reviewer: str
    accept: bool
    base: int = Field(ge=0)


class StaleRevision(ValueError):
    """Another reviewer saved a newer revision before this edit arrived."""


class Review:
    """The latest revision is authoritative. Previous revisions stay available for restore."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = PilotReviewManifest.model_validate_json((root / "review.json").read_bytes())
        self.pages = {p.id: p for p in self.manifest.pages}

    def scan(self, page: str) -> Path:
        return self.root / "scans" / f"{self.pages[page].id}.png"

    def _predraft(self, page: str) -> Path:
        return self.root / "predraft" / "pages" / f"{self.pages[page].id}.json"

    def _history(self, page: str) -> Path:
        return self.root / "history" / self.pages[page].id

    def _revisions(self, page: str) -> list[int]:
        paths = self._history(page).glob("*.json")
        return [0, *sorted(int(p.stem) for p in paths if p.stem.isdecimal())]

    def _revision(self, page: str, version: int) -> PilotPageRevision:
        if version == 0:
            return PilotPageRevision(
                schema_id=PILOT_PAGE_REVISION,
                schema_version=PILOT_REVIEW_VERSION,
                page=json.loads(self._predraft(page).read_bytes()),
                reviewer="",
                accepted=False,
                saved_at=None,
            )
        path = self._history(page) / f"{version:06}.json"
        return PilotPageRevision.model_validate_json(path.read_bytes())

    def status(self) -> list[dict[str, Any]]:
        """Progress without loading page images or disagreements."""
        rows = []
        for entry in self.manifest.pages:
            version = self._revisions(entry.id)[-1]
            revision = self._revision(entry.id, version)
            rows.append(
                {
                    **entry.model_dump(mode="json", exclude={"scan_sha256", "notes"}),
                    "version": version,
                    "accepted": revision.accepted,
                }
            )
        return rows

    def worksheet(self, page: str) -> dict[str, Any]:
        versions = self._revisions(page)
        version = versions[-1]
        return {
            "version": version,
            "versions": versions,
            **self._revision(page, version).model_dump(mode="json"),
            "flags": [note.model_dump(mode="json") for note in self.pages[page].notes],
        }

    def earlier_page(self, page: str, version: int) -> dict[str, Any]:
        return self._revision(page, version).page

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with (self.root / ".write.lock").open("ab") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def save(self, page: str, request: SaveRequest) -> int:
        """Write one complete revision. An edited page needs human acceptance again."""
        with self._lock():
            latest = self._revisions(page)[-1]
            if latest != request.base:
                raise StaleRevision("A newer revision exists. Reload before saving this page.")
            reviewer = request.reviewer.strip()
            content = request.page
            if request.accept:
                if not reviewer:
                    raise ValueError("enter a reviewer name before accepting a page")
                content = self._accepted_page(page, content).model_dump(mode="json")
            revision = PilotPageRevision(
                schema_id=PILOT_PAGE_REVISION,
                schema_version=PILOT_REVIEW_VERSION,
                page=content,
                reviewer=reviewer,
                accepted=request.accept,
                saved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            write_json(self._history(page) / f"{latest + 1:06}.json", revision.model_dump())
            return latest + 1

    def _accepted_page(self, page: str, raw: dict[str, Any]) -> RefPage:
        entry = self.pages[page]
        scan = self.scan(page)
        if sha256_file(scan) != entry.scan_sha256:
            raise ValueError(f"{page} scan checksum changed")
        original = self._revision(page, 0).page
        draft = RefPage.model_validate(raw)
        if (draft.pdf_page, draft.page_class) != (entry.number, entry.page_class):
            raise ValueError(f"{page} PDF page and class must match the frozen pilot")
        if (draft.width, draft.height) != (original["width"], original["height"]):
            raise ValueError(f"{page} coordinates must use the original scan dimensions")
        require_formulas([draft])
        return draft

    def finalize(self) -> Path:
        """Publish all accepted pages with a metadata-only record that can travel separately."""
        with self._lock():
            missing = [row["id"] for row in self.status() if not row["accepted"]]
            if missing:
                raise ValueError(f"awaiting acceptance for {', '.join(missing)}")
            if digest_pages(self.root / "predraft" / "pages") != self.manifest.predraft_sha256:
                raise ValueError("unedited pre-draft checksum changed")
            return self._publish()

    def _publish(self) -> Path:
        references = self.root / "references"
        references.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=".reference-", dir=references) as directory:
            export = Path(directory) / "reference"
            rows = []
            for entry in self.manifest.pages:
                version = self._revisions(entry.id)[-1]
                revision = self._revision(entry.id, version)
                if not revision.reviewer or not revision.saved_at:
                    raise ValueError(f"{entry.id} is missing its reviewer or acceptance time")
                page = self._accepted_page(entry.id, revision.page)
                path = export / "pages" / f"{entry.id}.json"
                write_json(path, page.model_dump(mode="json"))
                rows.append(
                    PilotPageAcceptance(
                        pdf_page=entry.number,
                        reviewer=revision.reviewer,
                        accepted_at=revision.saved_at,
                        scan_sha256=entry.scan_sha256,
                        page_sha256=sha256_file(path),
                        revision=version,
                    )
                )
            evidence = PilotReferenceAcceptance(
                schema_id=PILOT_REFERENCE_ACCEPTANCE,
                schema_version=PILOT_REVIEW_VERSION,
                source_sha256=self.manifest.source_sha256,
                pilot_sha256=self.manifest.pilot_sha256,
                predraft_sha256=self.manifest.predraft_sha256,
                reference_sha256=digest_pages(export / "pages"),
                pages=rows,
            )
            payload = evidence.model_dump(mode="json")
            write_json(export / "acceptance.json", payload)
            target = references / digest_json(payload)
            if not target.exists():
                export.rename(target)
            elif (target / "acceptance.json").read_bytes() != (
                export / "acceptance.json"
            ).read_bytes() or digest_pages(target / "pages") != evidence.reference_sha256:
                raise ValueError(f"existing reference {target} failed checksum validation")
            return target
