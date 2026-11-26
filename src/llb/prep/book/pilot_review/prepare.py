"""Assemble the portable review bundle from two existing recognition ledgers."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from llb.core.contracts.data_prep.book_classes import PilotPageEntry, PilotSelection
from llb.core.contracts.data_prep.book_pilot_review import (
    PILOT_REVIEW_MANIFEST,
    PILOT_REVIEW_VERSION,
    PilotReviewManifest,
    PilotScanPage,
)
from llb.core.fsutil import atomic_write_text
from llb.prep.book.cache import read_record
from llb.prep.book.config import read_config
from llb.prep.book.engines.image import png_size
from llb.prep.book.metrics.reference import RefPage, digest_pages
from llb.prep.book.metrics.submission import PageSubmission, load_run
from llb.prep.book.pilot_review.merge import merge_readings, printed_page
from llb.prep.book.ranges import page_id
from llb.prep.book.source import sha256_file


def write_json(path: Path, payload: object) -> None:
    """Commit a complete UTF-8 JSON document."""
    atomic_write_text(path, json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


@dataclass(frozen=True)
class DraftInput:
    source: Path
    pilot: Path
    primary: Path
    alternate: Path
    alternate_override: Path | None = None


def _reading(
    run: Path, pilot: PilotSelection, *, partial: bool = False
) -> dict[int, PageSubmission]:
    config = read_config(run)
    if config.source.sha256 != pilot.source_sha256:
        raise ValueError(f"{run} has a different source checksum from the frozen pilot")
    lanes = load_run(run, None)
    if len({page.lane for page in lanes}) != 1 or len({p.pdf_page for p in lanes}) != len(lanes):
        raise ValueError(f"{run} must supply one engine ledger per page")
    by_page = {page.pdf_page: page for page in lanes}
    missing = [p.pdf_page for p in pilot.pages if p.pdf_page not in by_page]
    if missing and not partial:
        raise ValueError(f"{run} is missing pilot ledgers for PDF pages {missing}")
    return by_page


def _scan(run: Path, entry: PilotPageEntry, count: int) -> Path:
    expected = page_id(entry.pdf_page, count)
    if entry.page_id != expected:
        raise ValueError(
            f"frozen page {entry.pdf_page} has id {entry.page_id}, expected {expected}"
        )
    render = run / "pages" / expected / "render"
    record = read_record(render)
    image = render / "page.png"
    if record is None or record.status != "done":
        raise ValueError(f"{render} has no completed render record")
    if not any(
        out.path == image.name and out.sha256 == sha256_file(image) for out in record.outputs
    ):
        raise ValueError(f"{image} differs from the completed render record")
    return image


def _write_page(
    root: Path, entry: PilotPageEntry, source: Path, first: PageSubmission, second: PageSubmission
) -> PilotScanPage:
    width, height = png_size(source)
    blocks, flags = merge_readings(first, second)
    page = RefPage.model_validate(
        {
            "pdf_page": entry.pdf_page,
            "page_class": entry.represents,
            "printed_page": printed_page(blocks),
            "width": width,
            "height": height,
            "blocks": blocks,
        }
    )
    name = entry.page_id
    write_json(root / "predraft" / "pages" / f"{name}.json", page.model_dump(mode="json"))
    scan = root / "scans" / f"{name}.png"
    shutil.copyfile(source, scan)
    return PilotScanPage.model_validate(
        {
            "id": name,
            "number": entry.pdf_page,
            "page_class": entry.represents,
            "orientation": entry.signals.orientation_deg,
            "scan_sha256": sha256_file(scan),
            "notes": flags,
        }
    )


def prepare(inputs: DraftInput, output: Path) -> None:
    """Keep the immutable blind pre-draft, disagreements and scan for each selected page."""
    if output.exists():
        raise ValueError(f"{output} already exists; open the existing review to resume")
    pilot = PilotSelection.model_validate_json(inputs.pilot.read_bytes())
    if not pilot.pages or pilot.size != len(pilot.pages):
        raise ValueError("frozen pilot size must match its page list")
    if sha256_file(inputs.source) != pilot.source_sha256:
        raise ValueError("source PDF checksum differs from the frozen pilot")
    primary = _reading(inputs.primary, pilot)
    alternate = _reading(inputs.alternate, pilot)
    if next(iter(primary.values())).lane == next(iter(alternate.values())).lane:
        raise ValueError("both ledger runs use the same engine")
    override = (
        _reading(inputs.alternate_override, pilot, partial=True)
        if inputs.alternate_override
        else {}
    )
    if any(
        number not in alternate or page.lane != alternate[number].lane
        for number, page in override.items()
    ):
        raise ValueError("alternate override must replace existing pages with the same engine")
    count = read_config(inputs.primary).source.page_count
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pilot-review-", dir=output.parent) as directory:
        root = Path(directory) / "bundle"
        (root / "scans").mkdir(parents=True)
        entries = []
        for entry in sorted(pilot.pages, key=lambda page: page.pdf_page):
            scan = _scan(inputs.primary, entry, count)
            other_run = (
                inputs.alternate_override if entry.pdf_page in override else inputs.alternate
            )
            assert other_run is not None
            other_scan = _scan(other_run, entry, count)
            if sha256_file(scan) != sha256_file(other_scan):
                raise ValueError(f"pilot page {entry.pdf_page} has different original renders")
            entries.append(
                _write_page(
                    root,
                    entry,
                    scan,
                    primary[entry.pdf_page],
                    override.get(entry.pdf_page, alternate[entry.pdf_page]),
                )
            )
        manifest = PilotReviewManifest(
            schema_id=PILOT_REVIEW_MANIFEST,
            schema_version=PILOT_REVIEW_VERSION,
            source_sha256=pilot.source_sha256,
            pilot_sha256=sha256_file(inputs.pilot),
            predraft_sha256=digest_pages(root / "predraft" / "pages"),
            pages=entries,
        )
        write_json(root / "review.json", manifest.model_dump(mode="json"))
        root.rename(output)
