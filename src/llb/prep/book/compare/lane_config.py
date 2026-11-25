"""The book-ocr request one roster entry runs on the pilot pages.

A page reader runs every recognition stage it serves, then the ledger and the
resolve stage, so the run can be rendered. A variant policy runs RapidOCR on
that variant only. RapidOCR is a CPU engine: its run config budgets zero
megabytes even when the dewarper itself needs a little GPU. The lane record
keeps the dewarper's declared budget for the comparison.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.core.contracts.data_prep.book_classes import PilotSelection
from llb.prep.book.compare.constants import ROLE_PAGE
from llb.prep.book.compare.manifest import registry_serves
from llb.prep.book.compare.roster import RosterEntry
from llb.prep.book.config import RunRequest
from llb.prep.book.constants import DEFAULT_BATCH_PAGES, DEFAULT_SOURCE_POLICY
from llb.prep.book.preprocess.constants import DEWARP_OPENCV
from llb.prep.book.source import sha256_file
from llb.prep.book.stages.recognize import RECOGNITION_STAGES

_BLOCKS = "blocks"
_RESOLVE = "resolve"
_PREP = ("render", "signals", "classes")


def require_source(pdf: Path, selection: PilotSelection) -> str:
    """The PDF digest, or a refusal when the pilot was drawn from another file."""
    digest = sha256_file(pdf)
    if digest != selection.source_sha256:
        raise ValueError(
            f"pilot source sha256 {selection.source_sha256} does not match {pdf.name} ({digest})"
        )
    return digest


def pilot_pages(selection: PilotSelection) -> list[int]:
    """PDF pages in the selection, refusing a repeated page."""
    pages = [page.pdf_page for page in selection.pages]
    if len(pages) != len(set(pages)):
        raise ValueError("pilot selection repeats a pdf page")
    return pages


def pages_spec(pages: Sequence[int]) -> str:
    """A ``--pages`` value. The numbers are already a set, so they do not overlap."""
    if not pages:
        raise ValueError("pilot selection has no pages")
    return ",".join(str(page) for page in sorted(pages))


def primary_counts(run_dir: Path) -> dict[str, int]:
    """How many pages route on each primary class. A page is counted once."""
    from llb.prep.book.classes.records import load_page_classes

    counts: dict[str, int] = {}
    seen: set[int] = set()
    for record in load_page_classes(run_dir):
        if record.pdf_page in seen:
            continue
        seen.add(record.pdf_page)
        counts[record.primary] = counts.get(record.primary, 0) + 1
    if not counts:
        raise ValueError(f"{run_dir} has no page class records")
    return counts


def lane_request(
    pdf: Path,
    pages: Sequence[int],
    entry: RosterEntry,
    language: str,
    reuse_from: Path | None,
    *,
    serves: Callable[[str, str], bool] | None = None,
) -> RunRequest:
    """The run request for one roster entry. ``serves`` defaults to the registry."""
    if not language:
        raise ValueError("pilot OCR language is empty")
    check = serves or registry_serves
    return RunRequest(
        source=pdf,
        pages=pages_spec(pages),
        batch_pages=DEFAULT_BATCH_PAGES,
        source_policy=DEFAULT_SOURCE_POLICY,
        stages=_stages(entry, language, check),
        reuse_from=reuse_from,
    )


def _stages(
    entry: RosterEntry, language: str, serves: Callable[[str, str], bool]
) -> tuple[BookStageSpec, ...]:
    recognition = [name for name in RECOGNITION_STAGES if serves(entry.engine, name)]
    if not recognition:
        raise ValueError(f"{entry.engine} serves no recognition stage")
    specs = [BookStageSpec(name=name) for name in _PREP]
    if entry.variant != "original":
        specs.append(
            BookStageSpec(
                name="variants",
                params={"dewarp": entry.dewarp or DEWARP_OPENCV},
            )
        )
    budget = entry.declared_vram_mb if entry.role == ROLE_PAGE else 0
    params = {
        "engine": entry.engine,
        "variant": entry.variant,
        "ocr_language": language,
        "vram_mb": {entry.engine: budget},
    }
    specs.extend(BookStageSpec(name=name, params=dict(params)) for name in recognition)
    specs.append(BookStageSpec(name=_BLOCKS))
    specs.append(
        BookStageSpec(
            name=_RESOLVE,
            params={
                "primary": {
                    "stage": recognition[0],
                    "engine": entry.engine,
                    "variant": entry.variant,
                }
            },
        )
    )
    return tuple(specs)
