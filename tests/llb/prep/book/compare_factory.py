"""Reference pages and lanes for the pilot-comparison tests. Not collected by pytest."""

from llb.prep.book.compare.assemble import LaneInput
from llb.prep.book.compare.constants import ROLE_PAGE, STATUS_MEASURED
from llb.prep.book.compare.inputs import AcceptRecord, PageAccept
from llb.prep.book.metrics.constants import LANE_EXISTING
from llb.prep.book.metrics.geometry import Box
from llb.prep.book.metrics.reference import RefBlock, RefPage, RefRegion
from llb.prep.book.metrics.submission import HypBlock, PageSubmission

BASELINE = LANE_EXISTING


def page(page_class: str, pdf_page: int, text: str) -> RefPage:
    """One body block. The box sits inside a 100 by 100 page."""
    return RefPage(
        pdf_page=pdf_page,
        page_class=page_class,
        printed_page="1",
        width=100,
        height=100,
        blocks=[
            RefBlock(
                block_id="body",
                kind="body",
                region=RefRegion(x0=0, y0=0, x1=10, y1=10),
                reading_order=0,
                text=text,
            )
        ],
    )


def submission(lane: str, pdf_page: int, text: str, *, text_only: bool = False) -> PageSubmission:
    """One body reading. The existing lane carries no region."""
    region = None if text_only else Box(0.0, 0.0, 10.0, 10.0)
    block = HypBlock("body", "body", region, 0, text, None, None, None, None)
    return PageSubmission(lane, pdf_page, (block,), text_only)


def accepts_for(pages: list[RefPage]) -> AcceptRecord:
    """Every page accepted by one reviewer."""
    return AcceptRecord(
        source_sha256="a" * 64,
        pages=[
            PageAccept(pdf_page=item.pdf_page, reviewer="reviewer", accepted=True) for item in pages
        ],
    )


def candidate(
    lane: str,
    engine: str,
    *,
    role: str = ROLE_PAGE,
    vram_mb: int = 1000,
    variant: str = "original",
    dewarp: str | None = None,
    seconds_per_page: float | None = None,
    status: str = STATUS_MEASURED,
    detail: str = "",
) -> LaneInput:
    """A manifest lane with the fields the decision reads."""
    return LaneInput(
        lane,
        engine,
        role,
        vram_mb,
        variant,
        dewarp,
        seconds_per_page,
        status,
        detail,
    )


def serves(engine: str, stage: str) -> bool:
    """Docling and the test reader serve every recognition stage. RapidOCR serves OCR."""
    if engine in {"docling", "reader"}:
        return stage in {"layout", "ocr", "formula", "table"}
    if engine == "rapidocr":
        return stage == "ocr"
    return False
