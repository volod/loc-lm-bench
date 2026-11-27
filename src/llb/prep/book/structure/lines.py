"""One line of a block ledger, in the coordinates structure inference reads.

`kind` is stored and never consulted. The acceptance case relabels every block `body`
and the page mapping and the running headers stay the same, so a kind is not an input.
A resolved block contributes its number text: a number an unresolved span touches is a
placeholder, so a folio, a section number, or a caption number is never read across the gap.
"""

from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.data_prep.book import BookOcrRunConfig
from llb.core.contracts.data_prep.book_blocks import Block, PageBlockLedger
from llb.prep.book.constants import BLOCKS_FILENAME, PAGES_DIRNAME
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.ranges import page_id
from llb.prep.book.resolve.constants import RESOLVED_FILENAME
from llb.prep.book.resolve.io import read_resolved
from llb.prep.book.resolve.numbers import number_text
from llb.prep.book.structure.text import norm_space


@dataclass(frozen=True)
class Line:
    """One block reduced to the text and the box inference needs."""

    block_id: str
    pdf_page: int
    text: str
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: int
    page_height: int

    @property
    def y_frac(self) -> float:
        return self.y0 / self.page_height

    @property
    def x_frac(self) -> float:
        return self.x0 / self.page_width

    @property
    def center_frac(self) -> float:
        return ((self.x0 + self.x1) / 2) / self.page_width

    @property
    def width_frac(self) -> float:
        return (self.x1 - self.x0) / self.page_width


def load_lines(run_dir: Path, config: BookOcrRunConfig) -> tuple[list[Line], list[str]]:
    """Every block of the configured pages. The second list names pages with no ledger."""
    lines: list[Line] = []
    missing: list[str] = []
    for pdf_page in config.pages:
        pid = page_id(pdf_page, config.source.page_count)
        path = run_dir / PAGES_DIRNAME / pid / BLOCKS_FILENAME
        if not path.is_file():
            missing.append(pid)
            continue
        ledger = read_ledger(path)
        texts = _resolved_text(run_dir, pid)
        lines.extend(_page_lines(ledger, texts))
    return lines, missing


def _resolved_text(run_dir: Path, pid: str) -> dict[str, str]:
    path = run_dir / PAGES_DIRNAME / pid / RESOLVED_FILENAME
    if not path.is_file():
        return {}
    resolved = read_resolved(path)
    return {block.block_id: number_text(block) for block in resolved.blocks}


def _page_lines(ledger: PageBlockLedger, texts: dict[str, str]) -> list[Line]:
    lines: list[Line] = []
    for block in ledger.blocks:
        lines.append(_line(block, texts.get(block.block_id) or block.readings[0].text))
    return lines


def _line(block: Block, text: str) -> Line:
    region = block.region
    return Line(
        block_id=block.block_id,
        pdf_page=block.pdf_page,
        text=norm_space(text),
        kind=block.kind,
        x0=region.x0,
        y0=region.y0,
        x1=region.x1,
        y1=region.y1,
        page_width=region.width,
        page_height=region.height,
    )
