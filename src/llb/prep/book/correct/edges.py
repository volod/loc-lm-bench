"""The facing block on the previous and next PDF pages, when those ledgers exist."""

from pathlib import Path

from llb.core.contracts.data_prep.book_blocks import Block
from llb.prep.book.cache import digest_json, page_dir
from llb.prep.book.constants import BLOCKS_FILENAME
from llb.prep.book.ledger.io import read_ledger
from llb.prep.book.ranges import page_id
from llb.prep.book.resolve.anchor import block_anchor
from llb.prep.book.resolve.primary import PrimaryChoice, select_reading
from llb.prep.book.correct.context import reading_order


def edge_text(
    run_dir: Path, pdf_page: int, page_count: int, choice: PrimaryChoice, *, last: bool
) -> str:
    """The primary text of the facing block, or empty when that page has no ledger yet."""
    block = _edge(run_dir, pdf_page, page_count, last=last)
    if block is None:
        return ""
    reading, _fallback = select_reading(block, choice)
    return reading.text


def neighbour_digest(run_dir: Path, current_page_id: str) -> dict[str, str]:
    """Digest of the two facing blocks. An interior change on a neighbour page is not in it."""
    page_count, pdf_page = _position(run_dir, current_page_id)
    prev = _edge_anchor(run_dir, pdf_page - 1, page_count, last=True)
    nxt = _edge_anchor(run_dir, pdf_page + 1, page_count, last=False)
    return {"neighbours": digest_json({"prev": prev, "next": nxt})}


def _position(run_dir: Path, current_page_id: str) -> tuple[int, int]:
    from llb.prep.book.config import read_config

    config = read_config(run_dir)
    return config.source.page_count, int(current_page_id.removeprefix("p"))


def _edge_anchor(run_dir: Path, pdf_page: int, page_count: int, *, last: bool) -> str:
    block = _edge(run_dir, pdf_page, page_count, last=last)
    if block is None:
        return ""
    return block_anchor(block)


def _edge(run_dir: Path, pdf_page: int, page_count: int, *, last: bool) -> Block | None:
    if page_count < 1 or pdf_page < 1 or pdf_page > page_count:
        return None
    path = page_dir(run_dir, page_id(pdf_page, page_count)) / BLOCKS_FILENAME
    if not path.is_file():
        return None
    ordered = reading_order(read_ledger(path).blocks)
    if not ordered:
        return None
    return ordered[-1] if last else ordered[0]
