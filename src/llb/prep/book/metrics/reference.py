"""The reference transcription a recovery score is computed against.

One directory, one JSON file per PDF page under ``pages/``. A page names its class, its printed
page number, and every block: kind, region, reading order, text, LaTeX for a formula, a cell tree
for a table, and a caption id for a figure. ``llb book-ocr score`` refuses a directory that breaks
these rules before it reports a number.
"""

import json
from pathlib import Path

from pydantic import Field, ValidationError, model_validator

from llb.core.contracts.data_prep.book_blocks import BlockKind
from llb.core.contracts.data_prep.book_pilot_review import PilotReferenceAcceptance
from llb.core.contracts.data_prep.corpus import DataPrepRow
from llb.prep.book.cache import digest_json
from llb.prep.book.metrics.constants import REFERENCE_PAGES_DIRNAME
from llb.prep.book.source import sha256_file


def _as_float(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    return value


class RefRegion(DataPrepRow):
    """An axis-aligned box in pixels of the rendered page, origin at the top left."""

    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float
    y1: float

    @model_validator(mode="before")
    @classmethod
    def _widen_ints(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        widened: dict[str, object] = {}
        for key, value in data.items():
            widened[str(key)] = _as_float(value) if key in {"x0", "y0", "x1", "y1"} else value
        return widened


class RefCell(DataPrepRow):
    """One table cell. A JSON string is the cell text with a span of one."""

    text: str
    colspan: int = Field(default=1, ge=1)
    rowspan: int = Field(default=1, ge=1)


class RefTable(DataPrepRow):
    """Row-major cell tree. Order is the tree the table metric edits."""

    rows: list[list[RefCell]] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _cells_from_strings(cls, data: object) -> object:
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            return data
        rows: list[object] = []
        for row in data["rows"]:
            if not isinstance(row, list):
                return data
            rows.append([{"text": cell} if isinstance(cell, str) else cell for cell in row])
        return {**data, "rows": rows}

    @model_validator(mode="after")
    def _rows_have_cells(self) -> "RefTable":
        if any(not row for row in self.rows):
            raise ValueError("a table row needs at least one cell")
        return self


class RefBlock(DataPrepRow):
    """One reference block. A figure points at the caption block that prints its caption."""

    block_id: str = Field(min_length=1)
    kind: BlockKind
    region: RefRegion
    reading_order: int = Field(ge=0)
    text: str = ""
    latex: str | None = None
    table: RefTable | None = None
    caption_id: str | None = None
    equation_number: str | None = None


class RefPage(DataPrepRow):
    """One PDF page of the reference transcription."""

    pdf_page: int = Field(ge=1)
    page_class: str = Field(min_length=1)
    printed_page: str | None = Field(min_length=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    blocks: list[RefBlock]

    @model_validator(mode="after")
    def _page_is_consistent(self) -> "RefPage":
        errors = page_errors(self)
        if errors:
            raise ValueError("; ".join(errors))
        return self


def page_errors(page: RefPage) -> list[str]:
    """Structural problems on one page. Empty when the page can be scored."""
    errors: list[str] = []
    _reject_duplicates(errors, "block ids repeat", [block.block_id for block in page.blocks])
    _reject_duplicates(
        errors, "reading orders repeat", [block.reading_order for block in page.blocks]
    )
    captions = {block.block_id for block in page.blocks if block.kind == "caption"}
    for block in page.blocks:
        errors.extend(_block_errors(page, block, captions))
    return errors


def _reject_duplicates(errors: list[str], message: str, values: list[object]) -> None:
    if len(values) != len(set(values)):
        errors.append(message)


def _block_errors(page: RefPage, block: RefBlock, captions: set[str]) -> list[str]:
    return _region_errors(page, block) + _kind_errors(block, captions)


def _region_errors(page: RefPage, block: RefBlock) -> list[str]:
    box = block.region
    errors: list[str] = []
    if box.x1 <= box.x0 or box.y1 <= box.y0:
        errors.append(f"{block.block_id} region has no area")
    if box.x1 > page.width or box.y1 > page.height:
        errors.append(f"{block.block_id} region extends outside the page")
    return errors


def _kind_errors(block: RefBlock, captions: set[str]) -> list[str]:
    errors: list[str] = []
    if block.kind == "formula" and not _has_latex(block):
        errors.append(f"{block.block_id} formula needs latex")
    if block.kind == "table" and block.table is None:
        errors.append(f"{block.block_id} table needs a structure")
    if block.kind == "figure" and block.caption_id is not None and block.caption_id not in captions:
        errors.append(f"{block.block_id} figure caption_id does not name a caption")
    if block.kind != "figure" and block.caption_id is not None:
        errors.append(f"{block.block_id} caption_id belongs on a figure")
    return errors


def _has_latex(block: RefBlock) -> bool:
    return bool(block.latex and block.latex.strip())


def load_reference(directory: Path) -> list[RefPage]:
    """Every page under ``directory/pages``, in PDF-page order. Refuses the first broken file."""
    pages_dir = directory / REFERENCE_PAGES_DIRNAME
    paths = sorted(pages_dir.glob("*.json")) if pages_dir.is_dir() else []
    if not paths:
        raise ValueError(f"{directory} has no {REFERENCE_PAGES_DIRNAME}/*.json reference pages")
    pages: list[RefPage] = []
    seen: dict[int, Path] = {}
    for path in paths:
        try:
            page = RefPage.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (ValidationError, json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        prior = seen.get(page.pdf_page)
        if prior is not None:
            raise ValueError(f"{path}: pdf page {page.pdf_page} is also {prior.name}")
        seen[page.pdf_page] = path
        if page.page_class.strip() != page.page_class or not page.page_class.strip():
            raise ValueError(f"{path}: page_class is empty")
        pages.append(page)
    return sorted(pages, key=lambda page: page.pdf_page)


def require_formulas(pages: list[RefPage]) -> None:
    """Refuse formulas the scoring normalizer cannot parse."""
    from llb.prep.book.metrics.latex import latex_parses

    errors = [
        f"pdf page {page.pdf_page} block {block.block_id} latex does not parse"
        for page in pages
        for block in page.blocks
        if block.kind == "formula" and not latex_parses(block.latex or "")
    ]
    if errors:
        raise ValueError("; ".join(errors))


def digest_pages(directory: Path) -> str:
    """Hash the ordered filenames and page file digests, including exact file bytes."""
    return digest_json({path.name: sha256_file(path) for path in sorted(directory.glob("*.json"))})


def verify_reference_digest(directory: Path, record: Path, source_sha256: str | None) -> None:
    """Bind a scoring run to the exact accepted reference and its source."""
    payload = PilotReferenceAcceptance.model_validate_json(record.read_bytes())
    if digest_pages(directory / REFERENCE_PAGES_DIRNAME) != payload.reference_sha256:
        raise ValueError(f"{directory} differs from the reference checksum in {record}")
    if source_sha256 is not None and source_sha256 != payload.source_sha256:
        raise ValueError(f"scoring run source checksum differs from {record}")
