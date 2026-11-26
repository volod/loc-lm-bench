"""Ink coverage: a confident block does not hide a column that produced no reading."""

from pathlib import Path

import numpy as np
import pytest

from llb.goldset.markers import marker_overlap
from llb.prep.book.config import read_config, write_config
from llb.prep.book.engines.protocol import EngineRead, Proposed, write_reading
from llb.prep.book.ledger.assemble import assemble
from llb.prep.book.review.build import build_review_queue
from llb.prep.book.review.coverage import account_page
from tests.llb.prep.book._review_factory import body, write_run

_WIDTH = 480
_HEIGHT = 640
_PAPER = 230
_INK = 30


def test_markers_refuse_an_omission_and_a_failed_page_and_keep_a_blank() -> None:
    omission = "{{llb-marker:omission:p0003-r0000:0-0}}"
    failed = "{{llb-marker:page-failed:p0005:0-0}}"
    blank = "{{llb-marker:blank:p0004:0-0}}"
    assert marker_overlap(omission, 0, len(omission)) == omission
    assert marker_overlap(failed, 0, len(failed)) == failed
    assert marker_overlap(blank, 0, len(blank)) is None


def test_layout_does_not_cover_a_column_the_reading_missed(tmp_path: Path) -> None:
    image = _columns()
    left, right = ((36.0, 80.0, 170.0, 600.0), (300.0, 80.0, 444.0, 600.0))
    page = tmp_path / "p0003"
    _store(page, "ocr", _read("ocr", (left,)))
    _store(page, "layout", _read("layout", (left, right)))
    ledger = assemble(
        "p0003",
        3,
        _WIDTH,
        _HEIGHT,
        (_read("ocr", (left,)), _read("layout", (left, right))),
    )
    account = account_page(image, ledger, page)
    assert account.coverage == "omission"
    assert account.regions
    assert all(region.region.x0 > left[2] for region in account.regions)
    assert any(block.region.x0 > left[2] for block in ledger.blocks)


def test_a_read_stretch_stays_out_of_the_unread_band(tmp_path: Path) -> None:
    image = np.full((_HEIGHT, _WIDTH), _PAPER, dtype=np.uint8)
    _lines(image, 40, 440, 80, 400)
    _lines(image, 40, 440, 500, 620)
    page = tmp_path / "p0001"
    bottom = (40.0, 500.0, 440.0, 620.0)
    _store(page, "ocr", _read("ocr", (bottom,)))
    ledger = assemble("p0001", 1, _WIDTH, _HEIGHT, (_read("ocr", (bottom,)),))
    account = account_page(image, ledger, page)
    assert account.coverage == "omission"
    assert account.regions
    assert all(region.region.y1 < bottom[1] for region in account.regions)


def test_blank_empty_and_covered_pages(tmp_path: Path) -> None:
    blank = np.full((_HEIGHT, _WIDTH), _PAPER, dtype=np.uint8)
    empty = assemble("p0002", 2, _WIDTH, _HEIGHT, ())
    assert account_page(blank, empty, None).coverage == "blank"
    inked = _one_column()
    page = tmp_path / "p0002"
    _store(page, "ocr", _read("ocr", ()))
    assert account_page(inked, empty, page).coverage == "omission"
    covered = assemble("p0001", 1, _WIDTH, _HEIGHT, (_read("ocr", ((40.0, 80.0, 560.0, 600.0),)),))
    assert account_page(inked, covered, None).coverage == "read"
    black = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    block = assemble("p0001", 1, _WIDTH, _HEIGHT, (_read("ocr", ((0.0, 0.0, 100.0, 40.0),)),))
    assert account_page(black, block, None).coverage == "read"


def test_an_unfinished_page_is_refused(tmp_path: Path) -> None:
    run = write_run(tmp_path, body("hello"))
    config = read_config(run)
    source = config.source.model_copy(update={"page_count": 2})
    write_config(
        run, config.model_copy(update={"pages": [1, 2], "pages_spec": "1-2", "source": source})
    )
    with pytest.raises(ValueError, match="has not finished"):
        build_review_queue(run)


def _columns() -> np.ndarray:
    page = np.full((_HEIGHT, _WIDTH), _PAPER, dtype=np.uint8)
    _lines(page, 36, 170)
    _lines(page, 300, 444)
    return page


def _one_column() -> np.ndarray:
    page = np.full((_HEIGHT, _WIDTH), _PAPER, dtype=np.uint8)
    _lines(page, 40, 560)
    return page


def _lines(page: np.ndarray, x0: int, x1: int, start: int = 80, stop: int | None = None) -> None:
    limit = page.shape[0] - 40 if stop is None else stop
    for top in range(start, limit, 36):
        page[top : top + 8, x0:x1] = _INK


def _read(stage: str, boxes: tuple[tuple[float, float, float, float], ...]) -> EngineRead:
    blocks = tuple(
        Proposed("body", index, box[0], box[1], box[2], box[3], "kept", 0.99)
        for index, box in enumerate(boxes)
    )
    return EngineRead("scripted", None, "original", stage, blocks)


def _store(page: Path, stage: str, reading: EngineRead) -> None:
    write_reading(page / stage, reading)
