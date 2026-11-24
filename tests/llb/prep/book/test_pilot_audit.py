"""Independent scan defects exercised at the input, crop, and scoring boundaries."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.config import RunRequest
from llb.prep.book.engines.fake import FakeEngine
from llb.prep.book.engines.protocol import EngineRead, EngineRequest, Proposed
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.ledger.io import ledger_path, read_ledger
from llb.prep.book.metrics.geometry import Box
from llb.prep.book.metrics.page_score import score_page
from llb.prep.book.metrics.reference import RefBlock, RefPage, RefRegion
from llb.prep.book.metrics.submission import HypBlock, PageSubmission
from llb.prep.book.preprocess.guards import refusal
from llb.prep.book.preprocess.images import read_grey
from llb.prep.book.preprocess.maps import map_box
from llb.prep.book.preprocess.records import grid_record
from llb.prep.book.review.crops import write_block_crop
from llb.prep.book.runner import start_run
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.preprocess import SignalsStage, VariantsStage
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage
from tests.llb.prep.book._page_factory import (
    BACKGROUND,
    INK,
    bleed_page,
    blob_box,
    clean_page,
    image_pdf,
    upside_down,
    with_blob,
)


class _CornerReader(FakeEngine):
    def read(self, request: EngineRequest, stage: str) -> EngineRead:
        assert request.variant == "rotated"
        x0, y0, x1, y1 = blob_box(read_grey(request.image))
        assert (x0, y0, x1, y1) == (0, 0, 11, 13)
        block = Proposed("body", 0, x0, y0, x1, y1, "MARK", None)
        return EngineRead(self.engine_id, None, request.variant, stage, (block,))


def test_rotated_corner_mark_reaches_an_exact_original_crop(tmp_path: Path):
    library = tmp_path / "library"
    library.mkdir()
    scan = upside_down(with_blob(clean_page(), 0, 0, width=11, height=13))
    source = image_pdf(library / "corner.pdf", scan)
    source_bytes = source.read_bytes()
    registry = StageRegistry(
        (
            RenderStage(),
            SignalsStage(),
            VariantsStage(),
            *recognition_stages(AdapterRegistry((_CornerReader(),))),
        )
    )
    stages = (
        BookStageSpec(name="render"),
        BookStageSpec(name="signals"),
        BookStageSpec(name="variants"),
        BookStageSpec(name="ocr", params={"engine": "fake", "variant": "rotated"}),
        BookStageSpec(name="blocks"),
    )
    result = start_run(
        RunRequest(source=source, pages="1", stages=stages), tmp_path / "runs", registry
    )
    assert result.outcomes == {"ran": 5}
    page = read_ledger(ledger_path(result.run_dir, "p0001"))
    block = page.blocks[0]
    region = block.region
    height, width = scan.shape
    assert (region.x0, region.y0, region.x1, region.y1) == (width - 11, height - 13, width, height)
    original = read_grey(result.run_dir / "pages/p0001/render/page.png")
    np.testing.assert_array_equal(scan, original)
    crop = tmp_path / "crop.png"
    assert write_block_crop(original, region, crop, padding=0)
    np.testing.assert_array_equal(read_grey(crop), np.zeros((13, 11), dtype=np.uint8))
    assert source.read_bytes() == source_bytes


def test_grid_box_contains_the_bow_between_its_corners():
    y, x = np.mgrid[:33, :33]
    y = y + 8 * (16 - np.abs(x - 16)) / 16
    transform = grid_record(x, y, src_width=33, src_height=41)
    proposed = Proposed("body", 0, 0, 8, 33, 12, "LINE", None)
    mapped = map_box(transform, proposed, 33, 41)
    assert mapped is not None
    assert (mapped.x0, mapped.x1, mapped.y1) == (0, 33, 20)


def _identity(x: float, y: float) -> tuple[float, float]:
    return x, y


@pytest.mark.parametrize("mark", [(slice(90, 92), slice(200, 202)), (130, slice(180, 200))])
def test_erasing_either_faint_sign_or_point_refuses_candidate(mark: tuple[slice | int, slice]):
    original = clean_page()
    original[90:92, 200:202] = 185
    original[130, 180:200] = 185
    # Nearby dark text must not count as surviving ink at the decimal point.
    original[88:96, 203:210] = INK
    assert refusal(original, original, _identity, geometric=False, covered=True) is None
    damaged = original.copy()
    damaged[mark] = BACKGROUND
    assert refusal(original, damaged, _identity, geometric=False, covered=True) == (
        "erased 1 of 2 thin strokes"
    )


def test_clipped_gutter_ink_is_checked_with_neighbour_ink_still_present():
    original = bleed_page()
    original[25:32, :9] = INK
    damaged = original.copy()
    damaged[25:32, :9] = BACKGROUND
    assert refusal(original, original, _identity, geometric=True, covered=True) is None
    reason = refusal(original, damaged, _identity, geometric=True, covered=True)
    assert reason == "lost 1 text line(s) or page-edge line(s)"


def _two_columns() -> tuple[RefPage, PageSubmission]:
    blocks = [
        RefBlock(
            block_id=name,
            kind="body",
            region=RefRegion(x0=x, y0=10, x1=x + 30, y1=60),
            reading_order=order,
            text=f"Bay {label} -3.2",
        )
        for order, (name, x, label) in enumerate((("left", 10, "5"), ("right", 60, "S")))
    ]
    page = RefPage(
        pdf_page=1,
        page_class="multi_column",
        printed_page=None,
        width=100,
        height=80,
        blocks=blocks,
    )
    hypothesis = tuple(
        HypBlock(
            block.block_id,
            block.kind,
            Box(block.region.x0, block.region.y0, block.region.x1, block.region.y1),
            block.reading_order,
            block.text,
            None,
            None,
            None,
            None,
        )
        for block in blocks
    )
    return page, PageSubmission("book-ocr/fake", 1, hypothesis, False)


def test_column_swap_loses_reading_order_without_a_false_digit_substitution():
    page, clean = _two_columns()
    left, right = clean.blocks
    swapped = replace(
        clean, blocks=(replace(left, reading_order=1), replace(right, reading_order=0))
    )
    control = score_page(page, clean)
    assert control.reading_order_agreement.value == 1 and control.character_error_rate.value == 0
    result = score_page(page, swapped)
    assert result.reading_order_agreement.value == 0
    assert result.character_error_rate.value == pytest.approx(2 / 21)
    assert result.missing_words == result.duplicated_words == 0
    assert sum(result.dangerous_substitutions.values()) == 0


@pytest.mark.parametrize(
    ("changed", "confusion"),
    [
        ("Bay S -3.2", "5/S"),
        ("Bay 6 -3.2", None),
        ("Bay 5 3.2", "minus/hyphen/dash"),
        ("Bay 5 -32", "decimal/comma"),
    ],
)
def test_independent_glyph_losses_keep_order_and_hit_only_their_own_measure(
    changed: str, confusion: str | None
):
    page, clean = _two_columns()
    left, right = clean.blocks
    result = score_page(page, replace(clean, blocks=(replace(left, text=changed), right)))
    assert result.reading_order_agreement.value == 1
    assert result.character_error_rate.value == pytest.approx(1 / 21)
    expected = {name: int(name == confusion) for name in result.dangerous_substitutions}
    assert result.dangerous_substitutions == expected


def test_neighbour_bleed_contributes_no_body_text():
    page, clean = _two_columns()
    bleed = HypBlock(
        "bleed", "neighbour-page-bleed", Box(0, 0, 8, 70), 2, "wrong page", None, None, None, None
    )
    result = score_page(page, replace(clean, blocks=(*clean.blocks, bleed)))
    assert result.character_error_rate.value == 0
    assert result.missing_words == result.duplicated_words == 0
