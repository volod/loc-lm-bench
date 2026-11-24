"""Guards, point maps, and the rule that a clean page is not rewritten."""

from pathlib import Path

import numpy as np

from llb.core.contracts.data_prep.book import BookStageSpec
from llb.core.paths import resolve_data_dir
from llb.prep.book.config import RunRequest, stage_errors
from llb.prep.book.engines.fake import FakeEngine
from llb.prep.book.engines.protocol import EngineRead, Proposed
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.preprocess.constants import BOX_TOLERANCE_PX, DEWARP, MAX_LINE_BOW_PX
from llb.prep.book.preprocess.decide import selected
from llb.prep.book.preprocess.frames import reading_on_render
from llb.prep.book.preprocess.guards import refusal
from llb.prep.book.preprocess.images import read_grey
from llb.prep.book.preprocess.maps import map_box
from llb.prep.book.preprocess.measure import measure_page
from llb.prep.book.preprocess.records import read_manifest
from llb.prep.book.runner import resume_run, start_run
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.preprocess import SignalsStage, VariantsStage
from llb.prep.book.stages.recognize import recognition_stages
from llb.prep.book.stages.render import RenderStage
from tests.llb.prep.book._page_factory import (
    BACKGROUND,
    blob_box,
    bow,
    clean_page,
    image_pdf,
    tilt,
    upside_down,
    with_blob,
    with_marks,
)


def _locate(x: float, y: float) -> tuple[float, float]:
    return x, y


def _build(page: np.ndarray, out: Path, method: str = "opencv"):
    from llb.prep.book.preprocess.build import build_variants

    signals = measure_page(page, "p0001", 1)
    return signals, build_variants(page, signals, out, method=method, data_dir=resolve_data_dir())


def test_stroke_guard_refuses_an_erased_minus_and_decimal():
    page = with_marks(clean_page())
    erased = page.copy()
    erased[90:92, 200:202] = BACKGROUND
    erased[130, 180:200] = BACKGROUND
    reason = refusal(page, erased, _locate, geometric=False, covered=True)
    assert reason is not None and "thin stroke" in reason


def test_content_guard_refuses_a_dewarp_that_drops_an_edge_line():
    page = with_marks(clean_page())
    dropped = page.copy()
    dropped[2:4, :] = BACKGROUND
    reason = refusal(page, dropped, _locate, geometric=True, covered=True)
    assert reason is not None and "page-edge" in reason


def test_opencv_variants_are_byte_identical_and_leave_the_page_alone(tmp_path: Path):
    page = with_marks(bow(clean_page(), 8.0))
    before = page.tobytes()
    _, first = _build(page, tmp_path / "a")
    _, second = _build(page, tmp_path / "b")
    assert page.tobytes() == before
    assert (tmp_path / "a" / "manifest.json").read_bytes() == (
        tmp_path / "b" / "manifest.json"
    ).read_bytes()
    for record in first.variants:
        if record.image is None:
            continue
        left = (tmp_path / "a" / record.image).read_bytes()
        right = (tmp_path / "b" / record.image).read_bytes()
        assert left == right
    dewarp = next(record for record in first.variants if record.name == DEWARP)
    assert (
        dewarp.status == "kept" and dewarp.transform is not None and dewarp.transform.kind == "grid"
    )


def test_unbend_removes_the_planted_bow_and_leaves_the_margin():
    from llb.prep.book.preprocess.dewarp import unbend

    page = bow(clean_page(), 8.0)
    before = measure_page(page, "p0001", 1).line_curvature_px
    image, _map_x, map_y = unbend(page)
    shift = map_y - np.arange(page.shape[0], dtype=np.float32)[:, None]
    assert float(np.max(np.abs(shift))) <= MAX_LINE_BOW_PX
    assert float(np.max(np.abs(shift[:, 0]))) < 1.0
    assert float(np.max(np.abs(shift[:, -1]))) < 1.0
    assert measure_page(image, "p0001", 1).line_curvature_px < before


def test_an_upside_down_skewed_block_maps_back_within_tolerance(tmp_path: Path):
    upright = with_blob(clean_page(), 180, 198, 36, 16)
    scan = upside_down(tilt(upright, 2.0))
    signals, manifest = _build(scan, tmp_path)
    assert signals.orientation_deg == 180
    assert selected(signals)["rotated"] and selected(signals)["deskew"]
    rotated = next(record for record in manifest.variants if record.name == "rotated")
    assert rotated.status == "kept" and rotated.image is not None
    assert measure_page(read_grey(tmp_path / rotated.image), "p0001", 1).orientation_deg == 0
    deskew = next(record for record in manifest.variants if record.name == "deskew")
    assert deskew.status == "kept" and deskew.image is not None and deskew.transform is not None
    found = blob_box(read_grey(tmp_path / deskew.image))
    proposed = Proposed(
        "body", 0, float(found[0]), float(found[1]), float(found[2]), float(found[3]), "", None
    )
    height, width = scan.shape
    mapped = map_box(deskew.transform, proposed, width, height)
    assert mapped is not None
    reading = reading_on_render(
        EngineRead("fake", None, "deskew", "ocr", (proposed,)), tmp_path, width, height
    )
    original = blob_box(scan)
    for got, expected in (
        (mapped.x0, original[0]),
        (mapped.y0, original[1]),
        (mapped.x1, original[2]),
        (mapped.y1, original[3]),
    ):
        assert abs(got - expected) <= BOX_TOLERANCE_PX
    assert reading.blocks[0].x0 == mapped.x0 and reading.blocks[0].y0 == mapped.y0


def test_a_missing_variant_is_refused_instead_of_falling_back(tmp_path: Path):
    page = clean_page()
    _build(page, tmp_path)
    reading = EngineRead("fake", None, "dewarp", "ocr", ())
    try:
        reading_on_render(reading, tmp_path, page.shape[1], page.shape[0])
    except RuntimeError as exc:
        assert "dewarp" in str(exc)
    else:
        raise AssertionError("expected the missing variant to be refused")


def test_unknown_variant_needs_the_variants_stage_first():
    registry = StageRegistry(
        (
            RenderStage(),
            SignalsStage(),
            VariantsStage(),
            *recognition_stages(AdapterRegistry((FakeEngine(),))),
        )
    )
    early = (
        BookStageSpec(name="render"),
        BookStageSpec(name="ocr", params={"engine": "fake", "variant": "deskew"}),
    )
    assert any("needs the variants stage" in error for error in stage_errors(early, registry, None))
    unknown = (
        BookStageSpec(name="render"),
        BookStageSpec(name="signals"),
        BookStageSpec(name="variants"),
        BookStageSpec(name="ocr", params={"engine": "fake", "variant": "cleaned"}),
    )
    assert any("not a known variant" in error for error in stage_errors(unknown, registry, None))
    ordered = (
        BookStageSpec(name="render", params={"dpi": 72, "colorspace": "grey"}),
        BookStageSpec(name="signals"),
        BookStageSpec(name="variants"),
        BookStageSpec(name="ocr", params={"engine": "fake", "variant": "deskew"}),
    )
    assert stage_errors(ordered, registry, None) == []


def test_a_rerun_reuses_the_variants_stage_and_the_render_is_unchanged(tmp_path: Path):
    library = tmp_path / "library"
    library.mkdir()
    source = library / "scan.pdf"
    image_pdf(source, bow(clean_page(), 8.0))
    registry = StageRegistry((RenderStage(), SignalsStage(), VariantsStage()))
    stages = (
        BookStageSpec(name="render", params={"dpi": 72, "colorspace": "grey"}),
        BookStageSpec(name="signals"),
        BookStageSpec(name="variants", params={"dewarp": "uvdoc"}),
    )
    first = start_run(
        RunRequest(source=source, pages="1", stages=stages), tmp_path / "runs", registry
    )
    render = first.run_dir / "pages" / "p0001" / "render" / "page.png"
    render_bytes = render.read_bytes()
    manifest = read_manifest(first.run_dir / "pages" / "p0001" / "variants" / "manifest.json")
    dewarp = next(record for record in manifest.variants if record.name == DEWARP)
    assert dewarp.status == "refused" and dewarp.image is None
    assert first.outcomes.get("ran") == 3
    second = resume_run(first.run_dir, registry=registry)
    assert second.outcomes.get("ran", 0) == 0
    assert second.outcomes.get("cached") == 3
    assert render.read_bytes() == render_bytes
    published = first.run_dir / "pages" / "p0001" / "signals.json"
    assert published.is_file()
    assert (
        published.read_bytes()
        == (first.run_dir / "pages" / "p0001" / "signals" / "signals.json").read_bytes()
    )
