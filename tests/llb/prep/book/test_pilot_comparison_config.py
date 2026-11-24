"""Projection, VRAM refusal, the roster, and the compare command."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llb.cli import app
from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.compare.assemble import compare_submissions
from llb.prep.book.compare.baseline import (
    docling_available,
    extract_baseline,
    recorded_language,
    write_baseline,
)
from llb.prep.book.compare.constants import (
    BASELINE_ENGINE,
    ROLE_PAGE,
    STATUS_OVER_VRAM,
)
from llb.prep.book.compare.project import projected_seconds
from llb.prep.book.compare.roster import PILOT_ROSTER, partition, run_roster
from llb.prep.book.config import RunRequest, load_request_file
from llb.prep.book.engines.candidate import with_ocr_language
from llb.prep.book.engines.fake import FakeEngine
from llb.prep.book.engines.protocol import EngineRequest, EngineUnavailable
from llb.prep.book.engines.registry import AdapterRegistry
from llb.prep.book.runner import check_request
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.classes import ClassesStage
from llb.prep.book.stages.preprocess import SignalsStage, VariantsStage
from llb.prep.book.stages.recognize import EngineStage, recognition_stages
from llb.prep.book.stages.render import RenderStage
from llb.prep.book.stages.routing import engine_inputs
from llb.prep.pdf.docling import ocr_language_pairs
from tests.llb.prep.book.compare_factory import (
    BASELINE,
    accepts_for,
    candidate,
    page,
    serves,
    submission,
)

_RUNNER = CliRunner()


def test_projection_counts_every_recognition_call_on_the_selection_set() -> None:
    counts = {"normal": 10}
    docling = {
        "layout": ("docling", {}),
        "ocr": ("docling", {}),
        "formula": ("docling", {}),
        "table": ("docling", {}),
    }
    seconds, missing = projected_seconds(counts, {"normal": "docling"}, docling, {"docling": 2.0})
    assert seconds == pytest.approx(80.0)
    assert missing == []
    split = {
        "layout": ("docling", {}),
        "ocr": ("docling", {"normal": "reader"}),
        "formula": ("docling", {}),
        "table": ("docling", {}),
    }
    seconds, missing = projected_seconds(
        counts, {"normal": "reader"}, split, {"docling": 2.0, "reader": 3.0}
    )
    assert seconds == pytest.approx(90.0)
    assert missing == []


def test_compare_labels_the_projection_selection_set() -> None:
    pages = [page("normal", 1, "abcd"), page("normal", 2, "abcd")]
    report, _config = compare_submissions(
        pages,
        pages,
        [
            submission(BASELINE, 1, "abxx", text_only=True),
            submission(BASELINE, 2, "abxx", text_only=True),
            submission("book-ocr/reader", 1, "abcd"),
            submission("book-ocr/reader", 2, "abcd"),
        ],
        [candidate("book-ocr/reader", "reader", seconds_per_page=3.0)],
        language="eng",
        accepts=accepts_for(pages),
        baseline_vram_mb=2970,
        baseline_seconds_per_page=2.0,
        class_page_counts={"normal": 10},
        serves=_ocr_only_reader,
    )
    assert report.projection_kind == "selection-set"
    assert report.projected_seconds == pytest.approx(90.0)
    assert report.classes[0].winner_lane == "book-ocr/reader"


def test_an_over_vram_lane_does_not_win() -> None:
    pages = [page("normal", 1, "abcd")]
    perfect = submission("book-ocr/huge", 1, "abcd")
    report, _config = compare_submissions(
        pages,
        pages,
        [submission(BASELINE, 1, "xxxx", text_only=True), perfect],
        [
            candidate(
                "book-ocr/huge",
                "huge",
                vram_mb=20000,
                status=STATUS_OVER_VRAM,
                detail="needs 20000 MB",
            )
        ],
        language="eng",
        accepts=accepts_for(pages),
        baseline_vram_mb=2970,
        class_page_counts={"normal": 1},
        serves=serves,
    )
    assert report.classes[0].winner_lane == BASELINE
    assert report.not_compared[0].lane == "book-ocr/huge"
    assert report.not_compared[0].reason == "over-vram"
    assert "book-ocr/huge" in report.won_no_class


def test_roster_skips_engines_the_host_cannot_hold() -> None:
    _runnable, skipped = partition(PILOT_ROSTER, 8151)
    refused = sorted(item.entry.engine for item in skipped if item.entry.role == ROLE_PAGE)
    assert refused == ["deepseek-ocr-2", "dots-ocr", "marker-balanced", "olmocr-2"]
    started: list[str] = []
    run_roster(PILOT_ROSTER, 8151, lambda entry: started.append(entry.lane))
    assert started == [entry.lane for entry in PILOT_ROSTER if entry.declared_vram_mb <= 8151]
    assert "paddleocr-vl" in {entry.engine for entry in PILOT_ROSTER if entry.lane in started}
    assert "mineru-standard" in {entry.engine for entry in PILOT_ROSTER if entry.lane in started}
    assert "docling-vlm-mineru2-pro" in {
        entry.engine for entry in PILOT_ROSTER if entry.lane in started
    }


def test_dry_run_refuses_an_engine_by_name(scan_pdf: Path) -> None:
    registry = _fake_registry()
    request = _vram_request(scan_pdf, {"fake": 20000})
    low = check_request(request, registry, host_vram_mb=8000)
    assert any(
        "fake" in error and "20000" in error and "EngineUnavailable" in error
        for error in low.errors
    )
    assert check_request(request, registry, host_vram_mb=20000).ok
    none = check_request(request, registry, host_vram_mb=None)
    assert any("fake" in error and "no CUDA device" in error for error in none.errors)
    assert any(
        "EngineUnavailable" in error
        for error in check_request(request, registry, host_vram_mb=19999).errors
    )
    boolean = _vram_request(scan_pdf, {"fake": True})
    refused = check_request(boolean, registry, host_vram_mb=8000)
    assert any("number" in error for error in refused.errors)
    assert not any("EngineUnavailable" in error for error in refused.errors)


def test_generated_config_dry_runs_against_the_named_budget(scan_pdf: Path, tmp_path: Path) -> None:
    pages = [page("normal", 1, "abcd")]
    _report, config = compare_submissions(
        pages,
        pages,
        [submission(BASELINE, 1, "abcd", text_only=True)],
        [],
        language="eng",
        accepts=accepts_for(pages),
        baseline_vram_mb=2970,
        class_page_counts={"normal": 1},
        serves=serves,
    )
    path = tmp_path / "book-ocr.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    loaded = load_request_file(path)
    request = RunRequest(source=scan_pdf, **loaded)
    registry = _scripted_registry()
    low = check_request(request, registry, host_vram_mb=1000)
    assert any(
        "docling" in error and "2970" in error and "EngineUnavailable" in error
        for error in low.errors
    )
    assert check_request(request, registry, host_vram_mb=4000).ok


def test_language_is_part_of_the_cache_key_and_the_docling_argv() -> None:
    assert ocr_language_pairs(None) == (["ukr", "eng"], ["uk", "en"])
    assert ocr_language_pairs(("eng",)) == (["eng"], ["en"])
    argv = ["{bin}/docling", "convert"]
    assert with_ocr_language("docling", argv, "eng")[-2:] == ["--ocr-lang", "eng"]
    assert with_ocr_language("mineru-standard", argv, "eng") == argv
    assert with_ocr_language("docling", argv, None) == argv
    stage = EngineStage("ocr", AdapterRegistry((FakeEngine(),)))
    with_language = stage.effective_params({"engine": "fake", "ocr_language": "eng"}, None, None, 1)
    without = stage.effective_params({"engine": "fake"}, None, None, 1)
    assert with_language["ocr_language"] == "eng"
    assert "ocr_language" not in without
    assert "vram_mb" not in with_language
    routed = stage.page_params(
        {
            "engine": "fake",
            "page_class": "severe_curvature",
            "variant_by_page_class": {"severe_curvature": "dewarp"},
        },
        {},
        None,
        None,
        1,
    )
    assert routed["variant"] == "dewarp"
    assert "classes" in engine_inputs(
        ["render", "classes"], {"variant_by_page_class": {"normal": "deskew"}}
    )
    errors = stage.validate_order({"engine": "fake", "variant": "deskew"}, ["render"])
    assert any("deskew" in error for error in errors)


def test_extract_baseline_names_docling_when_it_is_not_installed(tmp_path: Path) -> None:
    if docling_available():
        pytest.skip("docling is installed; the refusal is the missing-dependency path")
    with pytest.raises(EngineUnavailable, match="docling"):
        extract_baseline(tmp_path / "missing.pdf", [1], "eng", tmp_path / "out")


def test_a_sidecar_without_the_book_language_is_refused(tmp_path: Path) -> None:
    sidecar = write_baseline(tmp_path, [(1, "hello")], language="eng", source="scan.pdf")
    assert recorded_language(sidecar) == "eng"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["diagnostics"] = {"ocr_language": "ukr"}
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    pages = [page("normal", 1, "hello")]
    with pytest.raises(ValueError, match="ukr"):
        compare_submissions(
            pages,
            pages,
            [submission(BASELINE, 1, "hello", text_only=True)],
            [],
            language="eng",
            accepts=accepts_for(pages),
            baseline_vram_mb=2970,
            class_page_counts={},
            serves=serves,
            recorded_language=recorded_language(sidecar),
        )


def test_compare_command_refuses_an_unaccepted_reference_and_writes_an_ascii_report(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    _write_pages(root / "reference", [page("normal", 1, "hello")])
    _write_pages(root / "predraft", [page("normal", 1, "hello")])
    write_baseline(root, [(1, "hello")], language="eng", source="scan.pdf")
    missing = root / "missing-accepts.json"
    manifest = root / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "language": "eng",
                "reference": "reference",
                "predraft": "predraft",
                "accepts": missing.name,
                "citations": "baseline.citations.json",
                "class_counts": {"normal": 10},
            }
        ),
        encoding="utf-8",
    )
    refused = _RUNNER.invoke(
        app,
        [
            "book-ocr",
            "compare",
            str(manifest),
            "--out",
            str(root / "out.json"),
            "--config-out",
            str(root / "cfg.yaml"),
        ],
    )
    assert refused.exit_code == 2
    assert "cannot start" in refused.output
    (root / "accepts.json").write_text(
        accepts_for([page("normal", 1, "hello")]).model_dump_json(), encoding="utf-8"
    )
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(missing.name, "accepts.json"), encoding="utf-8"
    )
    out = root / "report.json"
    config = root / "book.yaml"
    accepted = _RUNNER.invoke(
        app,
        ["book-ocr", "compare", str(manifest), "--out", str(out), "--config-out", str(config)],
    )
    assert accepted.exit_code == 0, accepted.output
    assert all(ord(char) < 128 for char in accepted.output)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["negative"] is True
    assert report["language"] == "eng"
    assert load_request_file(config)["stages"]
    assert report["classes"][0]["configured_engine"] == BASELINE_ENGINE


def _ocr_only_reader(engine: str, stage: str) -> bool:
    """The reader wins OCR only, so the other three stages stay on Docling."""
    if engine == "docling":
        return stage in {"layout", "ocr", "formula", "table"}
    return engine == "reader" and stage == "ocr"


def _write_pages(directory: Path, pages) -> None:
    dest = directory / "pages"
    dest.mkdir(parents=True)
    for item in pages:
        (dest / f"p{item.pdf_page:04d}.json").write_text(item.model_dump_json(), encoding="utf-8")


def _vram_request(pdf: Path, budget: dict[str, object]) -> RunRequest:
    return RunRequest(
        source=pdf,
        pages="1",
        stages=(
            BookStageSpec(name="render"),
            BookStageSpec(name="ocr", params={"engine": "fake", "vram_mb": budget}),
        ),
    )


def _fake_registry() -> StageRegistry:
    return StageRegistry((RenderStage(), *recognition_stages(AdapterRegistry((FakeEngine(),)))))


class _Scripted:
    """A stand-in that does not import the real engine."""

    def __init__(self, engine_id: str) -> None:
        self.engine_id = engine_id
        self.stages = ("layout", "ocr", "formula", "table")

    def available(self) -> None:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, request: EngineRequest, stage: str):
        del request, stage
        raise AssertionError("dry run must not read a page")


def _scripted_registry() -> StageRegistry:
    adapters = AdapterRegistry((_Scripted("docling"), _Scripted("mineru-standard")))
    return StageRegistry(
        (
            RenderStage(),
            SignalsStage(),
            ClassesStage(),
            VariantsStage(),
            *recognition_stages(adapters),
        )
    )
