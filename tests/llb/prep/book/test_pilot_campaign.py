"""The pilot campaign: which lanes start, and that a failed page is named."""

from pathlib import Path

import pytest

from llb.core.contracts.data_prep.book import (
    BookPageManifest,
    PageInventoryRecord,
    PageLedgerRecord,
    PageStageState,
)
from llb.prep.book.classes.records import read_selection
from llb.prep.book.compare.campaign import execute_campaign, prepare_campaign
from llb.prep.book.compare.campaign_host import CampaignHooks, LaneRun, lane_timing
from llb.prep.book.compare.campaign_record import comparison_payload
from llb.prep.book.compare.constants import ROLE_PAGE
from llb.prep.book.compare.lane_config import lane_request, require_source
from llb.prep.book.compare.manifest import ComparisonManifest
from llb.prep.book.compare.roster import PILOT_ROSTER, RosterEntry
from llb.prep.book.runner import start_run
from llb.prep.book.stages import StageRegistry
from tests.llb.prep.book._stages import CountingRender

_SHA = "a" * 64


def test_a_page_reader_runs_every_recognition_stage_it_serves(scan_pdf: Path) -> None:
    paddle = next(item for item in PILOT_ROSTER if item.engine == "paddleocr-vl")
    request = lane_request(scan_pdf, [4, 18], paddle, "eng", None)
    names = [spec.name for spec in request.stages]
    assert names[:3] == ["render", "signals", "classes"]
    assert names[-2:] == ["blocks", "resolve"]
    layout = next(spec for spec in request.stages if spec.name == "layout")
    assert layout.params["engine"] == "paddleocr-vl"
    assert layout.params["ocr_language"] == "eng"
    assert layout.params["vram_mb"] == {"paddleocr-vl": paddle.declared_vram_mb}


def test_a_variant_policy_reads_rapidocr_on_that_variant(scan_pdf: Path) -> None:
    dewarp = next(item for item in PILOT_ROSTER if item.lane.endswith("@dewarp-page"))
    variant = lane_request(scan_pdf, [4], dewarp, "eng", None)
    names = {spec.name for spec in variant.stages}
    assert "layout" not in names
    assert next(spec for spec in variant.stages if spec.name == "variants").params == {
        "dewarp": "page-dewarp"
    }
    ocr = next(spec for spec in variant.stages if spec.name == "ocr")
    assert ocr.params["engine"] == "rapidocr"
    assert ocr.params["variant"] == "dewarp"
    assert ocr.params["vram_mb"] == {"rapidocr": 0}


def test_a_pilot_from_another_scan_is_refused(scan_pdf: Path) -> None:
    selection = read_selection(
        Path("samples/book_ocr/space-mission-engineering-the-new-smad/pilot_pages.json")
    )
    with pytest.raises(ValueError, match="sha256"):
        require_source(scan_pdf, selection)


def test_a_failed_page_stays_measured_and_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llb.prep.book.compare.campaign.extract_baseline",
        lambda *_args, **_kwargs: tmp_path / "baseline.citations.json",
    )
    record = prepare_campaign(
        pdf=tmp_path / "scan.pdf",
        pilot=tmp_path / "pilot.json",
        digest=_SHA,
        pages=[4, 18],
        language="eng",
        reuse_from=None,
        host_mb=1000,
        class_counts={"normal": 2},
        entries=(
            RosterEntry("book-ocr/huge", "huge", ROLE_PAGE, 20000),
            RosterEntry("book-ocr/gone", "gone", ROLE_PAGE, 100),
            RosterEntry("book-ocr/fake", "fake", ROLE_PAGE, 100),
        ),
        problem=lambda engine: "not installed" if engine == "gone" else None,
    )
    assert [lane.status for lane in record.lanes] == ["over-vram", "unavailable", "pending"]
    started: list[str] = []

    def start(request, _out_root, run_dir, _run_id):
        started.append(str(request.stages[3].params["engine"]))
        return LaneRun(run_dir, 1.25, [4], ["variant deskew unavailable"])

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    finished = execute_campaign(
        record,
        bundle,
        tmp_path / "runs",
        CampaignHooks(start, lambda: None, lambda _mb: None),
    )
    assert started == ["fake"]
    assert finished.baseline.status == "measured"
    assert finished.lanes[2].status == "measured"
    assert finished.lanes[2].failed_pages == [4]
    assert "4" in finished.lanes[2].detail
    saved = (bundle / "campaign.json").read_text(encoding="utf-8")
    assert "book-ocr/fake" in saved
    manifest = ComparisonManifest.model_validate(
        comparison_payload(
            finished,
            reference="reference",
            predraft="predraft",
            accepts="accepts.json",
        )
    )
    assert manifest.language == "eng"
    assert manifest.lanes[2].status == "measured"


def test_lane_timing_averages_one_call_and_names_a_missing_ledger(tmp_path: Path) -> None:
    run = tmp_path / "run"
    page = run / "pages" / "p0001"
    page.mkdir(parents=True)
    (page / "blocks.json").write_text("{}", encoding="utf-8")
    manifest = BookPageManifest(
        schema_id="llb.book-ocr-page-manifest",
        schema_version="1.0.0",
        kind="book-ocr-page-manifest",
        run_id="run",
        source_sha256=_SHA,
        source_page_count=1,
        n_pages=2,
        stage_names=["ocr", "layout", "blocks"],
        pages=[
            _record(1, "done"),
            _record(2, "failed"),
        ],
    )
    seconds, failed, errors = lane_timing(run, manifest)
    assert seconds == pytest.approx(3.0)
    assert failed == [2]
    assert errors


def test_start_run_uses_the_directory_it_is_given(scan_pdf: Path, tmp_path: Path) -> None:
    dest = tmp_path / "runs" / "named-runid"
    outcome = start_run(
        _render_request(scan_pdf),
        tmp_path / "runs",
        StageRegistry((CountingRender(),)),
        run_dir=dest,
        run_id="runid",
    )
    assert outcome.run_dir == dest
    assert (dest / "config.json").is_file()


def _render_request(scan_pdf: Path):
    from llb.core.contracts.data_prep.book import BookStageSpec
    from llb.prep.book.config import RunRequest

    return RunRequest(source=scan_pdf, pages="1", stages=(BookStageSpec(name="render"),))


def _record(pdf_page: int, status: str) -> PageLedgerRecord:
    stages = [
        PageStageState(stage="ocr", status=status, attempts=1, duration_s=2.0, error="no"),
        PageStageState(stage="layout", status=status, attempts=1, duration_s=4.0),
        PageStageState(stage="blocks", status="done", attempts=1, duration_s=0.1),
    ]
    if status == "done":
        stages[0] = PageStageState(stage="ocr", status="done", attempts=1, duration_s=2.0)
    return PageLedgerRecord(
        page_id=f"p{pdf_page:04d}",
        pdf_page=pdf_page,
        batch=0,
        inventory=PageInventoryRecord(
            width_pt=100.0,
            height_pt=120.0,
            rotation=0,
            orientation="portrait",
            image_count=1,
            embedded_text_chars=0,
        ),
        stages=stages,
    )
