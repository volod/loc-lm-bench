"""A fresh run: inventory, deterministic layout, contracts, source policy, and dry run."""

import hashlib
import json
from pathlib import Path

import pytest

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.config import RunRequest
from llb.prep.book.runner import BookRunError, check_request, start_run
from llb.prep.book.source import SourceError
from tests.llb.prep.book._stages import registry_with

STAGES = (BookStageSpec(name="render"), BookStageSpec(name="fake-ocr"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(scan: Path, **overrides) -> RunRequest:
    return RunRequest(
        **{"source": scan, "pages": "1-", "batch_pages": 4, "stages": STAGES, **overrides}
    )


def test_run_inventories_renders_and_names_deterministically(scan_pdf, out_root):
    outcome = start_run(_request(scan_pdf), out_root, registry_with())
    run = outcome.run_dir
    assert sorted(p.name for p in (run / "pages").iterdir()) == [f"p000{i}" for i in range(1, 7)]
    assert sorted(p.name for p in (run / "pages" / "p0001").iterdir()) == ["fake-ocr", "render"]
    pages = {page.pdf_page: page for page in outcome.manifest.pages}
    assert [page.batch for page in outcome.manifest.pages] == [0, 0, 0, 0, 1, 1]
    assert pages[1].inventory.native_dpi == 150
    assert pages[3].inventory.orientation == "landscape"
    rotated = pages[4].inventory
    assert (rotated.orientation, rotated.image_width_px, rotated.native_dpi) == (
        "portrait",
        750,
        150,
    )
    assert rotated.image_rotation in (90, 270) and pages[1].inventory.image_rotation == 0
    assert pages[5].inventory.embedded_text_chars > 0
    assert pages[1].inventory.embedded_text_chars == 0
    render = json.loads((run / "pages" / "p0002" / "render" / "stage.json").read_text())
    assert render["meta"] == {"dpi": 150, "colorspace": "rgb"}
    assert (
        json.loads((run / "pages" / "p0001" / "render" / "stage.json").read_text())["meta"][
            "colorspace"
        ]
        == "grey"
    )
    assert pages[1].image_sha256 == _digest(run / "pages" / "p0001" / "render" / "page.png")
    assert outcome.manifest.summary["fake-ocr"]["done"] == 6


def test_every_durable_file_reads_through_the_registry(scan_pdf, out_root):
    run = start_run(_request(scan_pdf), out_root, registry_with()).run_dir
    paths = [
        run / "config.json",
        run / "metadata" / "page_manifest.json",
        *sorted((run / "pages").glob("*/*/stage.json")),
    ]
    for path in paths:
        DEFAULT_REGISTRY.read_current(json.loads(path.read_text()), source=str(path))
    log_lines = (run / "logs" / "run.jsonl").read_text().splitlines()
    events = [json.loads(line)["event"] for line in log_lines]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert "processing_report.md" in {p.name for p in (run / "metadata").iterdir()}


def test_rerun_renders_byte_identical_pages(scan_pdf, out_root):
    first = start_run(_request(scan_pdf), out_root, registry_with()).run_dir
    second = start_run(_request(scan_pdf), out_root, registry_with()).run_dir
    for name in ("render/page.png", "fake-ocr/text.txt"):
        assert _digest(first / "pages" / "p0004" / name) == _digest(
            second / "pages" / "p0004" / name
        )


@pytest.mark.parametrize("policy", ["copy", "symlink", "reference"])
def test_source_is_byte_identical_after_a_run_under_every_policy(scan_pdf, out_root, policy):
    before = _digest(scan_pdf)
    outcome = start_run(_request(scan_pdf, source_policy=policy), out_root, registry_with())
    assert _digest(scan_pdf) == before
    stored = outcome.run_dir / "source" / "scan.pdf"
    if policy == "reference":
        assert not stored.exists()
    else:
        assert _digest(stored) == before
        assert stored.is_symlink() == (policy == "symlink")


def test_run_directory_that_would_hold_the_source_is_refused(scan_pdf):
    for out_root in (scan_pdf.parent, scan_pdf.parent / "runs"):
        with pytest.raises(SourceError, match="the directory holding the source"):
            start_run(_request(scan_pdf), out_root, registry_with())
    assert sorted(p.name for p in scan_pdf.parent.iterdir()) == ["scan.pdf"]


def test_dry_run_reports_every_problem_and_writes_nothing(scan_pdf, out_root):
    bad = _request(
        scan_pdf,
        pages="1-3,3",
        stages=(
            BookStageSpec(name="fake-ocr", params={"bogus": 1}),
            BookStageSpec(name="nope"),
            BookStageSpec(name="render", model="ollama:absent:1b"),
        ),
    )
    plan = check_request(bad, registry_with(), ollama_lister=lambda host: ["qwen2.5vl:7b"])
    text = "\n".join(plan.errors)
    assert "reads ['render'], which must be configured before it" in text
    assert "unknown parameter 'bogus'" in text
    assert "unknown stage 'nope'" in text
    assert "'ollama:absent:1b' is not served" in text
    assert "'1-3' and '3' overlap on page 3" in text
    assert not out_root.exists()
    with pytest.raises(BookRunError):
        start_run(bad, out_root, registry_with(), ollama_lister=lambda host: [])
    assert not out_root.exists()


def test_dry_run_accepts_a_served_model(scan_pdf):
    good = _request(
        scan_pdf,
        stages=(
            BookStageSpec(name="render"),
            BookStageSpec(name="fake-ocr", model="ollama:qwen2.5vl:7b"),
        ),
    )
    plan = check_request(good, registry_with(), ollama_lister=lambda host: ["qwen2.5vl:7b"])
    assert plan.ok and plan.page_count == 6 and len(plan.batches) == 2
