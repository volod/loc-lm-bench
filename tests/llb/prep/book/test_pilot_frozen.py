"""A frozen pilot checks the actual source and the published classification metadata."""

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.cli import app
from llb.core.contracts.data_prep.book import BookStageSpec
from llb.prep.book.classes.records import (
    read_classes,
    read_selection,
    write_classes,
    write_selection,
)
from llb.prep.book.config import RunRequest
from llb.prep.book.runner import start_run
from llb.prep.book.stages import StageRegistry
from llb.prep.book.stages.classes import ClassesStage
from llb.prep.book.stages.preprocess import SignalsStage
from llb.prep.book.stages.render import RenderStage

_RUNNER = CliRunner()


@pytest.fixture
def classified_run(scan_pdf: Path, out_root: Path) -> Path:
    stages = tuple(BookStageSpec(name=name) for name in ("render", "signals", "classes"))
    registry = StageRegistry((RenderStage(), SignalsStage(), ClassesStage()))
    return start_run(
        RunRequest(source=scan_pdf, pages="1-4", stages=stages), out_root, registry
    ).run_dir


def test_pilot_freeze_checks_source_coverage_and_signal_drift(
    classified_run: Path, scan_pdf: Path, tmp_path: Path
):
    frozen_path = tmp_path / "frozen.json"
    args = ["book-ocr", "pilot-select", str(classified_run), "--size", "4"]
    result = _RUNNER.invoke(app, [*args, "--out", str(frozen_path)])
    assert result.exit_code == 0, result.output
    frozen_bytes = frozen_path.read_bytes()
    frozen = read_selection(frozen_path)
    assert frozen.source_sha256 == hashlib.sha256(scan_pdf.read_bytes()).hexdigest()
    result = _RUNNER.invoke(app, [*args, "--frozen", str(frozen_path)])
    assert result.exit_code == 0, result.output
    result = _RUNNER.invoke(app, [*args[:3], "--size", "1", "--frozen", str(frozen_path)])
    assert result.exit_code == 2 and "coverage" in result.output

    foreign = tmp_path / "foreign.json"
    write_selection(foreign, frozen.model_copy(update={"source_sha256": "b" * 64}))
    result = _RUNNER.invoke(app, [*args, "--frozen", str(foreign)])
    assert result.exit_code == 2 and "source_sha256" in result.output

    published = classified_run / "pages" / frozen.pages[0].page_id / "classes.json"
    record = read_classes(published)
    changed = record.signals.model_copy(update={"contrast": 0.999})
    write_classes(published, record.model_copy(update={"signals": changed}))
    out = tmp_path / "failed.json"
    result = _RUNNER.invoke(app, [*args, "--frozen", str(frozen_path), "--out", str(out)])
    assert result.exit_code == 2 and "frozen pilot differs in pages" in result.output
    assert not out.exists()
    assert (classified_run / "pilot/selection.json").read_bytes() == frozen_bytes


def test_pilot_refuses_a_class_record_from_another_pdf_page(classified_run: Path):
    published = classified_run / "pages/p0001/classes.json"
    write_classes(published, read_classes(classified_run / "pages/p0002/classes.json"))
    result = _RUNNER.invoke(app, ["book-ocr", "pilot-select", str(classified_run)])
    assert result.exit_code == 2 and "record for p0001 (PDF page 1) names p0002" in result.output


def test_pilot_refuses_a_modified_source(classified_run: Path, scan_pdf: Path):
    with scan_pdf.open("ab") as stream:
        stream.write(b"\n% appended after classification\n")
    result = _RUNNER.invoke(app, ["book-ocr", "pilot-select", str(classified_run)])
    assert result.exit_code == 2 and "refusing to mix pages of two documents" in result.output
    assert not (classified_run / "pilot/selection.json").exists()
