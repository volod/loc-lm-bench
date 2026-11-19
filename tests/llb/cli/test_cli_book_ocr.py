"""`llb book-ocr run` / `status` through the Typer app."""

import json

from typer.testing import CliRunner

from llb.cli import app
from tests.llb.prep.book._scan_factory import make_scan

RUNNER = CliRunner()


def _scan(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    return make_scan(library / "scan.pdf")


def test_dry_run_validates_and_writes_nothing(tmp_path):
    scan, out = _scan(tmp_path), tmp_path / "runs"
    ok = RUNNER.invoke(
        app, ["book-ocr", "run", str(scan), "--dry-run", "--pages", "2-", "--out-dir", str(out)]
    )
    assert ok.exit_code == 0, ok.output
    assert "5 page(s) selected in 1 batch(es)" in ok.output and "dry run ok" in ok.output
    refused = RUNNER.invoke(app, ["book-ocr", "run", str(scan), "--dry-run", "--pages", "1-99"])
    assert refused.exit_code == 2
    assert "past the last page 6" in refused.output
    assert not out.exists()


def test_run_then_status_then_resume(tmp_path):
    scan, out = _scan(tmp_path), tmp_path / "runs"
    config = tmp_path / "book.yaml"
    config.write_text(
        "batch_pages: 3\nstages:\n  - render\n  - name: fake-ocr\n    params: {fail_pages: [2]}\n",
        encoding="utf-8",
    )
    result = RUNNER.invoke(
        app, ["book-ocr", "run", str(scan), "--config", str(config), "--out-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "fake-ocr: blocked 0, done 5, failed 1" in result.output
    (run,) = list(out.iterdir())

    status = RUNNER.invoke(app, ["book-ocr", "status", str(run), "--show", "open"])
    assert status.exit_code == 0, status.output
    assert (
        "p0002 pdf-page=2 batch=0 render=done fake-ocr=failed(2: ForcedStageFailure"
        in status.output
    )
    assert "p0001" not in status.output
    manifest = json.loads(RUNNER.invoke(app, ["book-ocr", "status", str(run), "--json"]).output)
    assert manifest["schema_id"] == "llb.book-ocr-page-manifest" and manifest["n_pages"] == 6

    resumed = RUNNER.invoke(app, ["book-ocr", "run", "--resume", str(run)])
    assert resumed.exit_code == 0, resumed.output
    assert "cached 11" in resumed.output
    shaped = RUNNER.invoke(app, ["book-ocr", "run", "--resume", str(run), "--pages", "1"])
    assert shaped.exit_code == 2 and "drop ['pages']" in shaped.output


def test_status_of_a_directory_that_is_not_a_run(tmp_path):
    result = RUNNER.invoke(app, ["book-ocr", "status", str(tmp_path)])
    assert result.exit_code == 2 and "is not a book-ocr run" in result.output
