"""The generated books, command driver, and decoded-page measurement for publication tests."""

import json
import weakref
from pathlib import Path

import fitz
import numpy as np
import pytest
from typer.testing import CliRunner

from llb.cli import app
from llb.prep.book import runner
from llb.prep.book.preprocess import images
from tests.llb.prep.book._stages import SimulatedKill
from tests.llb.prep.book.synthetic_book import make_registry, stage_specs
from tests.llb.prep.book.synthetic_pages import _folio, _line

_LONG = 64
_REPEATED_FOLIO_PAGE = 12
_CAP = "1000"
_END = "Pyrochlore marks the verified calibration target on page 18."
_RUNNER = CliRunner()


def _command(*args: str | Path, code: int = 0) -> str:
    result = _RUNNER.invoke(app, [str(arg) for arg in args])
    assert result.exit_code == code, (result.output, result.exception)
    return result.output


def _book(*args: str | Path, code: int = 0) -> str:
    return _command("book-ocr", *args, code=code)


def _run(scan: Path, root: Path, monkeypatch: pytest.MonkeyPatch, *, kill: bool = False) -> Path:
    with fitz.open(scan) as pdf:
        pages = pdf.page_count
    extra = [
        line
        for page in range(_REPEATED_FOLIO_PAGE, pages + 1)
        for line in (
            _line(page, 0.30, _END if page == pages else f"Calibration record {page} is filed."),
            _folio(page, 18 if page == _REPEATED_FOLIO_PAGE else page + 7),
        )
    ]
    if pages == _LONG:
        extra.append(_line(pages, 0.15, "Calibration record " + "recorded " * 107))
    registry = make_registry(kill=kill, extra=extra)
    monkeypatch.setattr(runner, "default_registry", lambda: registry)
    config = root.with_suffix(".json")
    config.write_text(json.dumps({"stages": [spec.model_dump() for spec in stage_specs()]}))
    args = [
        "book-ocr",
        "run",
        str(scan),
        "--out-dir",
        str(root),
        "--config",
        str(config),
        "--max-attempts",
        "1",
    ]
    if kill:
        with pytest.raises(SimulatedKill):
            _RUNNER.invoke(app, args)
    else:
        result = _RUNNER.invoke(app, args)
        assert result.exit_code == 0, result.output
    (run,) = root.iterdir()
    if kill:
        staged = run / "book/.vault.partial/metadata"
        staged.mkdir(parents=True)
        (staged / "render.json").write_text("unfinished")
        records = {
            path: path.read_bytes()
            for path in (run / "pages").glob("*/*/stage.json")
            if json.loads(path.read_bytes())["status"] == "done"
        }
        assert len(records) > pages
        assert "cached" in _book("run", "--resume", run)
        assert all(path.read_bytes() == before for path, before in records.items())
    _settle(run)
    return run


def _settle(run: Path) -> None:
    _book("review-queue", run)
    _book("structure", run)


def _render(run: Path, vault: Path) -> str:
    return _book(
        "render",
        run,
        "--output",
        vault,
        "--page-fallback",
        "--source-policy",
        "copy",
        "--section-cap",
        _CAP,
    )


def _project(run: Path, out: Path) -> str:
    return _book(
        "project",
        run,
        "--out",
        out,
        "--licence",
        "redistributable",
        "--language",
        "en",
        "--section-cap",
        _CAP,
    )


class _Decoded:
    def __init__(self) -> None:
        self.count = self.live = self.peak = 0

    def read(self, path: Path, mode: int) -> np.ndarray:
        array = _DECODE(path, mode)
        self.count += 1
        self.live += 1
        self.peak = max(self.live, self.peak)
        weakref.finalize(array, self._release)
        return array

    def _release(self) -> None:
        self.live -= 1


_DECODE = images._read


def _measured_render(run: Path, vault: Path, monkeypatch: pytest.MonkeyPatch) -> _Decoded:
    meter = _Decoded()
    with monkeypatch.context() as patch:
        patch.setattr(images, "_read", meter.read)
        _render(run, vault)
    assert meter.live == 0
    return meter
