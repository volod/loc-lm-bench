"""Shared fixtures: a synthetic scan in its own directory, and a separate run root."""

from pathlib import Path

import pytest

from tests.llb.prep.book._scan_factory import make_scan


@pytest.fixture
def scan_pdf(tmp_path: Path) -> Path:
    source_dir = tmp_path / "library"
    source_dir.mkdir()
    return make_scan(source_dir / "scan.pdf")


@pytest.fixture
def out_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"
