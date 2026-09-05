"""The toolchain pin is one table; a restated minor in mypy/ruff/basedpyright is a silent fork.

`make venv PYTHON_VERSION=` can change the interpreter without rewriting pyproject.toml, which is
why basedpyright must not carry pythonVersion -- a matching copy would still be a lie after that
override. The shipped-tree assertion is the point of the tool; the synthetic cases pin each
restatement the join is responsible for.
"""

import tomllib
from pathlib import Path

from llb.build import toolchain
from llb.core.paths import PROJECT_ROOT

_PINNED = """
[project]
requires-python = ">=3.12"

[tool.llb.toolchain]
python-version = "3.13"
venv = ".venv"

[tool.mypy]
python_version = "3.13"

[tool.ruff]
target-version = "py313"

[tool.basedpyright]
venvPath = "."
venv = ".venv"
extraPaths = ["src"]
"""


def _write_pyproject(root: Path, text: str) -> Path:
    path = root / toolchain.PYPROJECT_FILENAME
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_shipped_tree_matches_the_pin():
    assert toolchain.integrity_findings(PROJECT_ROOT) == []


def test_make_and_cli_read_the_same_pin():
    pin = toolchain.load_toolchain(PROJECT_ROOT)
    assert pin.python_version == "3.13"
    assert pin.venv == ".venv"
    assert pin.ruff_target == "py313"
    pyproject = tomllib.loads((PROJECT_ROOT / toolchain.PYPROJECT_FILENAME).read_text())
    assert pyproject["tool"]["mypy"]["python_version"] == pin.python_version
    assert pyproject["tool"]["ruff"]["target-version"] == pin.ruff_target
    assert pyproject["tool"]["basedpyright"]["venv"] == pin.venv
    assert "pythonVersion" not in pyproject["tool"]["basedpyright"]


def test_cli_prints_pin_fields(tmp_path):
    _write_pyproject(tmp_path, _PINNED)
    assert toolchain.main(["python-version", "--root", str(tmp_path)]) == 0
    assert toolchain.main(["venv", "--root", str(tmp_path)]) == 0
    assert toolchain.main(["check", "--root", str(tmp_path)]) == 0


def test_mypy_restatement_must_match_the_pin(tmp_path):
    _write_pyproject(
        tmp_path, _PINNED.replace('python_version = "3.13"', 'python_version = "3.12"')
    )
    findings = toolchain.integrity_findings(tmp_path)
    assert any("mypy" in item and "3.12" in item for item in findings)


def test_ruff_target_must_match_the_pin(tmp_path):
    _write_pyproject(tmp_path, _PINNED.replace("py313", "py312"))
    findings = toolchain.integrity_findings(tmp_path)
    assert any("target-version" in item and "py313" in item for item in findings)


def test_basedpyright_must_not_restate_python_version(tmp_path):
    extra = _PINNED.replace(
        'venv = ".venv"\nextraPaths', 'venv = ".venv"\npythonVersion = "3.13"\nextraPaths'
    )
    _write_pyproject(tmp_path, extra)
    findings = toolchain.integrity_findings(tmp_path)
    assert any("pythonVersion" in item for item in findings)


def test_basedpyright_venv_must_match_the_pin(tmp_path):
    extra = _PINNED.replace(
        '[tool.basedpyright]\nvenvPath = "."\nvenv = ".venv"',
        '[tool.basedpyright]\nvenvPath = "."\nvenv = ".elsewhere"',
    )
    _write_pyproject(tmp_path, extra)
    findings = toolchain.integrity_findings(tmp_path)
    assert any("basedpyright" in item and ".elsewhere" in item for item in findings)


def test_pin_below_requires_python_is_a_finding(tmp_path):
    _write_pyproject(
        tmp_path,
        _PINNED.replace('python-version = "3.13"', 'python-version = "3.11"')
        .replace('python_version = "3.13"', 'python_version = "3.11"')
        .replace("py313", "py311"),
    )
    findings = toolchain.integrity_findings(tmp_path)
    assert any("requires-python" in item for item in findings)
