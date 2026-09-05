"""The interpreter pin `make venv` and the IDE type checker both have to obey.

`[tool.llb.toolchain]` in pyproject.toml is the only place the default Python minor and the
default venv directory name are written. Make reads those values at parse time so `make venv`
creates that interpreter at that path. mypy, ruff, and basedpyright cannot interpolate the
section, so each still carries its own key -- this module is the join that fails when one of
them is redefined.

`make venv PYTHON_VERSION=...` and `make venv VENV=...` still override the defaults (Make
command-line variables). basedpyright cannot follow a one-off `VENV=`, so it names the default
venv directory and must not restate `pythonVersion`: the interpreter inside that venv is the
version, including after an override that still wrote into the default path.
"""

import argparse
import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

PYPROJECT_FILENAME = "pyproject.toml"
TOOL_TABLE = "llb"
TOOLCHAIN_TABLE = "toolchain"
PYTHON_VERSION_KEY = "python-version"
VENV_KEY = "venv"
CHECK_COMMAND = "check"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MINOR = re.compile(r"^(\d+)\.(\d+)$")
_REQUIRES_GE = re.compile(r"^>=\s*(\d+)\.(\d+)\s*$")


@dataclass(frozen=True)
class Toolchain:
    """Default `make venv` interpreter minor and venv directory name."""

    python_version: str
    venv: str

    @property
    def ruff_target(self) -> str:
        return "py" + self.python_version.replace(".", "")

    @property
    def minor(self) -> tuple[int, int]:
        matched = _MINOR.match(self.python_version)
        if not matched:
            raise ValueError(
                f"toolchain python-version must be major.minor, got {self.python_version!r}"
            )
        return int(matched.group(1)), int(matched.group(2))


def load_pyproject(root: Path) -> dict[str, Any]:
    return tomllib.loads((root / PYPROJECT_FILENAME).read_text(encoding="utf-8"))


def load_toolchain(root: Path = _REPO_ROOT) -> Toolchain:
    table = load_pyproject(root)["tool"][TOOL_TABLE][TOOLCHAIN_TABLE]
    return Toolchain(python_version=str(table[PYTHON_VERSION_KEY]), venv=str(table[VENV_KEY]))


def _tool(pyproject: dict[str, Any], *names: str) -> dict[str, Any]:
    cursor: Any = pyproject.get("tool", {})
    for name in names:
        if not isinstance(cursor, dict):
            return {}
        cursor = cursor.get(name, {})
    return cursor if isinstance(cursor, dict) else {}


def integrity_findings(root: Path = _REPO_ROOT) -> list[str]:
    """Every restatement of the toolchain pin that no longer matches `[tool.llb.toolchain]`."""
    pyproject = load_pyproject(root)
    findings: list[str] = []
    try:
        pin = load_toolchain(root)
        pin.minor
    except (KeyError, TypeError, ValueError) as exc:
        return [f"{PYPROJECT_FILENAME}: [tool.llb.toolchain] is unreadable: {exc}"]

    mypy_version = _tool(pyproject, "mypy").get("python_version")
    if mypy_version != pin.python_version:
        findings.append(
            f"{PYPROJECT_FILENAME}: [tool.mypy] python_version={mypy_version!r} must equal "
            f"[tool.llb.toolchain] python-version={pin.python_version!r}"
        )

    ruff_target = _tool(pyproject, "ruff").get("target-version")
    if ruff_target != pin.ruff_target:
        findings.append(
            f"{PYPROJECT_FILENAME}: [tool.ruff] target-version={ruff_target!r} must equal "
            f"{pin.ruff_target!r} (from [tool.llb.toolchain] python-version={pin.python_version!r})"
        )

    based = _tool(pyproject, "basedpyright")
    if based:
        if "pythonVersion" in based:
            findings.append(
                f"{PYPROJECT_FILENAME}: [tool.basedpyright] must not set pythonVersion -- "
                f"`make venv PYTHON_VERSION=` changes the interpreter in {pin.venv}; restating "
                f"a minor here would not follow it"
            )
        based_venv = based.get("venv")
        if based_venv != pin.venv:
            findings.append(
                f"{PYPROJECT_FILENAME}: [tool.basedpyright] venv={based_venv!r} must equal "
                f"[tool.llb.toolchain] venv={pin.venv!r}"
            )
        based_path = based.get("venvPath")
        if based_path not in (None, "."):
            findings.append(
                f"{PYPROJECT_FILENAME}: [tool.basedpyright] venvPath={based_path!r} must be '.' "
                f"(the default venv lives next to pyproject.toml)"
            )

    requires = str(pyproject.get("project", {}).get("requires-python", ""))
    matched = _REQUIRES_GE.match(requires)
    if not matched:
        findings.append(
            f"{PYPROJECT_FILENAME}: project.requires-python={requires!r} must be a '>=X.Y' floor "
            "so the toolchain pin can be checked against it"
        )
    else:
        floor = (int(matched.group(1)), int(matched.group(2)))
        if pin.minor < floor:
            findings.append(
                f"{PYPROJECT_FILENAME}: [tool.llb.toolchain] python-version={pin.python_version!r} "
                f"is below project.requires-python={requires!r}"
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="read or check the [tool.llb.toolchain] pin")
    parser.add_argument("command", choices=(PYTHON_VERSION_KEY, VENV_KEY, CHECK_COMMAND))
    parser.add_argument("--root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args(argv)
    if args.command == CHECK_COMMAND:
        findings = integrity_findings(args.root)
        for finding in findings:
            _LOG.error("ERROR: %s", finding)
        _LOG.info("[toolchain] %d integrity finding(s)", len(findings))
        return 1 if findings else 0
    pin = load_toolchain(args.root)
    print(pin.python_version if args.command == PYTHON_VERSION_KEY else pin.venv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
