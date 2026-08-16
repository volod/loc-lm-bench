"""Shared venv fixtures for the `make venv` staleness tests.

Not collected by pytest (the module name does not start with `test_`). `write_venv` builds only
the two things the build helpers read -- `pyvenv.cfg` and the `*.dist-info` directory names -- so a
stale venv, a removed interpreter, or a CUDA stack can be staged without creating a real one.
"""

import sys
from pathlib import Path

from llb.build import lock_reader, venv_interpreter

RUNNING_VERSION = venv_interpreter.format_version(tuple(sys.version_info[:3]))
# A version this interpreter cannot be, so a venv stamped with it reads as stale. Same
# `major.minor` on purpose: that is the patch move the restamp remedy covers.
PATCHED_AWAY = f"{sys.version_info[0]}.{sys.version_info[1]}.0"
OTHER_MINOR = f"{sys.version_info[0]}.{sys.version_info[1] - 1}.0"

LOCK_FIXTURE = """
version = 1

[[package]]
name = "torch"
version = "2.12.1"

[[package]]
name = "numpy"
version = "2.4.6"

[[package]]
name = "bitsandbytes"
version = "0.49.2"
"""

# `bitsandbytes` mirrors the real shape that made a lock match insufficient: the lock carries it,
# but only an extra `make venv` does not sync declares it.
PYPROJECT_FIXTURE = """
[project]
name = "llb"
dependencies = ["numpy>=2.0"]

[project.optional-dependencies]
dev = ["ruff>=0.6"]
finetune = ["bitsandbytes>=0.46.1"]
"""


def write_venv(
    venv_dir: Path,
    *,
    version_info: str = RUNNING_VERSION,
    home: str | None = None,
    packages: dict[str, str] | None = None,
) -> Path:
    """Write a venv's `pyvenv.cfg` plus `*.dist-info` directories; nothing else is read."""
    site_packages = venv_dir / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}"
    site_packages = site_packages / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    for name, version in (packages or {}).items():
        (site_packages / f"{name.replace('-', '_')}-{version}.dist-info").mkdir(exist_ok=True)
    home_dir = home if home is not None else str(Path(sys.executable).parent)
    (venv_dir / venv_interpreter.PYVENV_CFG).write_text(
        f"home = {home_dir}\nimplementation = CPython\nversion_info = {version_info}\n"
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    return venv_dir


def write_project(root: Path) -> Path:
    """A project root carrying the lock and pyproject the venv plan prices a rebuild against."""
    (root / lock_reader.LOCK_FILENAME).write_text(LOCK_FIXTURE, encoding="utf-8")
    (root / lock_reader.PYPROJECT_FILENAME).write_text(PYPROJECT_FIXTURE, encoding="utf-8")
    return root
