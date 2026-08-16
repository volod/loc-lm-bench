"""Read what `uv.lock` and `pyproject.toml` say, and what the active environment actually holds.

The three questions `llb.build.lock_guard` decides on are all answered here, so the policy module
stays about the DECISION: which packages this project declares (`declared_names`), which versions
the lock resolved for each (`lock_index`), and which of those the interpreter currently has
(`installed_versions`). Nothing here evaluates a marker or a specifier -- a package the lock forked
across conflicting extras simply reports both versions, and the guard decides what to do with that.

`venv_versions` answers the same "what is installed" question about a venv ON DISK rather than the
running interpreter, which is what `llb.build.venv_state` needs: it reads a venv uv is about to
replace, from a process that is not running inside it.
"""

import importlib
import importlib.metadata as metadata
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path

LOCK_FILENAME = "uv.lock"
PYPROJECT_FILENAME = "pyproject.toml"
SITE_PACKAGES_GLOB = "lib/python*/site-packages"
DIST_INFO_SUFFIX = ".dist-info"
# `declared_groups` keys are extra names; the base dependency list is not an extra, so it needs a
# key no `[project.optional-dependencies]` group can collide with.
BASE_GROUP = "(base)"

# PEP 508 requirement text ends the project name at the first extra, specifier, marker, or URL.
_NAME_END = re.compile(r"[\[<>=!~;\s@]")
_NAME_SEPARATORS = re.compile(r"[-_.]+")
_VERSION_SEGMENTS = re.compile(r"[.+-]")
_LEADING_DIGITS = re.compile(r"\d+")


def canonical_name(name: str) -> str:
    """PEP 503 name normalization, inline so the guard needs no `packaging` import."""
    return _NAME_SEPARATORS.sub("-", name.strip()).lower()


def _requirement_name(spec: str) -> str:
    return canonical_name(_NAME_END.split(spec, maxsplit=1)[0])


def declared_groups(pyproject_path: Path) -> dict[str, set[str]]:
    """Which requirement group declares each package: the base list plus one entry per extra.

    `declared_names` is the union and is what the constraint and the drift check work on. The
    split matters only to a REPORT: an off-lock package is put back by re-installing the group
    that declares it, so naming the group turns "ruff is off the lock" into a command to run.
    """
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    groups = {BASE_GROUP: {_requirement_name(spec) for spec in project.get("dependencies", [])}}
    for name, specs in project.get("optional-dependencies", {}).items():
        groups[name] = {_requirement_name(spec) for spec in specs}
    return groups


def declared_names(pyproject_path: Path) -> set[str]:
    """Every package this project declares: base dependencies plus every optional-extra group.

    That set -- not vLLM's dependency tree -- is what `make ci` type-checks and tests against, so
    it is exactly what must not move underneath the venv.
    """
    return set().union(*declared_groups(pyproject_path).values())


def lock_index(lock_path: Path) -> dict[str, set[str]]:
    """Map every locked package to the version(s) `uv.lock` carries for it.

    A package resolves to more than one version when conflicting extras pin it apart (`mcp` is
    1.26.0 under the crewai extra and 1.28.1 everywhere else), so the value is a SET and "matches
    the lock" means "is one of these" -- no extra-marker evaluation required.
    """
    locked: dict[str, set[str]] = {}
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    for package in lock.get("package", []):
        version = package.get("version")
        if version:
            locked.setdefault(canonical_name(package["name"]), set()).add(version)
    return locked


def sorted_versions(versions: Iterable[str]) -> list[str]:
    """Order versions low to high by release segment; a pre-release sorts under its release.

    Deliberately not `packaging.version`, which this project does not declare: the only versions
    ever ordered here are the two-or-three a SINGLE lock file resolved for one forked package, and
    the callers only need its lowest and highest. It is not full PEP 440 -- a `.postN` suffix sorts
    like a pre-release rather than above the release -- and it does not have to be: a constraint
    built from a mis-ordered bound still cannot admit a version the lock lacks without
    `lock_guard.enforce` naming it.
    """
    return sorted(versions, key=_version_key)


def _version_key(version: str) -> tuple[tuple[int, int, str], ...]:
    return tuple(_segment_key(part) for part in _VERSION_SEGMENTS.split(version))


def _segment_key(segment: str) -> tuple[int, int, str]:
    digits = _LEADING_DIGITS.match(segment)
    number = int(digits.group()) if digits else 0
    suffix = segment[digits.end() :] if digits else segment
    # A segment carrying a suffix (`0rc1`) is a pre-release of the bare number (`0`), so it sorts
    # first; the suffix itself only breaks ties between two pre-releases of the same number.
    return (number, 0 if suffix else 1, suffix)


def installed_versions(names: Iterable[str]) -> dict[str, str]:
    """Versions the ACTIVE environment holds, skipping packages it does not have.

    `importlib.invalidate_caches()` runs first because this executes in the same interpreter that
    just installed packages, and the metadata finder caches directory listings.
    """
    importlib.invalidate_caches()
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def venv_versions(venv_dir: Path) -> dict[str, str]:
    """Every distribution a venv on disk holds, read from its `*.dist-info` directory names.

    Deliberately a filesystem read rather than an interpreter call: the caller is deciding whether
    that venv is about to be REPLACED, so it must work from outside it and must not depend on the
    venv's own python still running. A wheel escapes `-`, `_`, and `.` in its name to `_` before
    joining it to the version, so the last hyphen is always the split point.
    """
    found: dict[str, str] = {}
    for site_packages in sorted(venv_dir.glob(SITE_PACKAGES_GLOB)):
        for entry in site_packages.glob(f"*{DIST_INFO_SUFFIX}"):
            name, separator, version = entry.name[: -len(DIST_INFO_SUFFIX)].rpartition("-")
            if separator and version:
                found[canonical_name(name)] = version
    return found
