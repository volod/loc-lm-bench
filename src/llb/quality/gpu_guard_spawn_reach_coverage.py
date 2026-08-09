"""What the reach scan FAILED to read -- so the result names the stdlib it was read from.

`llb.quality.gpu_guard_spawn_reach` counts the files it read, and the audit refuses only the
degenerate case where that count is ZERO. Between an empty read and a complete one is an unmeasured
middle: a module that ships without source -- a frozen or zipped stdlib, a `.pyc`-only install --
is never parsed and reports exactly like a module that starts no children. The file count cannot
tell them apart, because it says how much was read and not what was missed.

So the reading is measured against `sys.stdlib_module_names`, the interpreter's own list of what its
standard library contains. Every declared name the scan read no source for is classified, and the
classification is the interesting half, because most of that list has no source BY CONSTRUCTION and
a gate that refused every unread name would fail on every host:

- `compiled` -- linked into the interpreter binary (`sys.builtin_module_names`).
- `extensions` -- a shared object under the stdlib root or its `lib-dynload`.
- `declared` -- `SOURCELESS_STDLIB_MODULES`: the frozen bootstrap modules, plus the names of other
  platforms, which the list carries because it is documented platform-independent.
- `compiled_only` -- a `.pyc` under the stdlib root with no `.py` beside it. This is the one the
  audit REFUSES: the module is importable on this host, so it can start a child, and the scan did
  not read it. That is the frozen / source-stripped layout stated as a finding.
- `absent` -- nothing under the root at all, so this host does not ship the module. Reported, never
  refused: a `python3-minimal` or split-package layout (Debian ships `tkinter` apart) cannot import
  what it does not have, so "starts no child undeclared" holds vacuously for it.

Measured here (CPython 3.13, `/usr/lib/python3.13`): of 290 declared names, **184 read as source,
61 compiled in, 35 extensions, 10 declared sourceless, 0 compiled-only, 0 absent** -- every unread
name accounted for, which is what makes the stdlib result a statement about the stdlib rather than
about whichever files this host happened to ship.

The scan's excluded directory segments (`test`, `tests`, `idle_test`, `site-packages`) are not a
hole in this measurement: none of them is a name `sys.stdlib_module_names` carries, which is
asserted rather than assumed in `tests/llb/quality/test_gpu_guard_spawn_reach.py`.
"""

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

from llb.quality.gpu_guard_spawn_reach import SpawnScan

_FROZEN = (
    "frozen into the interpreter at build time, so it is importable with no file under the stdlib "
    "root; its source ships as `importlib/{module}.py`, which the scan does read"
)
# `sys.stdlib_module_names` is documented platform-independent: it names what CPython's standard
# library contains, not what this host can import. These are the extension modules of the other two
# platforms -- including `_winapi` and `nt`, the Windows twins the spawn alphabet already declares.
_OTHER_PLATFORM = "an extension module of {platform}, named by a platform-independent list"

SOURCELESS_STDLIB_MODULES: Mapping[str, str] = {
    "_frozen_importlib": _FROZEN.format(module="_bootstrap"),
    "_frozen_importlib_external": _FROZEN.format(module="_bootstrap_external"),
    "_overlapped": _OTHER_PLATFORM.format(platform="Windows"),
    "_winapi": _OTHER_PLATFORM.format(platform="Windows"),
    "_wmi": _OTHER_PLATFORM.format(platform="Windows"),
    "msvcrt": _OTHER_PLATFORM.format(platform="Windows"),
    "nt": _OTHER_PLATFORM.format(platform="Windows"),
    "winreg": _OTHER_PLATFORM.format(platform="Windows"),
    "winsound": _OTHER_PLATFORM.format(platform="Windows"),
    "_scproxy": _OTHER_PLATFORM.format(platform="macOS"),
}

# Where a compiled module can sit under the stdlib root without a `.py` beside it: a package's
# cached `__init__`, a top-level cached module, and the two flat layouts a stripped install uses.
_COMPILED_PATTERNS = (
    "*/__pycache__/__init__.*.pyc",
    "__pycache__/*.pyc",
    "*/__init__.pyc",
    "*.pyc",
)


# The classification, in the order a name is tried against it -- also the fields of the record
# below, so the buckets and the report cannot drift apart.
_FIELDS = ("read", "compiled", "extensions", "declared", "compiled_only", "absent")


@dataclass(frozen=True)
class ReadCoverage:
    """How much of this interpreter's declared stdlib the scan actually read, and why the rest not.

    A partition of `sys.stdlib_module_names`: every declared name lands in exactly one field, so the
    counts add up to the list and a name cannot go missing between two of them.
    """

    root: str
    read: tuple[str, ...]
    compiled: tuple[str, ...]
    extensions: tuple[str, ...]
    declared: tuple[str, ...]
    compiled_only: tuple[str, ...]
    absent: tuple[str, ...]

    @property
    def unread(self) -> tuple[str, ...]:
        """Every declared name no source was read for, whatever the reason."""
        return tuple(
            sorted(
                self.compiled + self.extensions + self.declared + self.compiled_only + self.absent
            )
        )


def stdlib_read_coverage(
    scan: SpawnScan,
    names: Iterable[str] | None = None,
    sourceless: Mapping[str, str] = SOURCELESS_STDLIB_MODULES,
) -> ReadCoverage:
    """Weigh what one stdlib scan read against what this interpreter says its stdlib contains."""
    root = Path(scan.root)
    declared_names = sorted(names if names is not None else sys.stdlib_module_names)
    read = frozenset(scan.modules_read)
    extensions = _extension_stems(root)
    compiled_only = _compiled_stems(root)
    buckets: dict[str, list[str]] = {}
    for name in declared_names:
        kind = "read" if name in read else _kind(name, extensions, compiled_only, sourceless)
        buckets.setdefault(kind, []).append(name)
    return ReadCoverage(
        root=scan.root, **{field: tuple(buckets.get(field, ())) for field in _FIELDS}
    )


def read_coverage_message(coverage: ReadCoverage) -> str:
    """The operator-facing line: what was read, and what each unread name is excused by."""
    listed = ", ".join(
        f"{len(getattr(coverage, field))} {field.replace('_', '-')}"
        for field in _FIELDS
        if field != "read"
    )
    return (
        f"[gpu-guard] stdlib read coverage under {coverage.root}: {len(coverage.read)} of "
        f"{len(coverage.read) + len(coverage.unread)} declared modules read as source; {listed}"
        f"{_named('compiled-only', coverage.compiled_only)}{_named('absent', coverage.absent)}"
    )


def _kind(
    name: str,
    extensions: frozenset[str],
    compiled_only: frozenset[str],
    sourceless: Mapping[str, str],
) -> str:
    """Why one declared name was not read -- the first construction that accounts for it."""
    if name in sys.builtin_module_names:
        return "compiled"
    if name in extensions:
        return "extensions"
    if name in sourceless:
        return "declared"
    return "compiled_only" if name in compiled_only else "absent"


def _extension_stems(root: Path) -> frozenset[str]:
    """Module names shipped as a shared object, read off the filenames rather than imported."""
    suffixes = tuple(EXTENSION_SUFFIXES)
    return frozenset(
        path.name.split(".")[0]
        for directory in (root, root / "lib-dynload")
        if directory.is_dir()
        for path in directory.iterdir()
        if path.name.endswith(suffixes)
    )


def _compiled_stems(root: Path) -> frozenset[str]:
    """Module names with a `.pyc` under the root -- importable whether or not source is there."""
    return frozenset(
        _compiled_stem(root, path) for pattern in _COMPILED_PATTERNS for path in root.glob(pattern)
    )


def _compiled_stem(root: Path, path: Path) -> str:
    """A cached `__init__` names its package; anything else names itself."""
    if path.name.split(".")[0] == "__init__":
        return path.relative_to(root).parts[0]
    return path.name.split(".")[0]


def _named(label: str, names: tuple[str, ...]) -> str:
    return f" -- {label}: {', '.join(names)}" if names else ""
