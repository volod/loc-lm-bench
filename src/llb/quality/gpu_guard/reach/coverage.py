"""What the reach scan FAILED to read -- so the result names the stdlib it was read from.

`llb.quality.gpu_guard.reach.scan` counts the files it read, and the audit refuses only the
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
- `compiled_only` -- a `.pyc` under the stdlib root with no `.py` beside it. This is one the audit
  REFUSES: the module is importable on this host, so it can start a child, and the scan did not read
  it. That is the frozen / source-stripped layout stated as a finding.
- `archived` -- shipped inside a zip the interpreter imports from, which every read here and in the
  scan is too directory-shaped to see. Refused for the same reason as `compiled_only`, off the
  archive's own name list; `llb.quality.gpu_guard.reach.archive` states the case.
- `absent` -- nothing under the root at all, so this host does not ship the module. Reported, never
  refused: a `python3-minimal` or split-package layout (Debian ships `tkinter` apart) cannot import
  what it does not have, so "starts no child undeclared" holds vacuously for it.

Measured here (CPython 3.13, `/usr/lib/python3.13`): of 290 declared names, **184 read as source,
61 compiled in, 35 extensions, 10 declared sourceless, 0 compiled-only, 0 archived, 0 absent** --
every unread name accounted for, which is what makes the stdlib result a statement about the stdlib
rather than about whichever files this host happened to ship.

That classification is per TOP-LEVEL name, because `sys.stdlib_module_names` is the only list of its
kind CPython publishes -- so a package that ships its `__init__.py` and not its submodules counts as
read, and the source-stripped layout the measurement exists for hides one level down:
`multiprocessing/__init__.py` present with `multiprocessing/util.py` stripped reads exactly like a
complete package. That level needs no published list, because the interpreter leaves the evidence on
disk. `compiled_only_submodules` walks the same directories the scan walked and compares each
package's `.py` stems against its `__pycache__/*.pyc` stems: a submodule with a cached module and no
source beside it is the same compiled-only finding one level down, and an absent file is -- as at the
name level -- not a finding at all.

Both of those comparisons read a DIRECTORY, which is the layout every source-stripped install this
host can produce -- and not the only layout CPython ships. A stdlib imported from a zip has no
package directory to walk, so `archived` / `archived_submodules` account for it off the archive's
own name list instead; `llb.quality.gpu_guard.reach.archive` owns that reading and states the
evidence it was decided on.

The same measurement of the INSTALLED half is
`llb.quality.gpu_guard.reach.installed_coverage`, which weighs its scan against the list
`importlib.metadata` publishes in place of the one the interpreter does. It reuses the submodule
walk and the message rendering here, and classifies what is left with the classes a venv can have
rather than the ones a stdlib can.

The scan's excluded directory segments (`test`, `tests`, `idle_test`, `site-packages`) are not a
hole in any of these measurements: none of them is a name `sys.stdlib_module_names` carries, which
is asserted rather than assumed in `tests/llb/quality/test_gpu_guard_spawn_reach.py`, and the
submodule comparisons skip them through the scan's own `is_excluded` rather than a second copy of
the rule.
"""

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from llb.quality.gpu_guard.reach.scan import SpawnScan
from llb.quality.gpu_guard.reach.archive import (
    archived_modules,
    stdlib_archives,
    unread_submodules,
)
from llb.quality.gpu_guard.reach.layout import (
    compiled_only_submodules,
    extension_stems,
    compiled_stems,
)

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


# The classification, in the order a name is tried against it -- also the fields of the record
# below, so the buckets and the report cannot drift apart.
_FIELDS = ("read", "compiled", "extensions", "declared", "compiled_only", "archived", "absent")


@dataclass(frozen=True)
class ReadCoverage:
    """How much of this interpreter's declared stdlib the scan actually read, and why the rest not.

    The seven name fields are a partition of `sys.stdlib_module_names`: every declared name lands in
    exactly one of them, so the counts add up to the list and a name cannot go missing between two.

    The two submodule fields are deliberately outside that partition: they are the compiled-only and
    archived classes one level down, read off the package directories and the archive name lists
    rather than off a published list, and their entries are dotted submodule names that the declared
    list does not contain. `archives` records which archives were read for the latter, so a coverage
    line says what layout it was measured against.
    """

    root: str
    read: tuple[str, ...]
    compiled: tuple[str, ...]
    extensions: tuple[str, ...]
    declared: tuple[str, ...]
    compiled_only: tuple[str, ...]
    archived: tuple[str, ...]
    absent: tuple[str, ...]
    compiled_only_submodules: tuple[str, ...] = ()
    archived_submodules: tuple[str, ...] = ()
    archives: tuple[str, ...] = ()

    @property
    def unread(self) -> tuple[str, ...]:
        """Every declared name no source was read for, whatever the reason."""
        return tuple(
            sorted(
                self.compiled
                + self.extensions
                + self.declared
                + self.compiled_only
                + self.archived
                + self.absent
            )
        )


def stdlib_read_coverage(
    scan: SpawnScan,
    names: Iterable[str] | None = None,
    sourceless: Mapping[str, str] = SOURCELESS_STDLIB_MODULES,
    archives: Iterable[Path] | None = None,
) -> ReadCoverage:
    """Weigh what one stdlib scan read against what this interpreter says its stdlib contains."""
    root = Path(scan.root)
    declared_names = sorted(names if names is not None else sys.stdlib_module_names)
    read = frozenset(scan.modules_read)
    extensions = extension_stems(root)
    compiled_only = compiled_stems(root)
    found = stdlib_archives(root) if archives is None else tuple(archives)
    archived = archived_modules(found)
    packaged = frozenset(name.split(".")[0] for name in archived)
    buckets: dict[str, list[str]] = {}
    for name in declared_names:
        kind = (
            "read" if name in read else _kind(name, extensions, compiled_only, packaged, sourceless)
        )
        buckets.setdefault(kind, []).append(name)
    return ReadCoverage(
        root=scan.root,
        archives=tuple(str(archive) for archive in found),
        compiled_only_submodules=compiled_only_submodules(root),
        archived_submodules=unread_submodules(root, archived, buckets.get("archived", ())),
        **{field: tuple(buckets.get(field, ())) for field in _FIELDS},
    )


def read_coverage_message(coverage: ReadCoverage) -> str:
    """The operator-facing line: what was read, and what each unread name is excused by."""
    submodules = coverage.compiled_only_submodules
    return (
        f"[gpu-guard] stdlib read coverage under {coverage.root}: {len(coverage.read)} of "
        f"{len(coverage.read) + len(coverage.unread)} declared modules read as source; "
        f"{class_counts(coverage, _FIELDS)}; {len(submodules)} compiled-only and "
        f"{len(coverage.archived_submodules)} archived submodules"
        f"{named_list('compiled-only', coverage.compiled_only)}"
        f"{named_list('archived', coverage.archived)}{named_list('absent', coverage.absent)}"
        f"{named_list('compiled-only submodules', submodules)}"
        f"{named_list('archived submodules', coverage.archived_submodules)}"
        f"{named_list('archives read', coverage.archives)}"
    )


def class_counts(coverage: object, fields: Iterable[str]) -> str:
    """`61 compiled, 35 extensions, ...` -- the unread classes of one partition, in its own order.

    Shared with the installed coverage, whose classes differ and whose reading is the same: the
    counts come off the field tuple, so a class added to either partition is a class its line
    reports without a second edit.
    """
    return ", ".join(
        f"{len(getattr(coverage, field))} {field.replace('_', '-')}"
        for field in fields
        if field != "read"
    )


def named_list(label: str, names: tuple[str, ...]) -> str:
    """`-- label: a, b` for a class with entries, and nothing at all for an empty one."""
    return f" -- {label}: {', '.join(names)}" if names else ""


def _kind(
    name: str,
    extensions: frozenset[str],
    compiled_only: frozenset[str],
    archived: frozenset[str],
    sourceless: Mapping[str, str],
) -> str:
    """Why one declared name was not read -- the first construction that accounts for it."""
    if name in sys.builtin_module_names:
        return "compiled"
    if name in extensions:
        return "extensions"
    if name in sourceless:
        return "declared"
    if name in compiled_only:
        return "compiled_only"
    return "archived" if name in archived else "absent"
