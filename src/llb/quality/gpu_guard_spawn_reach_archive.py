"""What ships inside an ARCHIVE -- so a zip-imported stdlib is not read as a stdlib that is absent.

Both halves of the reach measurement are DIRECTORY-shaped. `gpu_guard_spawn_reach.spawn_scan` walks
`rglob("*.py")`, and `gpu_guard_spawn_reach_coverage` reads its evidence off filenames under the
root: a shared object beside `lib-dynload`, a `.pyc` with no `.py`, a `__pycache__` stem inside a
package. That is the layout every source-stripped install this host can produce actually uses, and
it is not the only layout CPython ships. A stdlib imported from a zip -- `pythonXY.zip` on
`sys.path`, what an embedded or single-file build carries -- has no package directory to walk at
all, so every one of those reads finds nothing and the names fall through to `absent`: *nothing
under the root, so this host does not ship the module, and the claim holds vacuously for it*. Which
is false. The module is importable, it can start a child, and the scan never saw it.

Measured over fabricated archives (see `tests/llb/quality/test_gpu_guard_spawn_reach.py`), the
untreated reading reports:

- root IS `pythonXY.zip`, or the root holds it beside `lib-dynload`: 0 files read, so
  `audit_spawn_reach` refuses the tree as `unscanned` -- but the coverage line calls every declared
  name `absent` and `audit_read_coverage` passes clean, so the one check that speaks about the
  stdlib says the stdlib is not there.
- MIXED -- part of the library as source on disk, the rest in the archive: files ARE read, so
  nothing is refused anywhere, and an archived `subprocess` is reported `absent` while an archived
  `multiprocessing/util.pyc` produces no submodule finding at all. A clean coverage line over a
  library half of which was never read.

The entries of the archive are the evidence here, exactly as the directory entries are one layout
over. `zipfile` reads the name list without importing anything and without disassembling a `.pyc`,
so a name that ships in an archive is accounted for by the same evidence-based split the rest of the
classification uses -- and is REFUSED rather than counted, because reading the name list says the
module is there and says nothing about whether it starts a child. That is the honest statement this
scan can make about an archive: not measured here. An archive that carries `.py` entries could in
principle be parsed instead of refused; no layout that actually ships does (the embeddable builds
carry `.pyc` only), and the finding names the archive, so widening the scan to read INTO one is a
decision to take on a host that produces the case rather than an unexercised reader written ahead of
it.
"""

import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path

from llb.quality.gpu_guard_spawn_reach import is_excluded

# The name CPython puts on `sys.path` for a zip-imported standard library, beside the stdlib
# directory rather than inside it -- `/usr/lib/python313.zip` on this host, where the file is a
# placeholder that does not exist. Only a candidate that opens as a zip is read, so an ordinary
# source install contributes nothing.
ZIP_STDLIB_NAME = f"python{sys.version_info.major}{sys.version_info.minor}.zip"

# What zipimport loads out of an archive. `__pycache__` is deliberately absent: zipimport reads
# `pkg/util.pyc` in place of `pkg/util.py`, never a cache directory, so an archive is flat where a
# directory tree is not.
_IMPORTABLE_SUFFIXES = (".py", ".pyc")

# A cached or archived `__init__` names its PACKAGE, which is a top-level name the declared list
# already carries -- reporting it again one level down would be the same finding twice.
PACKAGE_INIT = "__init__"


def stdlib_archives(root: Path) -> tuple[Path, ...]:
    """The archives a module can be imported from at this root, in the three places one can sit.

    The root IS the archive (a single-file build points its stdlib path straight at the zip), the
    root CONTAINS one (beside `lib-dynload`, the ordinary zip-stdlib layout), or the interpreter
    carries one next to the root under the name it puts on `sys.path`. The sibling is looked up by
    that exact name rather than by glob: the parent of the stdlib directory is a shared library
    directory on most hosts, and an unrelated zip found there is not evidence about this stdlib.
    """
    candidates = (root, *sorted(root.glob("*.zip")), root.parent / ZIP_STDLIB_NAME)
    found: list[Path] = []
    for candidate in candidates:
        if candidate not in found and candidate.is_file() and zipfile.is_zipfile(candidate):
            found.append(candidate)
    return tuple(found)


def archived_modules(archives: Iterable[Path]) -> tuple[str, ...]:
    """Every module name these archives carry -- top-level names bare, submodules dotted.

    Read off the name list, so nothing is imported and no `.pyc` is disassembled. Source and cached
    entries count alike, because the question is what this host can import and the scan did not
    read, which a `.py` inside an archive answers the same way a `.pyc` does. The scan's own
    exclusion rule applies to the entry paths, so the two halves cannot disagree about which
    directories the statement covers.
    """
    names: set[str] = set()
    for archive in archives:
        for entry in _entries(archive):
            name = _module_name(entry)
            if name is not None:
                names.add(name)
    return tuple(sorted(names))


def unread_submodules(root: Path, names: Iterable[str], packages: Iterable[str]) -> tuple[str, ...]:
    """Archived submodules of a package whose source the scan DID read off the directory tree.

    The mixed layout, one level down: a package present on disk classifies as read, and whichever of
    its submodules ship only in the archive were never parsed. A submodule whose `.py` (or package
    `__init__.py`) is on disk is not one of them -- the archive entry is then a second copy of
    something the scan read. A submodule of a package that is itself archived is left to that
    package's own finding, the same way a cached `__init__` is left to the name that classifies it.
    """
    refused = frozenset(packages)
    return tuple(
        name
        for name in sorted(names)
        if "." in name and name.split(".")[0] not in refused and not _has_source(root, name)
    )


def _has_source(root: Path, name: str) -> bool:
    """Whether the directory tree carries this dotted name as source, as a module or a package."""
    path = root.joinpath(*name.split("."))
    return path.with_suffix(".py").is_file() or (path / f"{PACKAGE_INIT}.py").is_file()


def _entries(archive: Path) -> tuple[str, ...]:
    """The archive's own name list, or nothing if it cannot be read as one."""
    try:
        with zipfile.ZipFile(archive) as opened:
            return tuple(opened.namelist())
    except (OSError, zipfile.BadZipFile):
        return ()


def _module_name(entry: str) -> str | None:
    """The dotted module one archive entry ships, or None if the entry is not an importable one."""
    if entry.endswith("/") or is_excluded(entry):
        return None
    parts = entry.split("/")
    if not parts[-1].endswith(_IMPORTABLE_SUFFIXES):
        return None
    # `util.pyc` and `util.cpython-313.pyc` alike: the stem before the first dot is the module,
    # which is the same rule the directory-side classification reads a cached name with.
    named = [*parts[:-1], parts[-1].split(".")[0]]
    if named[-1] == PACKAGE_INIT:
        named.pop()
    return ".".join(named) if named and all(part.isidentifier() for part in named) else None
