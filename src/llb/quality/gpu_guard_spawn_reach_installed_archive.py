"""A dependency that ships ZIPPED, read rather than skipped -- the archive half of the venv scan.

`llb.quality.gpu_guard_spawn_reach_installed` walks a DIRECTORY, and the import path does not stop
at one. A dependency shipped as an archive -- a zipped egg, a `--zip-ok` install, any `sys.path`
entry that is a file rather than a directory -- has no package directory to walk, so `rglob("*.py")`
finds nothing in it, nothing is parsed, and the audit returns clean for a venv half of which it never
opened. This module is the other half of that pass: the archives the import path carries, read as
files rather than as trees.

The read-or-refuse call `gpu_guard_spawn_reach_archive` deferred for the stdlib is taken the other
way here, and on evidence. A zip-shipped STDLIB is `.pyc`-only -- the layout the embeddable builds
ship -- so there is nothing to parse there and every archived name is refused as unread. A zip-shipped
DEPENDENCY is not: `bdist_egg` zips the source tree, so a zipped egg carries `.py` entries, and one
fabricated on this host imports through `zipimport` with its source readable straight out of the
archive (`tests/llb/quality/test_gpu_guard_spawn_reach_installed.py` builds exactly that layout).
Reading `pkg/backend/start.py` out of a zip is the same evidence as reading it off disk -- the same
bytes through the same parser, resolved through the same imports -- so it is parsed and counted as
read, and whatever reach it finds is weighed against the same package excuse a file on disk would be.

What is left over is what gets refused: a module the archive ships compiled with no source anywhere
-- not in the archive, not as a copy in the directory tree. That is the same statement the stdlib
half makes about its own archives, for the same reason: a name list says the module is there and
says nothing about whether it starts a child.

The two halves are folded into ONE `SpawnScan` by `with_archives` rather than reported beside it, so
the counts add up over the whole import path. A venv whose packages all ship zipped then reads as a
scan that read source, instead of as a tree `audit_spawn_reach` refuses as `unscanned` while the
source sat in a zip nobody opened.
"""

import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from llb.quality.gpu_guard_spawn_reach import ModuleReach, SpawnScan, module_triggers
from llb.quality.gpu_guard_spawn_reach_archive import (
    SOURCE_SUFFIX,
    ZIP_STDLIB_NAME,
    archive_entries,
    entry_module,
    has_source,
    openable_archives,
)
from llb.quality.gpu_guard_spawn_source import source_reaches

# The suffixes a zip-shipped distribution arrives under. `.egg` is what `bdist_egg` and a `--zip-ok`
# install produce; a bare `.zip` is what a hand-built or vendored one does. A `.whl` is deliberately
# absent: it is an install FORMAT that is unpacked, never an import path entry.
_ARCHIVE_SUFFIXES = ("*.egg", "*.zip")


def installed_archives(root: Path, path_entries: Iterable[str] | None = None) -> tuple[Path, ...]:
    """The archives a dependency can be imported from: `sys.path` entries plus zips under the root.

    Both places one can sit. An entry ON the import path is what an installed zipped egg actually
    is -- `easy-install.pth` and `sys.path` manipulation put it there, and it need not live under
    site-packages at all -- while an archive sitting IN the root is the same distribution before
    anything has added it, which this reads too rather than deciding on a `.pth` file's behalf.

    The stdlib's own zip is skipped by name: `gpu_guard_spawn_reach_archive` accounts for that
    layout, and reporting it here would be the same finding twice, from the scan that has no
    business speaking about the stdlib.
    """
    candidates = (
        *(Path(entry) for entry in path_entries or () if entry),
        *sorted(path for suffix in _ARCHIVE_SUFFIXES for path in root.glob(suffix)),
    )
    return openable_archives(
        candidate for candidate in candidates if candidate.name != ZIP_STDLIB_NAME
    )


@dataclass(frozen=True)
class ArchiveReading:
    """What one archive on the import path contributed: what was parsed, and what could not be.

    `read` and `unread` are the dotted module names either side of that split, so the two add up to
    what the archive ships and a name cannot fall between them. Carrying them as names rather than
    as a count is what lets a module shipped compiled in one archive and as source in another come
    out read -- the same rule that keeps an archived copy of an installed file out of the refusal.
    """

    read: tuple[str, ...]
    reaches: tuple[ModuleReach, ...]
    unread: tuple[str, ...]


def archive_reaches(
    archive: Path, alphabet: Mapping[str, frozenset[str]], triggers: Iterable[bytes]
) -> ArchiveReading:
    """Read one archive the way the directory scan reads a tree: parse the source, name the rest.

    A module the archive carries as `.py` is parsed out of the zip -- evidence, not a refusal. A
    module it carries only compiled is `unread`: importable here, and nothing says whether it starts
    a child. The trigger prefilter is the directory scan's, so an archive costs a parse only for the
    entries whose bytes could contain a call in the alphabet.
    """
    prefilter = tuple(triggers)
    reaches: list[ModuleReach] = []
    read: list[str] = []
    unread: list[str] = []
    for name, entry in sorted(_archive_sources(archive).items()):
        source = None if entry is None else _entry_bytes(archive, entry)
        if entry is None or source is None:
            unread.append(name)
            continue
        read.append(name)
        if any(trigger in source for trigger in prefilter):
            primitives = source_reaches(source, alphabet)
            if primitives:
                reaches.append(ModuleReach(entry, primitives, container=str(archive)))
    return ArchiveReading(tuple(read), tuple(reaches), tuple(unread))


def with_archives(
    scan: SpawnScan,
    root: Path,
    archives: Iterable[Path],
    alphabet: Mapping[str, frozenset[str]],
) -> SpawnScan:
    """Fold what the archives contributed into the directory pass, as one scan of one import path.

    A name any archive carries as source is not unread, whichever archive shipped it compiled, and
    neither is one the directory tree carries: an archive beside an installed source tree is a copy
    of what was already parsed, which is the rule the stdlib half applies to its own submodules.
    """
    opened = tuple(archives)
    if not opened:
        return scan
    readings = [archive_reaches(archive, alphabet, module_triggers(alphabet)) for archive in opened]
    reaches = tuple(reach for reading in readings for reach in reading.reaches)
    read = [name for reading in readings for name in reading.read]
    unread = {name for reading in readings for name in reading.unread} - set(read)
    return SpawnScan(
        root=scan.root,
        files_read=scan.files_read + len(read),
        modules_read=tuple(sorted(set(scan.modules_read) | {name.split(".")[0] for name in read})),
        reaches=scan.reaches + reaches,
        archives=tuple(str(archive) for archive in opened),
        unread_archived=tuple(sorted(name for name in unread if not has_source(root, name))),
        sites=scan.sites,
    )


def _archive_sources(archive: Path) -> dict[str, str | None]:
    """Every module the archive ships -> the `.py` entry carrying its source, or None if compiled.

    Two entries of a module shipped twice collapse onto one name, source winning: a zipped egg
    carrying `pkg/util.py` beside `pkg/util.pyc` is a module that CAN be read, and reading it is the
    whole point of preferring the archive's source to its name list.
    """
    sources: dict[str, str | None] = {}
    for entry in archive_entries(archive):
        name = entry_module(entry)
        if name is None:
            continue
        if entry.endswith(SOURCE_SUFFIX):
            sources[name] = entry
        else:
            sources.setdefault(name, None)
    return sources


def _entry_bytes(archive: Path, entry: str) -> bytes | None:
    """One entry's contents, or None if the archive will not give them up."""
    try:
        with zipfile.ZipFile(archive) as opened:
            return opened.read(entry)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
