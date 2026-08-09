"""The other DIRECTORIES on the import path -- so the tree this repo itself ships is read too.

`llb.quality.gpu_guard_spawn_reach_installed` walks site-packages and
`llb.quality.gpu_guard_spawn_reach_installed_archive` opens the archives beside it, and between them
they still miss a whole KIND of import-path entry: a directory somewhere else that a `.pth` file
adds. That is not an exotic layout, it is how this repo is installed -- `__editable__.llb-0.1.0.pth`
contains one line, `<repo>/src`, so `llb`'s own modules are parsed by neither scan. The code an
unmarked test runs the most was the one tree nobody asked the question of, while every dependency
around it was held to it.

The evidence is the `.pth` files themselves, read with `site.addpackage`'s own rule: a line that
starts with `import ` or `import\t` is CODE the interpreter runs, anything else is a path resolved
against the directory the file sits in, and a comment or a blank line is neither. Reading them is
what makes this a statement about the import path rather than about the running process --
`sys.path` would answer too, and would answer wrong here, because a test runner puts the repo root
and the test directories on it and a scan of those walks the venv it is trying to describe.

One kind of entry is deliberately left alone:

- An entry INSIDE the scan root. `nvidia-cutlass-dsl` ships one (`nvidia_cutlass_dsl/python_packages`,
  which makes `cutlass` importable), and the directory pass has already read every file in it.
  Reading it again would count those files twice and report one file under two package names --
  once as `cutlass` and once as the `nvidia_cutlass_dsl` its distribution actually publishes, which
  is the name an excuse would be written at.
The common executable form is not left silent: setuptools' generated
`import __editable___pkg_finder; __editable___pkg_finder.install()` line is parsed with `ast`, and
the generated finder's literal `MAPPING` is read without importing or executing either file. Every
other executable line is retained in `SpawnScan.unread_path_entries`, which the audit refuses.

Measured over this host: two literal entries, one of which is under the root, so ONE tree is scanned
-- `<repo>/src`, with no reach below the seams at all -- while `_virtualenv.pth:1` and
`distutils-precedence.pth:1` are named as the two unresolved executable lines. The source-tree
answer is that this repo needs no declaration like a dependency's: its child starts all go through
`subprocess.run` / `subprocess.call` / `subprocess.Popen`, which the denial patches.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.quality.gpu_guard_spawn_reach import (
    ImportPathEntry,
    ModuleReach,
    SpawnScan,
    spawn_scan,
)
from llb.quality.gpu_guard_spawn_reach_installed_finder import finder_paths
from llb.quality.gpu_guard_spawn_reach_installed_paths import source_top_level_names
from llb.quality.gpu_guard_spawn_source import source_reaches

# What `site.addpackage` executes rather than resolves. Matched exactly as the interpreter matches
# it, so a module called `imports.py` on a path line stays a path.
_EXECUTED_PREFIXES = ("import ", "import\t")

PTH_SUFFIX = "*.pth"


@dataclass(frozen=True)
class SitePathEntry:
    """One statically resolved entry; `module` is set for an editable finder mapping."""

    path: Path
    module: str = ""


@dataclass(frozen=True)
class SitePathReading:
    """The entries a static read resolved and the executable lines it could not."""

    entries: tuple[SitePathEntry, ...]
    unread: tuple[str, ...]


def site_path_entries(root: Path) -> SitePathReading:
    """The paths the `.pth` files under one root add, without executing their code.

    Every existing directory is retained, including one inside the scan root: the root walk already
    parsed its files, but coverage still needs the top-level import names that entry provides.
    Duplicate routes collapse to one entry. The setuptools editable finder shape is resolved from
    its literal `MAPPING`; every other executable line is named as unread rather than silently
    skipped.
    """
    found: list[SitePathEntry] = []
    unread: list[str] = []
    for pth in sorted(root.glob(PTH_SUFFIX)):
        reading = _pth_entries(root, pth)
        found.extend(entry for entry in reading.entries if entry not in found)
        unread.extend(reading.unread)
    return SitePathReading(_without_covered_mappings(found), tuple(unread))


def _pth_entries(root: Path, pth: Path) -> SitePathReading:
    literal = tuple(
        entry for line in path_lines(pth) if (entry := _literal_entry(root, line)) is not None
    )
    mapped: list[SitePathEntry] = []
    unread = []
    for number, line in executed_lines(pth):
        paths = finder_paths(root, line)
        if paths is None:
            unread.append(f"{pth.name}:{number}")
        else:
            mapped.extend(SitePathEntry(entry.path, entry.module) for entry in paths)
    return SitePathReading(literal + tuple(mapped), tuple(unread))


def _literal_entry(root: Path, line: str) -> SitePathEntry | None:
    path = (root / line).resolve()
    return SitePathEntry(path) if path.is_dir() else None


def _without_covered_mappings(entries: Iterable[SitePathEntry]) -> tuple[SitePathEntry, ...]:
    found = tuple(entries)
    direct = tuple(entry.path for entry in found if not entry.module)
    return tuple(
        entry
        for entry in found
        if not entry.module or not any(entry.path.is_relative_to(path) for path in direct)
    )


def with_path_entries(
    scan: SpawnScan,
    entries: Iterable[SitePathEntry],
    alphabet: Mapping[str, frozenset[str]],
    triggers: Sequence[bytes],
    unread: Iterable[str] = (),
) -> SpawnScan:
    """Fold what the extra trees contributed into the directory pass, as one scan of one path.

    Each tree is read exactly as the root is -- same alphabet, same prefilter, same exclusions --
    and its reaches carry the tree as their `container`, so a finding names the file an operator has
    to open rather than a path that looks like it is under site-packages and is not.
    """
    resolved = tuple(entries)
    unread_entries = tuple(unread)
    if not resolved and not unread_entries:
        return scan
    base = Path(scan.root).resolve()
    trees = tuple(entry for entry in resolved if not entry.path.is_relative_to(base))
    covered = tuple(entry for entry in resolved if entry.path.is_relative_to(base))
    read = [_scan_entry(tree, alphabet, triggers) for tree in trees]
    covered_modules = tuple(
        name
        for entry in covered
        for name in source_top_level_names(ImportPathEntry(str(entry.path), entry.module))
    )
    return SpawnScan(
        root=scan.root,
        files_read=scan.files_read + sum(tree.files_read for tree in read),
        modules_read=tuple(
            sorted(
                set(scan.modules_read).union(covered_modules, *(tree.modules_read for tree in read))
            )
        ),
        reaches=scan.reaches
        + tuple(
            ModuleReach(reach.path, reach.primitives, container=tree.root)
            for tree in read
            for reach in tree.reaches
        ),
        archives=scan.archives,
        unread_archived=scan.unread_archived,
        sites=scan.sites + tuple(str(entry.path) for entry in resolved),
        unread_path_entries=scan.unread_path_entries + unread_entries,
        path_entries=scan.path_entries
        + tuple(ImportPathEntry(str(entry.path), entry.module) for entry in resolved),
    )


def _scan_entry(
    entry: SitePathEntry,
    alphabet: Mapping[str, frozenset[str]],
    triggers: Sequence[bytes],
) -> SpawnScan:
    """Read a literal path root or precisely one package/module exposed by a finder mapping."""
    if not entry.module:
        return spawn_scan(entry.path, alphabet, triggers)
    if entry.path.is_dir():
        scanned = spawn_scan(entry.path, alphabet, triggers)
        prefix = entry.module.replace(".", "/")
        return SpawnScan(
            root=str(entry.path),
            files_read=scanned.files_read,
            modules_read=(entry.module.split(".")[0],) if scanned.files_read else (),
            reaches=tuple(
                ModuleReach(f"{prefix}/{reach.path}", reach.primitives) for reach in scanned.reaches
            ),
        )
    return _scan_mapped_module(entry, alphabet, triggers)


def _scan_mapped_module(
    entry: SitePathEntry,
    alphabet: Mapping[str, frozenset[str]],
    triggers: Sequence[bytes],
) -> SpawnScan:
    try:
        source = entry.path.read_bytes()
    except OSError:
        return SpawnScan(str(entry.path), 0, (), ())
    primitives = (
        source_reaches(source, alphabet) if any(item in source for item in triggers) else ()
    )
    relative = f"{entry.module.replace('.', '/')}.py"
    reaches = (ModuleReach(relative, primitives),) if primitives else ()
    return SpawnScan(str(entry.path), 1, (entry.module.split(".")[0],), reaches)


def path_lines(pth: Path) -> tuple[str, ...]:
    """The lines of one `.pth` that name a path, by the rule the interpreter reads it with.

    A comment and a blank line name nothing, a line starting with `import ` or `import\t` is code
    the interpreter runs rather than a path, and a directory whose NAME begins with `import` is
    still a path -- which is why the match is on the trailing space and not on the word.

    A file that cannot be read contributes nothing, the same tolerance the directory scan gives an
    unreadable source file: an unreadable `.pth` is not evidence about what is importable.
    """
    try:
        text = pth.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    return tuple(
        stripped
        for line in text.splitlines()
        if (stripped := line.rstrip())
        and not stripped.startswith("#")
        and not line.startswith(_EXECUTED_PREFIXES)
    )


def executed_lines(pth: Path) -> tuple[tuple[int, str], ...]:
    """Executable `.pth` lines, retaining their line number so an unread one is actionable."""
    try:
        lines = pth.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    return tuple(
        (number, line.rstrip())
        for number, line in enumerate(lines, 1)
        if line.startswith(_EXECUTED_PREFIXES)
    )
