"""What the INSTALLED scan failed to read -- so the venv verdict names the venv it was read from.

The stdlib half weighs its reading against `sys.stdlib_module_names` and accounts for every declared
name it read no source for (`llb.quality.gpu_guard_spawn_reach_coverage`). The installed half had
nothing of the kind: its only guard was the degenerate end -- an empty read, plus a `files_read`
assertion -- which is exactly the check the stdlib half outgrew, because a file count says how much
was read and not what was missed. A dependency installed with its sources stripped is parsed by
nothing and reported by nothing, and "no dependency goes below the seams" then covers a venv part of
which was never opened: the directory-tree twin of the archive case
`llb.quality.gpu_guard_spawn_reach_installed_archive` closed.

`importlib.metadata` publishes the list the stdlib gets from the interpreter.
`packages_distributions()` maps every importable top-level name an installed distribution provides
to the distributions providing it, which is the same unit the scan reports (`modules_read`) and the
same unit an excuse is declared at. Every published name the pass read no source for is classified,
and -- as one tree over -- the classification is the deliverable, because most of what is left has
no `.py` by construction:

- `extensions` -- the name resolves to a shared object rather than to source: an extension module
  installed under the name itself (`_duckdb`, `mmh3`, the mypyc-compiled `..._mypyc` objects), or a
  directory that ships shared objects and no Python at all (`librt`, and the `nvidia-*` wheels,
  whose payload is CUDA libraries). Reported, never refused -- this is the case a naive gate breaks
  on, and there is nothing to parse either way.
- `namespace` -- a directory with no module of its own: an implicit namespace package (`nvidia`
  before its libraries are counted), a PEP 561 stub directory (`wrapt-stubs`), or a data directory a
  distribution plants at the top level (`include`, `schemas`). Importing it yields a namespace
  package with no code in it, so it starts no child.
- `compiled_only` -- a cached module with no source beside it. This is the one REFUSED by
  `gpu_guard_spawn_reach_installed_audit.audit_installed_read_coverage`: the stripped tree this
  measurement exists for, where the module is importable on this host and the scan did not read it.
- `archived` -- nothing in the tree, and an archive on the import path carries it. Classified so the
  partition stays whole and left to `unread_archived_packages`, which already refuses it per package
  -- reporting it twice would be one finding wearing two names.
- `absent` -- nothing the pass read provides it. The scan reads the whole import path now (the tree,
  its archives, and the extra directories a `.pth` adds), so this is an answer rather than an
  artifact of reading one root: what is left is a distribution recording a SUBMODULE as a top-level
  name, which `tree-sitter-*` (`_binding`) and `xxhash` (`_xxhash`) both do here. Reported, never
  refused -- a name nothing provides cannot start a child, and refusing two third-party metadata
  quirks would be the naive gate again.

Measured over this repo's venv (CPython 3.13, 35636 files read over site-packages plus the editable
`<repo>/src` its `.pth` adds): of 421 published top-level names, **403 read as source,
10 extensions, 6 namespace, 0 compiled-only, 0 archived, 2 absent** -- every unread name accounted
for, which is what makes the below-the-seams verdict a statement about the venv rather than about
whichever installed files happened to carry source.

One level down is the same problem the stdlib half has and the same answer: the published list is
per TOP-LEVEL name, so a package that ships its `__init__.py` and strips `pkg/util.py` classifies as
read. `compiled_only_submodules` is reused unchanged from the stdlib coverage -- it needs no
published list, because the interpreter leaves the evidence in the package directory.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import packages_distributions
from pathlib import Path

from llb.quality.gpu_guard_spawn_reach import SpawnScan, is_excluded
from llb.quality.gpu_guard_spawn_reach_coverage import (
    class_counts,
    compiled_only_submodules,
    named_list,
)

# The classification, in the order a name is tried against it -- also the fields of the record
# below, so the buckets and the report cannot drift apart.
_FIELDS = ("read", "extensions", "namespace", "compiled_only", "archived", "absent")

# What an extension module is named on this platform, reduced to the shortest suffixes that still
# match all of them: `.abi3.so` and `.cpython-313-x86_64-linux-gnu.so` both end in `.so`, so one
# glob answers for the family and a directory is walked once instead of three times.
_EXTENSION_GLOBS = tuple(
    f"*{suffix}"
    for suffix in EXTENSION_SUFFIXES
    if not any(suffix.endswith(other) for other in EXTENSION_SUFFIXES if other != suffix)
)


@dataclass(frozen=True)
class InstalledReadCoverage:
    """How much of this venv's published import surface the scan read, and why the rest not.

    The six name fields are a partition of what `packages_distributions()` publishes: every name
    lands in exactly one, so the counts add up to the list and a name cannot go missing between two.
    `compiled_only_submodules` sits outside it deliberately, as it does for the stdlib -- its
    entries are dotted names the published list does not contain.
    """

    root: str
    read: tuple[str, ...]
    extensions: tuple[str, ...]
    namespace: tuple[str, ...]
    compiled_only: tuple[str, ...]
    archived: tuple[str, ...]
    absent: tuple[str, ...]
    compiled_only_submodules: tuple[str, ...] = ()
    archives: tuple[str, ...] = ()
    sites: tuple[str, ...] = ()

    @property
    def unread(self) -> tuple[str, ...]:
        """Every published name no source was read for, whatever the reason."""
        return tuple(
            sorted(
                self.extensions + self.namespace + self.compiled_only + self.archived + self.absent
            )
        )


def installed_read_coverage(
    scan: SpawnScan, names: Iterable[str] | None = None
) -> InstalledReadCoverage:
    """Weigh what one installed scan read against what this environment says it can import."""
    root = Path(scan.root)
    published = sorted(names) if names is not None else list(importable_top_level_names())
    read = frozenset(scan.modules_read)
    archived = frozenset(name.split(".")[0] for name in scan.unread_archived)
    buckets: dict[str, list[str]] = {}
    for name in published:
        kind = "read" if name in read else _kind(name, root, archived)
        buckets.setdefault(kind, []).append(name)
    return InstalledReadCoverage(
        root=scan.root,
        archives=scan.archives,
        sites=scan.sites,
        compiled_only_submodules=compiled_only_submodules(root),
        **{field: tuple(buckets.get(field, ())) for field in _FIELDS},
    )


def importable_top_level_names(
    published: Mapping[str, list[str]] | None = None,
) -> tuple[str, ...]:
    """Every top-level name the installed distributions publish, as an `import` would spell it.

    `packages_distributions()` reads each distribution's own record of what it installed, and a few
    write a PATH there rather than a name -- `nvidia/cusparselt`, `sentencepiece/__init__` on this
    host. The first segment is the top-level name in each such case, which is both what the scan
    reports and what an excuse is declared at, so the two lists are comparable at all.
    """
    provided = published if published is not None else packages_distributions()
    return tuple(sorted({name.split("/")[0] for name in provided}))


def installed_read_coverage_message(coverage: InstalledReadCoverage) -> str:
    """The operator-facing line: what was read, and what each unread name is excused by."""
    submodules = coverage.compiled_only_submodules
    return (
        f"[gpu-guard] installed read coverage under {coverage.root}: {len(coverage.read)} of "
        f"{len(coverage.read) + len(coverage.unread)} published top-level names read as source; "
        f"{class_counts(coverage, _FIELDS)}; {len(submodules)} compiled-only submodules"
        f"{named_list('compiled-only', coverage.compiled_only)}"
        f"{named_list('compiled-only submodules', submodules)}"
        f"{named_list('namespace', coverage.namespace)}"
        f"{named_list('archived', coverage.archived)}{named_list('absent', coverage.absent)}"
        f"{named_list('archives read', coverage.archives)}"
        f"{named_list('path entries read', coverage.sites)}"
    )


def _kind(name: str, root: Path, archived: frozenset[str]) -> str:
    """Why one published name was not read -- the first construction that accounts for it.

    A shared object installed under the name itself is read first, because that IS the module: a
    directory of the same name beside it is whatever the distribution shipped along (`mmh3` is an
    extension module at the root and a directory of C sources and a stub). Inside a directory, cache
    evidence comes before shared objects -- a `.pyc` is the interpreter's own record that source WAS
    there, so a tree that also ships an object is a stripped tree and not a pure-extension one. The
    scan reads every `.py` it is not told to skip, so a name that reaches here has no readable
    source by construction and none is looked for.
    """
    if any((root / f"{name}{suffix}").is_file() for suffix in EXTENSION_SUFFIXES):
        return "extensions"
    directory = root / name
    if directory.is_dir():
        if _holds(directory, root, "*.pyc"):
            return "compiled_only"
        return "extensions" if _holds(directory, root, *_EXTENSION_GLOBS) else "namespace"
    if _cached_module(root, name):
        return "compiled_only"
    return "archived" if name in archived else "absent"


def _holds(directory: Path, root: Path, *globs: str) -> bool:
    """Whether the directory carries a file matching any glob, in a place the scan would have read.

    The scan's own exclusion rule applies, so a vendored `tests/` tree whose caches outlived its
    sources is not read as a stripped dependency -- the two halves cannot disagree about which
    directories the statement covers.
    """
    return any(
        not is_excluded(path.relative_to(root).as_posix())
        for glob in globs
        for path in directory.rglob(glob)
    )


def _cached_module(root: Path, name: str) -> bool:
    """Whether a top-level name ships as a cached module -- either cache layout, no source."""
    return (root / f"{name}.pyc").is_file() or any((root / "__pycache__").glob(f"{name}.*.pyc"))
