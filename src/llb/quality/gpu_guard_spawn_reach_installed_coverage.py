"""What the INSTALLED scan failed to read -- so the venv verdict names the venv it was read from.

The stdlib half weighs its reading against `sys.stdlib_module_names` and accounts for every declared
name it read no source for (`llb.quality.gpu_guard_spawn_reach_coverage`). The installed half had
nothing of the kind: its only guard was the degenerate end -- an empty read, plus a `files_read`
assertion -- which is exactly the check the stdlib half outgrew, because a file count says how much
was read and not what was missed. A dependency installed with its sources stripped is parsed by
nothing and reported by nothing, and "no dependency goes below the seams" then covers a venv part of
which was never opened: the directory-tree twin of the archive case
`llb.quality.gpu_guard_spawn_reach_installed_archive` closed.

`importlib.metadata` publishes most of the list the stdlib gets from the interpreter.
`packages_distributions()` maps top-level names to the distributions providing them, but a `.pth`
entry can expose a name no distribution records. The declared surface is therefore the union of
metadata and the names each resolved filesystem entry provides. Each unread name is classified
against its provider entry rather than only against site-packages, so a stripped editable tree is
the same refusal wherever it sits. The classification remains the deliverable because most of what
is left has no `.py` by construction:

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

Measured over this repo's venv (CPython 3.13, 35638 files read over site-packages plus the resolved
`.pth` entries): of 424 provided or metadata-published top-level names, **406 read as source,
10 extensions, 6 namespace, 0 compiled-only, 0 archived, 2 absent** -- every unread name accounted
for, which is what makes the below-the-seams verdict a statement about the venv rather than about
whichever installed files happened to carry source. The three names missing from distribution
metadata are `OleFileIO_PL`, `_virtualenv`, and `cutlass`; each is accounted for as read.

One level down is the same problem the stdlib half has and the same answer: the published list is
per TOP-LEVEL name, so a package that ships its `__init__.py` and strips `pkg/util.py` classifies as
read. `compiled_only_submodules` is reused unchanged from the stdlib coverage -- it needs no
published list, because the interpreter leaves the evidence in the package directory.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import packages_distributions
from pathlib import Path

from llb.quality.gpu_guard_spawn_reach import SpawnScan
from llb.quality.gpu_guard_spawn_reach_coverage import (
    class_counts,
    compiled_only_submodules,
    named_list,
)
from llb.quality.gpu_guard_spawn_reach_installed_paths import (
    classify_name,
    declared_name_providers,
    provided_top_level_names,
    scan_path_entries,
)

# The classification, in the order a name is tried against it -- also the fields of the record
# below, so the buckets and the report cannot drift apart.
_FIELDS = ("read", "extensions", "namespace", "compiled_only", "archived", "absent")


@dataclass(frozen=True)
class InstalledReadCoverage:
    """How much of this venv's published import surface the scan read, and why the rest not.

    The six name fields partition metadata names plus names the resolved path entries provide:
    every name lands in exactly one, so a name cannot disappear merely because no distribution
    records it. `compiled_only_submodules` sits outside deliberately, as it does for the stdlib --
    its entries are dotted names the top-level list does not contain.
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
        """Every declared name no source was read for, whatever the reason."""
        return tuple(
            sorted(
                self.extensions + self.namespace + self.compiled_only + self.archived + self.absent
            )
        )


def installed_read_coverage(
    scan: SpawnScan, names: Iterable[str] | None = None
) -> InstalledReadCoverage:
    """Weigh a scan against metadata plus names its resolved path entries actually provide."""
    entries = scan_path_entries(scan)
    providers = provided_top_level_names(entries)
    metadata_names = set(names) if names is not None else set(importable_top_level_names())
    published = sorted(metadata_names | set(providers))
    read = frozenset(scan.modules_read)
    archived = frozenset(name.split(".")[0] for name in scan.unread_archived)
    buckets: dict[str, list[str]] = {}
    for name in published:
        name_providers = providers.get(name) or declared_name_providers(name, entries)
        kind = "read" if name in read else classify_name(name, name_providers, archived)
        buckets.setdefault(kind, []).append(name)
    return InstalledReadCoverage(
        root=scan.root,
        archives=scan.archives,
        sites=scan.sites,
        compiled_only_submodules=compiled_only_submodules(Path(scan.root)),
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
        f"{len(coverage.read) + len(coverage.unread)} provided/metadata top-level names read as "
        "source; "
        f"{class_counts(coverage, _FIELDS)}; {len(submodules)} compiled-only submodules"
        f"{named_list('compiled-only', coverage.compiled_only)}"
        f"{named_list('compiled-only submodules', submodules)}"
        f"{named_list('namespace', coverage.namespace)}"
        f"{named_list('archived', coverage.archived)}{named_list('absent', coverage.absent)}"
        f"{named_list('archives read', coverage.archives)}"
        f"{named_list('path entries read', coverage.sites)}"
    )
