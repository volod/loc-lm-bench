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

Two kinds of entry are deliberately left alone:

- An entry INSIDE the scan root. `nvidia-cutlass-dsl` ships one (`nvidia_cutlass_dsl/python_packages`,
  which makes `cutlass` importable), and the directory pass has already read every file in it.
  Reading it again would count those files twice and report one file under two package names --
  once as `cutlass` and once as the `nvidia_cutlass_dsl` its distribution actually publishes, which
  is the name an excuse would be written at.
- A `.pth` that adds its paths by RUNNING code -- the `import __editable___pkg_finder` style
  setuptools uses for a flat layout, where the paths live in a dict inside the finder module. Its
  tree is not read, so whatever it provides stays unread and is reported as such rather than
  silently counted.

Measured over this host: two entries, one of which is under the root, so ONE tree is scanned --
`<repo>/src`, 931 files in 0.04s, and **no reach below the seams at all**. That is the answer to
whether this repo's own source needs a declaration like a dependency's: it starts children in 15
modules and every one of them goes through `subprocess.run` / `subprocess.call` /
`subprocess.Popen`, which the denial patches, so it is held to exactly the question a dependency is
held to and needs no excuse to pass it.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from llb.quality.gpu_guard_spawn_reach import ModuleReach, SpawnScan, spawn_scan

# What `site.addpackage` executes rather than resolves. Matched exactly as the interpreter matches
# it, so a module called `imports.py` on a path line stays a path.
_EXECUTED_PREFIXES = ("import ", "import\t")

PTH_SUFFIX = "*.pth"


def site_path_entries(root: Path) -> tuple[Path, ...]:
    """The directories the `.pth` files under one root add to the import path.

    Only the entries this scan can say something new about: a path that exists, is a directory, and
    is not the root or a tree inside it that the root's own walk already read. Resolved, because two
    `.pth` files naming one tree by different routes are one tree to read.
    """
    base = root.resolve()
    found: list[Path] = []
    for pth in sorted(root.glob(PTH_SUFFIX)):
        for line in path_lines(pth):
            entry = (root / line).resolve()
            if entry not in found and entry.is_dir() and not entry.is_relative_to(base):
                found.append(entry)
    return tuple(found)


def with_path_entries(
    scan: SpawnScan,
    entries: Iterable[Path],
    alphabet: Mapping[str, frozenset[str]],
    triggers: Sequence[bytes],
) -> SpawnScan:
    """Fold what the extra trees contributed into the directory pass, as one scan of one path.

    Each tree is read exactly as the root is -- same alphabet, same prefilter, same exclusions --
    and its reaches carry the tree as their `container`, so a finding names the file an operator has
    to open rather than a path that looks like it is under site-packages and is not.
    """
    trees = tuple(entries)
    if not trees:
        return scan
    read = [spawn_scan(tree, alphabet, triggers) for tree in trees]
    return SpawnScan(
        root=scan.root,
        files_read=scan.files_read + sum(tree.files_read for tree in read),
        modules_read=tuple(
            sorted(set(scan.modules_read).union(*(tree.modules_read for tree in read)))
        ),
        reaches=scan.reaches
        + tuple(
            ModuleReach(reach.path, reach.primitives, container=tree.root)
            for tree in read
            for reach in tree.reaches
        ),
        archives=scan.archives,
        unread_archived=scan.unread_archived,
        sites=tuple(str(tree) for tree in trees),
    )


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
