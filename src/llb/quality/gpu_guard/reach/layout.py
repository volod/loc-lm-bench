"""What the interpreter's own on-disk layout says is importable, read off the files themselves.

The stdlib coverage measurement needs three filesystem facts and no imports to get them: which
module names ship as a shared object, which ship as a `.pyc` with no source beside it, and which
source file one `__pycache__` entry actually claims. All three are name rules over paths, kept here
so `gpu_guard_spawn_reach_coverage` reads as the classification and the report it produces.
"""

import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

from llb.quality.gpu_guard.reach.scan import is_excluded
from llb.quality.gpu_guard.reach.archive import PACKAGE_INIT

# What this interpreter writes between a cached module's stem and its `.pyc` -- the one part of a
# cache name that is neither the module nor the suffix, and therefore the only place to split it.
_CACHE_TAG = sys.implementation.cache_tag

# Where a compiled module can sit under the stdlib root without a `.py` beside it: a package's
# cached `__init__`, a top-level cached module, and the two flat layouts a stripped install uses.
_COMPILED_PATTERNS = (
    "*/__pycache__/__init__.*.pyc",
    "__pycache__/*.pyc",
    "*/__init__.pyc",
    "*.pyc",
)


def compiled_only_submodules(root: Path) -> tuple[str, ...]:
    """Submodules inside a package the scan walked that are cached with no source beside them.

    The evidence the interpreter leaves on disk, in place of the per-submodule list CPython does not
    publish: a `__pycache__` entry whose source file is not beside the package is a module this host
    can import and the scan never parsed. A `.py` with no `.pyc` is nothing -- caching is incidental
    -- and a name that is in neither is simply not there, which is the same evidence-based decision
    the `compiled_only` / `absent` split makes at the name level.

    Which source file an entry claims is `cached_source`, the interpreter's own rule rather than a
    stem comparison, because a package can ship `v3.0.0.a.py` and a stem is not the text before the
    first dot.
    """
    names: set[str] = set()
    for path in root.rglob("__pycache__/*.pyc"):
        relative = path.relative_to(root)
        # `<package>/.../__pycache__/<stem>.<tag>.pyc`, so fewer than three parts is a TOP-LEVEL
        # module's cache, which `_compiled_stems` already classifies as a declared name.
        if len(relative.parts) < 3 or is_excluded(relative.as_posix()):
            continue
        source = cached_source(path)
        # A cached `__init__` names its PACKAGE, which the declared list already carries.
        if source.stem == PACKAGE_INIT or source.is_file():
            continue
        names.add(".".join((*source.relative_to(root).parts[:-1], source.stem)))
    return tuple(sorted(names))


def cached_source(path: Path) -> Path:
    """The source file one `__pycache__` entry claims -- `v3.0.0.a.cpython-313.pyc` -> `v3.0.0.a.py`.

    PEP 3147 names a cache `<stem>.<tag>.pyc`, and neither half of that is a dot away from the
    other: `optuna` ships alembic revisions as `v3.0.0.a.py`, so the stem is not the text before the
    first dot, and pytest writes its rewritten caches under `cpython-313-pytest-9.1`, so the tag is
    not one dot-separated component either. Read against the running interpreter's own
    `cache_tag`, both come out right -- as does the `.opt-1` an optimized cache appends.
    (`importlib.util.source_from_cache` answers neither: it refuses any name with more than three
    dots.) Two layouts carry no tag to split on -- a cache written by another interpreter version,
    and the tagless `pkg/__pycache__/util.pyc` -- and are read on the PEP's shape instead, which is
    what puts a stale `util.cpython-311.pyc` back on the `util.py` sitting beside it.

    A cached module is measured against the source it claims and never against the source's own
    existence elsewhere, so this stays a pure name rule; whether the file is there is the caller's
    question.
    """
    head, tagged, _ = path.name.partition(f".{_CACHE_TAG}") if _CACHE_TAG else ("", "", "")
    parts = path.name.rsplit(".", 2)
    stem = head if tagged else (parts[0] if len(parts) == 3 else path.name.removesuffix(".pyc"))
    return path.parent.parent / f"{stem}.py"


def extension_stems(root: Path) -> frozenset[str]:
    """Module names shipped as a shared object, read off the filenames rather than imported."""
    suffixes = tuple(EXTENSION_SUFFIXES)
    return frozenset(
        path.name.split(".")[0]
        for directory in (root, root / "lib-dynload")
        if directory.is_dir()
        for path in directory.iterdir()
        if path.name.endswith(suffixes)
    )


def compiled_stems(root: Path) -> frozenset[str]:
    """Module names with a `.pyc` under the root -- importable whether or not source is there."""
    return frozenset(
        _compiled_stem(root, path) for pattern in _COMPILED_PATTERNS for path in root.glob(pattern)
    )


def _compiled_stem(root: Path, path: Path) -> str:
    """A cached `__init__` names its package; anything else names itself."""
    if path.name.split(".")[0] == "__init__":
        return path.relative_to(root).parts[0]
    return path.name.split(".")[0]
