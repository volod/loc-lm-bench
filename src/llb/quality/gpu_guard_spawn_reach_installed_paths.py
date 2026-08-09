"""Filesystem evidence for every directory or finder mapping on the installed import path."""

from collections.abc import Iterable, Mapping
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

from llb.quality.gpu_guard_spawn_reach import ImportPathEntry, SpawnScan, is_excluded
from llb.quality.gpu_guard_spawn_reach_coverage import cached_source

_EXTENSION_GLOBS = tuple(
    f"*{suffix}"
    for suffix in EXTENSION_SUFFIXES
    if not any(suffix.endswith(other) for other in EXTENSION_SUFFIXES if other != suffix)
)
_KIND_PRIORITY = {"compiled_only": 0, "extensions": 1, "namespace": 2, "absent": 3}


def scan_path_entries(scan: SpawnScan) -> tuple[ImportPathEntry, ...]:
    """The root plus every resolved `.pth` entry, retaining finder-mapped names."""
    recorded = scan.path_entries or tuple(ImportPathEntry(path) for path in scan.sites)
    entries = (ImportPathEntry(scan.root), *recorded)
    return tuple(dict.fromkeys(entries))


def provided_top_level_names(
    entries: Iterable[ImportPathEntry],
) -> Mapping[str, tuple[ImportPathEntry, ...]]:
    """Every import name the filesystem entries expose, without importing any of them."""
    providers: dict[str, list[ImportPathEntry]] = {}
    for entry in entries:
        names = (
            (entry.module.split(".")[0],) if entry.module else _directory_names(Path(entry.path))
        )
        for name in names:
            providers.setdefault(name, []).append(entry)
    return {name: tuple(paths) for name, paths in providers.items()}


def classify_name(
    name: str,
    providers: Iterable[ImportPathEntry],
    archived: frozenset[str],
) -> str:
    """Why a provided or metadata-declared name yielded no source to the scan."""
    entries = tuple(providers)
    if not entries:
        return "archived" if name in archived else "absent"
    kinds = tuple(_entry_kind(name, entry) for entry in entries)
    return min(kinds, key=_KIND_PRIORITY.__getitem__)


def declared_name_providers(
    name: str, entries: Iterable[ImportPathEntry]
) -> tuple[ImportPathEntry, ...]:
    """Entries carrying a metadata-declared name, including non-identifier stub/data names."""
    return tuple(
        entry
        for entry in entries
        if (
            entry.module.split(".")[0] == name
            if entry.module
            else _direct_kind(name, Path(entry.path)) != "absent"
        )
    )


def source_top_level_names(entry: ImportPathEntry) -> tuple[str, ...]:
    """Import names whose source the root walk already parsed beneath an in-root entry."""
    target = Path(entry.path)
    if entry.module:
        paths = (target,) if target.is_file() else target.rglob("*.py")
        return (
            (entry.module.split(".")[0],)
            if any(_included_source(path, target) for path in paths)
            else ()
        )
    return tuple(
        sorted(
            {
                path.relative_to(target).parts[0].removesuffix(".py")
                for path in target.rglob("*.py")
                if _included_source(path, target)
            }
        )
    )


def _directory_names(root: Path) -> tuple[str, ...]:
    try:
        children = tuple(root.iterdir())
    except OSError:
        return ()
    names = {
        child.name
        for child in children
        if child.is_dir() and child.name != "__pycache__" and child.name.isidentifier()
    }
    names.update(
        name for child in children if child.is_file() and (name := _module_name(child.name))
    )
    cache = root / "__pycache__"
    if cache.is_dir():
        names.update(
            cached_source(path).stem
            for path in cache.glob("*.pyc")
            if cached_source(path).stem.isidentifier()
        )
    return tuple(sorted(names))


def _module_name(filename: str) -> str:
    suffixes = (".py", ".pyc", *EXTENSION_SUFFIXES)
    name = next(
        (filename.removesuffix(suffix) for suffix in suffixes if filename.endswith(suffix)), ""
    )
    return name if name.isidentifier() else ""


def _entry_kind(name: str, entry: ImportPathEntry) -> str:
    if entry.module:
        target = Path(entry.path)
        if target.is_dir():
            return _directory_kind(target, target.parent)
        if target.suffix == ".pyc":
            return "compiled_only"
        if any(target.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
            return "extensions"
        return "namespace"
    return _direct_kind(name, Path(entry.path))


def _direct_kind(name: str, root: Path) -> str:
    if any((root / f"{name}{suffix}").is_file() for suffix in EXTENSION_SUFFIXES):
        return "extensions"
    directory = root / name
    if directory.is_dir():
        return _directory_kind(directory, root)
    if _cached_module(root, name):
        return "compiled_only"
    return "absent"


def _directory_kind(directory: Path, root: Path) -> str:
    if _holds(directory, root, "*.pyc"):
        return "compiled_only"
    return "extensions" if _holds(directory, root, *_EXTENSION_GLOBS) else "namespace"


def _holds(directory: Path, root: Path, *globs: str) -> bool:
    return any(
        not is_excluded(path.relative_to(root).as_posix())
        for glob in globs
        for path in directory.rglob(glob)
    )


def _cached_module(root: Path, name: str) -> bool:
    return (root / f"{name}.pyc").is_file() or any((root / "__pycache__").glob(f"{name}.*.pyc"))


def _included_source(path: Path, root: Path) -> bool:
    return path.is_file() and not is_excluded(path.relative_to(root).as_posix())
