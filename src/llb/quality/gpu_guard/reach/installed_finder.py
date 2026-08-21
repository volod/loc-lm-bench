"""Static reading of setuptools' executable editable-install finder.

Setuptools can put `import finder; finder.install()` in a `.pth` file and keep the exposed package
roots in the generated module's `MAPPING`. Importing either file would execute host code while a
quality check is trying to describe it, so this reader accepts only that exact installer statement
and an `ast.literal_eval`-compatible mapping assignment.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

_FINDER_PREFIX = "__editable___"
_FINDER_SUFFIX = "_finder"


@dataclass(frozen=True)
class FinderPath:
    """One package or module name and the source target the editable finder exposes for it."""

    module: str
    path: Path


def finder_paths(root: Path, line: str) -> tuple[FinderPath, ...] | None:
    """Resolve a generated installer line, or None when it is not the known static shape."""
    module = _finder_module(line)
    if module is None:
        return None
    finder = root.joinpath(*module.split(".")).with_suffix(".py")
    mapping = _literal_mapping(finder)
    if mapping is None:
        return None
    entries = []
    for name, raw_path in mapping.items():
        path = Path(raw_path)
        path = path.resolve() if path.is_absolute() else (finder.parent / path).resolve()
        readable = _source_target(path)
        if readable is not None:
            entries.append(FinderPath(name, readable))
    return tuple(entries)


def _finder_module(line: str) -> str | None:
    try:
        parsed = ast.parse(line)
    except SyntaxError:
        return None
    if len(parsed.body) != 2 or not isinstance(parsed.body[0], ast.Import):
        return None
    imported = parsed.body[0].names
    call_stmt = parsed.body[1]
    if len(imported) != 1 or not isinstance(call_stmt, ast.Expr):
        return None
    alias = imported[0]
    if not alias.name.startswith(_FINDER_PREFIX) or not alias.name.endswith(_FINDER_SUFFIX):
        return None
    call = call_stmt.value
    bound = alias.asname or alias.name.split(".")[0]
    if not isinstance(call, ast.Call) or call.args or call.keywords:
        return None
    function = call.func
    if not isinstance(function, ast.Attribute) or function.attr != "install":
        return None
    if not isinstance(function.value, ast.Name) or function.value.id != bound:
        return None
    return alias.name


def _literal_mapping(path: Path) -> dict[str, str] | None:
    try:
        parsed = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None
    assignments = tuple(
        value for statement in parsed.body if (value := _mapping_value(statement)) is not None
    )
    return _literal_dict(assignments[0]) if len(assignments) == 1 else None


def _mapping_value(statement: ast.stmt) -> ast.expr | None:
    if isinstance(statement, ast.Assign):
        named = any(
            isinstance(target, ast.Name) and target.id == "MAPPING" for target in statement.targets
        )
        return statement.value if named else None
    if isinstance(statement, ast.AnnAssign):
        named = isinstance(statement.target, ast.Name) and statement.target.id == "MAPPING"
        return statement.value if named else None
    return None


def _literal_dict(value: ast.expr) -> dict[str, str] | None:
    try:
        mapping = ast.literal_eval(value)
    except (ValueError, TypeError):
        return None
    if not isinstance(mapping, dict) or not all(
        isinstance(name, str) and isinstance(raw, str) for name, raw in mapping.items()
    ):
        return None
    return mapping


def _source_target(path: Path) -> Path | None:
    """The package directory or source module a finder target actually makes importable."""
    if path.is_dir() or (path.is_file() and path.suffix == ".py"):
        return path
    source = path.with_suffix(".py")
    return source if source.is_file() else None
