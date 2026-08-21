"""What one module's source resolves when it starts a child.

The half of the reach scan that reads a single file: given a source buffer and an alphabet of
`module -> process-starting attributes`, it returns the names that file would actually call. The
resolution goes through the module's OWN imports, which is what separates a call from a coincidence
-- `os.fork()` counts, `operating.fork()` counts when `operating` is `import os as operating`, and a
module's own `def fork()` does not count at all.

What it cannot see is stated rather than papered over: a dynamic import (`importlib.import_module`),
a call through an object attribute rather than a module name, and anything a compiled extension does
below Python. The scan reads source, so those are residuals of the reading, not of the declaration.
"""

import ast
import warnings
from collections.abc import Mapping


def source_reaches(source: bytes, alphabet: Mapping[str, frozenset[str]]) -> tuple[str, ...]:
    """The process-starting names one module's source resolves, through its own imports."""
    try:
        with warnings.catch_warnings():
            # Third-party source that compiles with a SyntaxWarning (an invalid escape in a
            # docstring, say) is the package's business, not a finding of this scan.
            warnings.simplefilter("ignore")
            parsed = ast.parse(source)
    except (SyntaxError, ValueError):
        return ()
    modules, names, calls = _imports_and_calls(parsed, alphabet)
    reached = {_call_label(call, modules, names, alphabet) for call in calls}
    return tuple(sorted(label for label in reached if label is not None))


def _imports_and_calls(
    parsed: ast.AST, alphabet: Mapping[str, frozenset[str]]
) -> tuple[dict[str, str], dict[str, str], list[ast.expr]]:
    """One walk: local name -> module, local name -> label, and every call target in the module."""
    modules: dict[str, str] = {}
    names: dict[str, str] = {}
    calls: list[ast.expr] = []
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            calls.append(node.func)
        elif isinstance(node, ast.Import):
            modules.update(_module_aliases(node, alphabet))
        elif isinstance(node, ast.ImportFrom):
            names.update(_name_aliases(node, alphabet))
    return modules, names, calls


def _module_aliases(node: ast.Import, alphabet: Mapping[str, frozenset[str]]) -> dict[str, str]:
    """`import os as operating` -> `{"operating": "os"}`, for the modules the alphabet names."""
    return {
        (alias.asname or alias.name): alias.name for alias in node.names if alias.name in alphabet
    }


def _name_aliases(node: ast.ImportFrom, alphabet: Mapping[str, frozenset[str]]) -> dict[str, str]:
    """`from subprocess import Popen as Runner` -> `{"Runner": "subprocess.Popen"}`."""
    module = node.module
    if module is None or module not in alphabet:
        return {}
    return {
        (alias.asname or alias.name): f"{module}.{alias.name}"
        for alias in node.names
        if alias.name in alphabet[module]
    }


def _call_label(
    call: ast.expr,
    modules: Mapping[str, str],
    names: Mapping[str, str],
    alphabet: Mapping[str, frozenset[str]],
) -> str | None:
    """`os.fork(...)` / `fork(...)` resolved back to the declared name it calls, or None."""
    if isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name):
        module = modules.get(call.value.id)
        if module is not None and call.attr in alphabet[module]:
            return f"{module}.{call.attr}"
        return None
    if isinstance(call, ast.Name):
        return names.get(call.id)
    return None
