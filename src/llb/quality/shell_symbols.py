"""Every `llb_*` helper a caller names is defined in that file or in one it declares it sources.

The shell lint follows a sourced file (`shellcheck -x -P SCRIPTDIR`), but shellcheck resolves
VARIABLES across that boundary, not FUNCTIONS: a call to a helper that no longer exists passes every
scan and fails as `command not found` on an operator's host, at whatever point of the script the
call sits. The shell layer here is exactly the shape that invites it -- one prefix, definitions in
`scripts/shared/`, call sites in a dozen entrypoints and in the `make` recipes that source them.

Scope is what the caller DECLARES, not what happens to be loaded at run time: a `# shellcheck
source=` directive (the shell gate already checks those resolve), a make recipe's literal
`$(PROJECT_ROOT)/...` source, or an explicit `# llb-requires:` line for a shared module whose
functions assume a sibling was sourced by whoever sourced it. A call built by `eval` or through a
variable is out of reach and stays out of scope.
"""

import argparse
import logging
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from llb.core.paths import PROJECT_ROOT

_LOG = logging.getLogger(__name__)

_DEFINITION = re.compile(r"^\s*(?:function\s+)?(llb_[A-Za-z0-9_]+)\s*\(\s*\)")
# A name in command or argument position. The lookbehind drops the shapes that only READ like a
# call: an expansion operand (`${name#llb_prefix}`, `${x:-llb_default}`), a variable (`$llb_x`), a
# path segment, and an assigned string -- none of which stops working when a function is renamed.
_REFERENCE = re.compile(r"(?<![\w#%$/=:.-])llb_[A-Za-z0-9_]+")
_DIRECTIVE = re.compile(r"#\s*shellcheck\s+source=(\S+)")
_REQUIRES = re.compile(r"#\s*llb-requires:\s*(\S+)")
_MAKE_SOURCE = re.compile(r"(?:source|\.)\s+\"?\$\(PROJECT_ROOT\)/([^\"\s;]+)")
_GLOBS = ("*.sh", "Makefile", "*.mk")


def listed_callers(project_root: Path) -> list[Path]:
    """Tracked and not-yet-tracked shell and make files, so a new caller is checked before commit."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", *_GLOBS],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = (project_root / name for name in sorted(set(listed.stdout.splitlines())))
    return [path for path in paths if path.is_file()]


def strip_comment(line: str) -> str:
    """The code half of one line: a `#` opening a comment ends it, one inside quotes does not.

    Prose in this repo names helpers constantly (`llb_load_env` reads .env), and a `#` also appears
    mid-word in parameter expansion (`${name#prefix}`), so both cases have to survive.
    """
    quote = ""
    for index, char in enumerate(line):
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index]
    return line


def _code_lines(path: Path) -> list[str]:
    return [strip_comment(line) for line in path.read_text(encoding="utf-8").splitlines()]


def definitions(path: Path) -> set[str]:
    """Every `llb_*` function one file defines."""
    matched = (_DEFINITION.match(line) for line in _code_lines(path))
    return {found.group(1) for found in matched if found}


def references(path: Path) -> Iterator[tuple[int, str]]:
    """Every `llb_*` name one file NAMES, as `(line number, name)`.

    A bare name is a reference whether it is called directly or handed to a runner
    (`llb_fail_if_output "$LABEL" llb_cyclomatic_scan`) -- both break the same way when the
    definition goes. The defining line itself is not a reference to itself.
    """
    for number, line in enumerate(_code_lines(path), 1):
        defined = _DEFINITION.match(line)
        for name in _REFERENCE.findall(line):
            if defined is None or name != defined.group(1):
                yield number, name


def _resolved(candidate: str, caller: Path, project_root: Path) -> Path | None:
    """A declared source path, resolved the way shellcheck resolves it: script dir, then root."""
    for base in (caller.parent, project_root):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return None


def declared_sources(caller: Path, project_root: Path) -> set[Path]:
    """The files one caller declares it loads."""
    text = caller.read_text(encoding="utf-8")
    declared = _DIRECTIVE.findall(text) + _REQUIRES.findall(text)
    declared += [found for line in _code_lines(caller) for found in _MAKE_SOURCE.findall(line)]
    resolved = (_resolved(candidate, caller, project_root) for candidate in declared)
    return {path for path in resolved if path is not None}


def scope(caller: Path, project_root: Path) -> set[Path]:
    """The caller plus everything it transitively declares it sources."""
    seen: set[Path] = set()
    pending = [caller.resolve()]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(declared_sources(current, project_root))
    return seen


def unresolved_calls(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Every `llb_*` name used with no definition in its caller's declared scope."""
    known: dict[Path, set[str]] = {}
    findings: list[str] = []
    for caller in listed_callers(project_root):
        in_scope: set[str] = set()
        for member in scope(caller, project_root):
            in_scope |= known.setdefault(member, definitions(member))
        for number, name in references(caller):
            if name not in in_scope:
                where = caller.relative_to(project_root)
                findings.append(f"{where}:{number}: no definition in scope -> {name}")
    return findings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="check llb_* helper calls against their scope")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    findings = unresolved_calls(args.root)
    for finding in findings:
        _LOG.error("ERROR: %s", finding)
    _LOG.info("[shell-symbols] %d unresolved llb_* call(s)", len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
