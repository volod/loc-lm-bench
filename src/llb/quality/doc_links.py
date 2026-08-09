"""Resolve every relative link in the docs tree: the file exists, and the `#anchor` is a heading.

The current-implementation docs are a three-level tree (index -> area -> topic), so most navigation
is now a link between files rather than a scroll inside one. A link that rots is a topic that
becomes unfindable, and nothing else in the toolchain checks one: `pymarkdown` lints style, not
targets. This walks every tracked Markdown file, resolves each relative link, and names the ones
that no longer land -- a missing file, or an anchor no heading in the target produces.

Anchors are computed the way a Markdown host slugs a heading (lowercase, punctuation dropped, spaces
to hyphens, `-1` on a repeat), so moving a section BETWEEN files keeps its anchor and only the path
has to change.
"""

import argparse
import logging
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from llb.core.paths import PROJECT_ROOT

_LOG = logging.getLogger(__name__)

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^(#{1,6}) (.+)$")
_FENCE = "```"
_EXTERNAL = ("http://", "https://", "mailto:", "/")


def heading_anchor(title: str) -> str:
    """The fragment a Markdown host derives from one heading."""
    text = title.replace("`", "").strip().lower()
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", text))


def anchors(path: Path) -> set[str]:
    """Every fragment a document offers, including the `-1` form a repeated heading produces."""
    found: set[str] = set()
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(_FENCE):
            fenced = not fenced
            continue
        matched = _HEADING.match(line) if not fenced else None
        if not matched:
            continue
        base = heading_anchor(matched.group(2))
        anchor, repeat = base, 0
        while anchor in found:
            repeat += 1
            anchor = f"{base}-{repeat}"
        found.add(anchor)
    return found


def tracked_docs(project_root: Path) -> list[Path]:
    """Tracked and not-yet-tracked Markdown files, so a new page is checked before it is committed."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [project_root / name for name in sorted(set(listed.stdout.split()))]


def relative_links(doc: Path) -> Iterator[tuple[int, str]]:
    """Every non-external link target in one document, as `(line number, target)`.

    Fenced blocks are skipped: a command sample routinely contains bracket-paren text that reads
    like a link and that no reader can click.
    """
    fenced = False
    for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith(_FENCE):
            fenced = not fenced
        if fenced:
            continue
        for target in _LINK.findall(line):
            if not target.startswith(_EXTERNAL):
                yield number, target


def _landing_failure(doc: Path, target: str, known: dict[Path, set[str]]) -> str | None:
    """Why one link does not land, or None when it does."""
    path_part, _, anchor = target.partition("#")
    resolved = ((doc.parent / path_part) if path_part else doc).resolve()
    if not resolved.exists():
        return "missing file"
    if not anchor or resolved.suffix != ".md":
        return None
    if resolved not in known:
        known[resolved] = anchors(resolved)
    return None if anchor in known[resolved] else "missing anchor"


def broken_links(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Every relative link that does not land, named `file:line: reason -> target`."""
    known: dict[Path, set[str]] = {}
    findings: list[str] = []
    for doc in tracked_docs(project_root):
        for number, target in relative_links(doc):
            failure = _landing_failure(doc, target, known)
            if failure is not None:
                findings.append(f"{doc.relative_to(project_root)}:{number}: {failure} -> {target}")
    return findings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="check relative links across the docs tree")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    findings = broken_links(args.root)
    for finding in findings:
        _LOG.error("ERROR: %s", finding)
    _LOG.info("[doc-links] %d broken link(s)", len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
