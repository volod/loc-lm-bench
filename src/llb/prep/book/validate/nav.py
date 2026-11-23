"""Contents reachability, and the previous/next chain.

The contents note opens the chain with Next. Each note's Next is the following
note, and that note's Previous points back. Walking Next from the first note
visits every content note once.
"""

from pathlib import Path

from llb.prep.book.render.constants import TOC_FILENAME
from llb.prep.book.validate.constants import (
    PREV_NEXT_CHAIN,
    PREV_NEXT_RECIPROCAL,
    UNREACHABLE_CONTENT,
)
from llb.prep.book.validate.types import BookContext, Finding, VaultDoc, VaultLink

_NAV = frozenset({"Previous", "Next"})


def nav_findings(ctx: BookContext) -> list[Finding]:
    """Reachability from the contents, reciprocal neighbours, and one chain."""
    by_path = {doc.path.resolve(): doc for doc in ctx.content}
    nxt, prv, bad = _nav_maps(ctx.links, by_path)
    findings = list(bad)
    if ctx.toc is None:
        findings.append(Finding(UNREACHABLE_CONTENT, TOC_FILENAME, "the contents file is missing"))
        return findings
    names = {path: doc.rel for path, doc in by_path.items()}
    names[ctx.toc.path.resolve()] = ctx.toc.rel
    findings.extend(_reciprocal(nxt, prv, set(by_path), names))
    findings.extend(_chain(ctx.toc, nxt, prv, by_path))
    findings.extend(_reachable(ctx.toc, ctx.links, by_path))
    return findings


def _nav_maps(
    links: list[VaultLink], by_path: dict[Path, VaultDoc]
) -> tuple[dict[Path, Path], dict[Path, Path], list[Finding]]:
    nxt: dict[Path, Path] = {}
    prv: dict[Path, Path] = {}
    findings: list[Finding] = []
    grouped = _group_nav(links, by_path, findings)
    _assign_nav(grouped, by_path, nxt, prv, findings)
    return nxt, prv, findings


def _group_nav(
    links: list[VaultLink], by_path: dict[Path, VaultDoc], findings: list[Finding]
) -> dict[tuple[Path, str], set[Path]]:
    grouped: dict[tuple[Path, str], set[Path]] = {}
    for link in links:
        if link.label not in _NAV:
            continue
        source = link.source_path.resolve()
        target = link.located.resolve() if link.exists else None
        if target is None or target not in by_path:
            findings.append(_bad_nav(link))
            continue
        grouped.setdefault((source, link.label), set()).add(target)
    return grouped


def _assign_nav(
    grouped: dict[tuple[Path, str], set[Path]],
    by_path: dict[Path, VaultDoc],
    nxt: dict[Path, Path],
    prv: dict[Path, Path],
    findings: list[Finding],
) -> None:
    for (source, label), targets in grouped.items():
        doc = by_path.get(source)
        file = doc.rel if doc is not None else TOC_FILENAME
        if len(targets) != 1:
            findings.append(Finding(PREV_NEXT_CHAIN, file, f"{label} names more than one note"))
            continue
        target = next(iter(targets))
        if label == "Next":
            nxt[source] = target
        else:
            prv[source] = target


def _bad_nav(link: VaultLink) -> Finding:
    shown = link.target or link.located.name
    detail = f"line {link.line}: {link.label} does not name a content note ({shown})"
    return Finding(PREV_NEXT_CHAIN, link.source, detail)


def _reciprocal(
    nxt: dict[Path, Path],
    prv: dict[Path, Path],
    content: set[Path],
    names: dict[Path, str],
) -> list[Finding]:
    """Previous and next agree between content notes. The contents Next starts the chain."""
    findings: list[Finding] = []
    for source, target in nxt.items():
        if source not in content or prv.get(target) == source:
            continue
        findings.append(
            Finding(
                PREV_NEXT_RECIPROCAL,
                names.get(target, target.name),
                "Previous does not point back at the note that links Next here",
            )
        )
    for source, target in prv.items():
        if source not in content or nxt.get(target) == source:
            continue
        findings.append(
            Finding(
                PREV_NEXT_RECIPROCAL,
                names.get(source, source.name),
                "Previous points at a note that does not link Next back",
            )
        )
    return findings


def _chain(
    toc: VaultDoc,
    nxt: dict[Path, Path],
    prv: dict[Path, Path],
    by_path: dict[Path, VaultDoc],
) -> list[Finding]:
    if not by_path:
        return []
    findings: list[Finding] = []
    heads = [path for path in by_path if path not in prv]
    if len(heads) != 1:
        for path in heads:
            findings.append(
                Finding(
                    PREV_NEXT_CHAIN, by_path[path].rel, "more than one note has no Previous link"
                )
            )
        if not heads:
            findings.append(
                Finding(PREV_NEXT_CHAIN, TOC_FILENAME, "no note starts the previous/next chain")
            )
        return findings
    head = heads[0]
    toc_next = nxt.get(toc.path.resolve())
    if toc_next != head:
        findings.append(Finding(PREV_NEXT_CHAIN, toc.rel, "Next does not start the note chain"))
    visited = _walk(head, nxt, findings, by_path)
    for path, doc in by_path.items():
        if path not in visited:
            findings.append(
                Finding(
                    PREV_NEXT_CHAIN, doc.rel, "the previous/next chain does not visit this note"
                )
            )
    return findings


def _walk(
    head: Path,
    nxt: dict[Path, Path],
    findings: list[Finding],
    by_path: dict[Path, VaultDoc],
) -> set[Path]:
    visited: set[Path] = set()
    current: Path | None = head
    while current is not None:
        if current in visited:
            findings.append(
                Finding(
                    PREV_NEXT_CHAIN,
                    by_path[current].rel,
                    "the previous/next chain repeats this note",
                )
            )
            break
        visited.add(current)
        current = nxt.get(current)
    return visited


def _reachable(
    toc: VaultDoc, links: list[VaultLink], by_path: dict[Path, VaultDoc]
) -> list[Finding]:
    edges: dict[Path, set[Path]] = {}
    for link in links:
        if not link.exists or link.located.suffix.lower() != ".md":
            continue
        edges.setdefault(link.source_path.resolve(), set()).add(link.located.resolve())
    seen: set[Path] = set()
    stack = [toc.path.resolve()]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edges.get(current, ()))
    return [
        Finding(UNREACHABLE_CONTENT, doc.rel, "not reachable from the contents")
        for path, doc in by_path.items()
        if path not in seen
    ]
