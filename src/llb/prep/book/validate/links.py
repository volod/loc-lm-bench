"""Relative links, wikilinks, embeds, block anchors, and assets.

Heading fragments use the same slug as `llb.quality.doc_links`. A PDF fragment
must be `#page=<n>` inside the source page count. An image under `assets/`,
`review/`, or `source/pages/` is an asset: every one that is linked must exist,
and every one on disk must be linked.
"""

from pathlib import Path

from llb.quality.doc_links import anchors
from llb.prep.book.validate.constants import (
    ASSET_PREFIXES,
    BROKEN_ANCHOR,
    BROKEN_EMBED,
    BROKEN_RELATIVE,
    BROKEN_WIKILINK,
    IMAGE_SUFFIXES,
    MISSING_ASSET,
    PDF_FRAGMENT,
    UNREFERENCED_ASSET,
)
from llb.prep.book.validate.types import BookContext, Finding, VaultDoc, VaultLink


def link_findings(ctx: BookContext, block_anchors: dict[Path, set[str]]) -> list[Finding]:
    """Every link that does not land, and every asset that is missing or unused."""
    headings: dict[Path, set[str]] = {}
    findings = [_missing_or_fragment(link, ctx, headings, block_anchors) for link in ctx.links]
    findings.extend(_unused_assets(ctx))
    return [item for item in findings if item is not None]


def anchor_index(docs: list[VaultDoc]) -> dict[Path, set[str]]:
    """Resolved path to the block anchors defined in that note."""
    return {doc.path.resolve(): set(_names(doc)) for doc in docs}


def repeated_anchor_findings(docs: list[VaultDoc]) -> list[Finding]:
    """A block anchor repeated in one note does not resolve to one block."""
    findings: list[Finding] = []
    for doc in docs:
        seen: set[str] = set()
        for name in _names(doc):
            if name in seen:
                findings.append(
                    Finding(BROKEN_ANCHOR, doc.rel, f"block anchor ^{name} is repeated")
                )
            seen.add(name)
    return findings


def _names(doc: VaultDoc) -> list[str]:
    from llb.prep.book.validate.text import block_anchor_names

    return block_anchor_names(doc.body)


def _missing_or_fragment(
    link: VaultLink,
    ctx: BookContext,
    headings: dict[Path, set[str]],
    block_anchors: dict[Path, set[str]],
) -> Finding | None:
    if not link.exists:
        return Finding(_missing_class(link), link.source, _missing_detail(link))
    problem = _fragment_problem(link, ctx, headings, block_anchors)
    if problem is None:
        return None
    return Finding(_fragment_class(link), link.source, f"line {link.line}: {problem}")


def _missing_class(link: VaultLink) -> str:
    if link.kind == "embed":
        return BROKEN_EMBED
    if link.kind == "wikilink":
        return BROKEN_WIKILINK
    if _is_asset_target(link.target):
        return MISSING_ASSET
    return BROKEN_RELATIVE


def _missing_detail(link: VaultLink) -> str:
    shown = link.target if not link.fragment else f"{link.target}#{link.fragment}"
    if link.kind == "embed":
        return f"line {link.line}: missing embed -> {shown}"
    if link.kind == "wikilink":
        return f"line {link.line}: missing wikilink -> {shown}"
    if _is_asset_target(link.target):
        return f"line {link.line}: missing asset -> {shown}"
    return f"line {link.line}: missing file -> {shown}"


def _fragment_problem(
    link: VaultLink,
    ctx: BookContext,
    headings: dict[Path, set[str]],
    block_anchors: dict[Path, set[str]],
) -> str | None:
    if not link.fragment:
        return None
    if link.located.suffix.lower() == ".pdf":
        return _pdf_fragment(link.fragment, ctx)
    if link.located.suffix.lower() != ".md":
        return f"fragment #{link.fragment} is not on a note or a PDF"
    if link.fragment.startswith("^"):
        name = link.fragment[1:]
        known = block_anchors.get(link.located.resolve(), set())
        return None if name in known else f"missing block anchor ^{name}"
    return _heading_fragment(link, headings)


def _heading_fragment(link: VaultLink, headings: dict[Path, set[str]]) -> str | None:
    resolved = link.located.resolve()
    if resolved not in headings:
        headings[resolved] = anchors(link.located)
    if link.fragment in headings[resolved]:
        return None
    return f"missing anchor #{link.fragment}"


def _pdf_fragment(fragment: str, ctx: BookContext) -> str | None:
    key, sep, value = fragment.partition("=")
    name = ctx.fragment_name
    if key != name or not sep or not value.isdigit():
        return f"PDF fragment must be #{name}=<n>"
    number = int(value)
    last = ctx.config.source.page_count
    if number < 1 or number > last:
        return f"PDF page {number} is outside 1-{last}"
    return None


def _fragment_class(link: VaultLink) -> str:
    if link.located.suffix.lower() == ".pdf":
        return PDF_FRAGMENT
    if link.fragment.startswith("^"):
        return BROKEN_ANCHOR
    return BROKEN_RELATIVE


def _unused_assets(ctx: BookContext) -> list[Finding]:
    used: set[str] = set()
    for link in ctx.links:
        if not link.exists:
            continue
        rel = _asset_rel(ctx.book_dir, link.located)
        if rel is not None:
            used.add(rel)
    findings: list[Finding] = []
    for rel in sorted(_disk_assets(ctx.book_dir) - used):
        findings.append(Finding(UNREFERENCED_ASSET, rel, "no note embeds or links this file"))
    return findings


def _disk_assets(book_dir: Path) -> set[str]:
    found: set[str] = set()
    for prefix in ASSET_PREFIXES:
        base = book_dir / prefix
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                found.add(path.relative_to(book_dir).as_posix())
    return found


def _asset_rel(book_dir: Path, path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(book_dir.resolve()).as_posix()
    except ValueError:
        return None
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    if any(rel.startswith(prefix) for prefix in ASSET_PREFIXES):
        return rel
    return None


def _is_asset_target(target: str) -> bool:
    path = Path(target)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    parts = path.parts
    if "assets" in parts or "review" in parts:
        return True
    return "source" in parts and "pages" in parts
