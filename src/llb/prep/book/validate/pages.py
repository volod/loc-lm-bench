"""Every PDF page is completed, failed, or skipped, and every note names its pages."""

import re

from llb.prep.book.ranges import page_id
from llb.prep.book.render.constants import TOC_FILENAME
from llb.prep.book.validate.constants import (
    BLOCK_WITHOUT_PAGE,
    BROKEN_ANCHOR,
    UNACCOUNTED_PAGE,
)
from llb.prep.book.validate.text import block_anchor_names, matter_int, page_anchor_number
from llb.prep.book.validate.types import BookContext, Finding, PageRow, VaultDoc

_FAILED = re.compile(r'\[!page-failed\]\(([^)\s]+)(?:\s+"([^"]*)")?\)')


def page_findings(ctx: BookContext) -> tuple[list[PageRow], list[Finding]]:
    """Account for every PDF page, and refuse a note that does not name its pages."""
    ranges, anchors, failed, findings = _scan_notes(ctx)
    rows = _rows(ctx, anchors, failed, ranges, findings)
    return rows, findings


def _scan_notes(
    ctx: BookContext,
) -> tuple[list[tuple[str, int, int]], dict[int, list[str]], dict[int, str], list[Finding]]:
    ranges: list[tuple[str, int, int]] = []
    anchors: dict[int, list[str]] = {}
    failed: dict[int, str] = {}
    findings: list[Finding] = []
    page_count = ctx.config.source.page_count
    for doc in ctx.content:
        start, end, problem = _range(doc, page_count)
        if problem is not None:
            findings.append(Finding(BLOCK_WITHOUT_PAGE, doc.rel, problem))
        elif start is not None and end is not None:
            ranges.append((doc.rel, start, end))
        _collect_anchors(doc, start, end, page_count, anchors, findings)
        for number, reason in _failures(doc.body, ctx.fragment_name):
            failed.setdefault(number, reason)
    _repeated_page_anchors(anchors, findings)
    return ranges, anchors, failed, findings


def _collect_anchors(
    doc: VaultDoc,
    start: int | None,
    end: int | None,
    page_count: int,
    anchors: dict[int, list[str]],
    findings: list[Finding],
) -> None:
    for name in block_anchor_names(doc.body):
        number = page_anchor_number(name)
        if number is None:
            continue
        anchors.setdefault(number, []).append(doc.rel)
        if start is None or end is None or start <= number <= end:
            continue
        label = page_id(number, page_count)
        findings.append(
            Finding(
                BLOCK_WITHOUT_PAGE,
                doc.rel,
                f"page anchor {label} is outside PDF {start}-{end}",
            )
        )


def _repeated_page_anchors(anchors: dict[int, list[str]], findings: list[Finding]) -> None:
    for number, files in sorted(anchors.items()):
        if len(files) < 2:
            continue
        findings.append(
            Finding(BROKEN_ANCHOR, files[1], f"PDF page {number} has more than one page anchor")
        )


def _rows(
    ctx: BookContext,
    anchors: dict[int, list[str]],
    failed: dict[int, str],
    ranges: list[tuple[str, int, int]],
    findings: list[Finding],
) -> list[PageRow]:
    selected = set(ctx.config.pages)
    spec = ctx.config.pages_spec
    page_count = ctx.config.source.page_count
    rows: list[PageRow] = []
    for number in range(1, page_count + 1):
        rows.append(_one(number, page_count, selected, spec, anchors, failed, ranges, findings))
    return rows


def _one(
    number: int,
    page_count: int,
    selected: set[int],
    spec: str,
    anchors: dict[int, list[str]],
    failed: dict[int, str],
    ranges: list[tuple[str, int, int]],
    findings: list[Finding],
) -> PageRow:
    pid = page_id(number, page_count)
    if number not in selected:
        return PageRow(number, pid, "skipped", f"outside configured pages {spec}")
    if number in failed:
        return PageRow(number, pid, "failed", failed[number])
    if number in anchors:
        return PageRow(number, pid, "completed", "")
    findings.append(
        Finding(
            UNACCOUNTED_PAGE,
            _holder(ranges, number),
            f"PDF page {number} is not completed, failed, or skipped",
        )
    )
    return PageRow(number, pid, "unaccounted", "no page anchor and no failure marker")


def _holder(ranges: list[tuple[str, int, int]], number: int) -> str:
    for rel, start, end in ranges:
        if start <= number <= end:
            return rel
    return TOC_FILENAME


def _range(doc: VaultDoc, page_count: int) -> tuple[int | None, int | None, str | None]:
    start = matter_int(doc.matter, "pdf_start")
    end = matter_int(doc.matter, "pdf_end")
    if start is None or end is None:
        return start, end, "front matter does not name a PDF page"
    if start < 1 or end < start or end > page_count:
        return start, end, f"front matter PDF range {start}-{end} is outside 1-{page_count}"
    return start, end, None


def _failures(body: str, fragment_name: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for match in _FAILED.finditer(body):
        number = _link_page(match.group(1), fragment_name)
        if number is None:
            continue
        title = match.group(2) or ""
        reason = title.split(" ", 1)[1].strip() if " " in title else ""
        found.append((number, reason or "recovery failed"))
    return found


def _link_page(url: str, fragment_name: str) -> int | None:
    fragment = url.partition("#")[2]
    key, sep, value = fragment.partition("=")
    if key != fragment_name or not sep or not value.isdigit():
        return None
    return int(value)
