"""Assemble the hierarchy and the page-range fallback.

A heading observed on a page can be a node. A contents entry that matches one is evidence
for that node. A contents entry that matches nothing is a conflict and is not a node.
"""

import re

from llb.core.contracts.data_prep.book_structure import (
    NODE_EVIDENCE,
    RunningHeader,
    StructureConflict,
    StructureNode,
)
from llb.prep.book.structure.constants import BOOK_TITLE
from llb.prep.book.structure.contents import Entry
from llb.prep.book.structure.draft import RANK, Draft, evidence_of, level_of
from llb.prep.book.structure.headings import Heading
from llb.prep.book.structure.reconcile import reconcile, unmatched_contents


def build_hierarchy(
    pages: list[int],
    headings: list[Heading],
    entries: list[Entry],
    headers: list[RunningHeader],
    printed_of: dict[int, str | None],
) -> tuple[StructureNode, list[StructureConflict]]:
    """The book tree and every recorded disagreement."""
    if not headings:
        root = _fallback(pages, headers)
        return _model(root, printed_of), unmatched_contents(entries)
    nodes = [_from_heading(item) for item in headings]
    _ends(nodes, pages)
    conflicts = reconcile(entries, nodes, headers, printed_of)
    root = _link(pages, nodes)
    return _model(root, printed_of), conflicts


def _from_heading(item: Heading) -> Draft:
    return Draft(
        item.level,
        item.number,
        item.title,
        item.pdf_page,
        item.pdf_page,
        item.block_id,
        [item.evidence],
        item.evidence == "typography",
    )


def _ends(nodes: list[Draft], pages: list[int]) -> None:
    for index, node in enumerate(nodes):
        end = _run_end(node.pdf_start, pages)
        for later in nodes[index + 1 :]:
            if later.pdf_start > end:
                break
            if RANK[later.level] <= RANK[node.level]:
                end = _cut(node.pdf_start, later.pdf_start)
                break
        node.pdf_end = max(node.pdf_start, end)


def _run_end(start: int, pages: list[int]) -> int:
    """The last configured page in the same contiguous run as `start`.

    A sample that skips a span of PDF pages does not extend a chapter across the hole.
    """
    end = start
    for pdf in pages:
        if pdf <= end:
            continue
        if pdf > end + 1:
            break
        end = pdf
    return end


def _cut(start: int, later: int) -> int:
    if later > start:
        return later - 1
    return start


def _link(pages: list[int], nodes: list[Draft]) -> Draft:
    root = Draft("book", "", BOOK_TITLE, pages[0], pages[-1], None, ["root"], False)
    stack = [root]
    for node in nodes:
        while _closed(stack[-1], node):
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)
    return root


def _closed(parent: Draft, node: Draft) -> bool:
    """Pop a parent that this heading does not belong under."""
    if parent.level == "book":
        return False
    if RANK[parent.level] >= RANK[node.level]:
        return True
    return parent.pdf_end + 1 < node.pdf_start


def _fallback(pages: list[int], headers: list[RunningHeader]) -> Draft:
    labels = _labels(pages, headers)
    children = [_range(text, group) for text, group in _groups(pages, labels)]
    return Draft(
        "book", "", BOOK_TITLE, pages[0], pages[-1], None, ["root", "fallback"], True, children
    )


def _labels(pages: list[int], headers: list[RunningHeader]) -> dict[int, str]:
    label = {pdf: "" for pdf in pages}
    for span in headers:
        for pdf in pages:
            if span.pdf_start <= pdf <= span.pdf_end and len(span.text) > len(label[pdf]):
                label[pdf] = span.text
    return label


def _groups(pages: list[int], labels: dict[int, str]) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    for pdf in pages:
        text = labels[pdf]
        if groups and groups[-1][0] == text and pdf == groups[-1][1][-1] + 1:
            groups[-1][1].append(pdf)
        else:
            groups.append((text, [pdf]))
    return groups


def _range(text: str, group: list[int]) -> Draft:
    title = f"Page {group[0]}" if len(group) == 1 else f"Pages {group[0]}-{group[-1]}"
    evidence = ["fallback"]
    if text:
        title = f"{title}: {text}"
        evidence.append("running-header")
    return Draft("page-range", "", title, group[0], group[-1], None, evidence, True)


def _model(
    node: Draft, printed_of: dict[int, str | None], parent: str = "", ordinal: int = 0
) -> StructureNode:
    node_id = "book" if node.level == "book" else _child_id(parent, node, ordinal)
    return StructureNode(
        node_id=node_id,
        level=level_of(node.level),
        number=node.number,
        title=node.title,
        pdf_start=node.pdf_start,
        pdf_end=node.pdf_end,
        printed_start=printed_of.get(node.pdf_start),
        printed_end=printed_of.get(node.pdf_end),
        block_id=node.block_id,
        evidence=[evidence_of(name) for name in _order(node.evidence)],
        fallback=node.fallback,
        children=[
            _model(child, printed_of, node_id, index)
            for index, child in enumerate(node.children, start=1)
        ],
    )


def _child_id(parent: str, node: Draft, ordinal: int) -> str:
    raw = node.number.lower().replace(".", "-") if node.number else f"{ordinal:02d}"
    slug = re.sub(r"[^a-z0-9-]", "", raw) or f"{ordinal:02d}"
    return f"{parent}/{node.level}-{slug}"


def _order(names: list[str]) -> list[str]:
    sequence = [str(name) for name in NODE_EVIDENCE]

    def key(name: str) -> int:
        if name in sequence:
            return sequence.index(name)
        return len(sequence)

    return sorted(set(names), key=key)
