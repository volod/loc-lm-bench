"""Group open review work by primary page class and inferred chapter."""

from collections import Counter
from typing import TypedDict

from llb.core.contracts.data_prep.book_structure import StructureNode
from llb.prep.book.classes.records import load_page_classes
from llb.prep.book.render.names import walk
from llb.prep.book.validate.types import OpenMark, ValidationReport


class ReviewCount(TypedDict):
    group: str
    selected_pages: int
    open_entries: int


def review_counts(
    report: ValidationReport, marks: list[OpenMark]
) -> tuple[list[ReviewCount], list[ReviewCount]]:
    """Include failed-page entries and pages with an empty queue."""
    pages = set(report.context.config.pages)
    entries = Counter(mark.pdf_page for mark in marks)
    if report.facts.queue_marks is None:
        entries.update(row.pdf_page for row in report.pages if row.status == "failed")
    else:
        by_id = {row.page_id: row.pdf_page for row in report.pages}
        entries.update(by_id[pid] for pid in report.facts.failed_pages if pid in by_id)
    classes: dict[int, str] = {
        row.pdf_page: row.primary for row in load_page_classes(report.context.run_dir)
    }
    chapters = _chapters(report.context.structure.hierarchy, pages | entries.keys())
    return (
        _counts(pages, entries, classes, "unclassified"),
        _counts(pages, entries, chapters, "outside inferred chapters"),
    )


def _chapters(root: StructureNode, pages: set[int]) -> dict[int, str]:
    chapters: dict[int, str] = {}
    for node in walk(root):
        if node.level != "chapter":
            continue
        label = f"{node.node_id} (PDF {node.pdf_start}-{node.pdf_end})"
        chapters.update({page: label for page in pages if node.pdf_start <= page <= node.pdf_end})
    return chapters


def _counts(
    pages: set[int], entries: Counter[int], groups: dict[int, str], fallback: str
) -> list[ReviewCount]:
    page_counts = Counter(groups.get(page, fallback) for page in pages)
    entry_counts: Counter[str] = Counter()
    for page, count in entries.items():
        entry_counts[groups.get(page, fallback)] += count
    return [
        ReviewCount(
            group=group, selected_pages=page_counts[group], open_entries=entry_counts[group]
        )
        for group in sorted(page_counts.keys() | entry_counts.keys())
    ]


def render_counts(rows: list[ReviewCount]) -> list[str]:
    return [
        "| Group | Selected pages | Open entries |",
        "| --- | --- | --- |",
        *(f"| {row['group']} | {row['selected_pages']} | {row['open_entries']} |" for row in rows),
    ]
