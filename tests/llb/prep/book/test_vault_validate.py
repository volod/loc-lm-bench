"""A clean vault passes and each failure class names the file that caused it."""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.cli.app import app
from llb.core.contracts.data_prep.book_resolve import (
    CorrectionEdit,
    OrphanedEdit,
    PageCorrectionOverlay,
)
from llb.core.contracts.data_prep.book_review import (
    QUEUE_KIND,
    QUEUE_SCHEMA_ID,
    REVIEW_SCHEMA_VERSION,
    ReviewQueue,
    ReviewQueueEntry,
)
from llb.prep.book.config import read_config, write_config
from llb.prep.book.constants import METADATA_DIRNAME, PROCESSING_REPORT_FILENAME
from llb.prep.book.render.constants import RENDER_MANIFEST_FILENAME
from llb.prep.book.resolve.constants import OVERLAY_KIND, OVERLAY_SCHEMA_ID, SCHEMA_VERSION
from llb.prep.book.resolve.io import write_overlay
from llb.prep.book.review.io import queue_path, read_queue, write_queue
from llb.prep.book.review.markers import ParsedMarker, parse_markers, render_marker
from llb.prep.book.validate import validate_book
from llb.prep.book.validate.text import extract_links
from llb.prep.book.validate.types import VaultDoc
from llb.prep.book.validate.constants import (
    BLOCK_WITHOUT_PAGE,
    BROKEN_ANCHOR,
    BROKEN_EMBED,
    BROKEN_RELATIVE,
    BROKEN_WIKILINK,
    MARKER_MISMATCH,
    MISSING_ASSET,
    PDF_FRAGMENT,
    PREV_NEXT_CHAIN,
    PREV_NEXT_RECIPROCAL,
    UNACCOUNTED_PAGE,
    UNREACHABLE_CONTENT,
    UNREFERENCED_ASSET,
)
from tests.llb.prep.book.test_render import _oversized, _sample, _vault

runner = CliRunner()
_HEX = "ab" * 32


def test_display_math_is_not_a_relative_link(tmp_path: Path) -> None:
    note = tmp_path / "content" / "page.md"
    note.parent.mkdir(parents=True)
    body = "\n".join(["$$", r"E](0.5\text{s})", "![[review/crop.png]]", "$$", "[real](missing.md)"])
    note.write_text(body, encoding="utf-8")
    links = extract_links(VaultDoc("content/page.md", note, {}, body), tmp_path)
    assert [(link.line, link.target) for link in links] == [
        (2, "review/crop.png"),
        (3, "missing.md"),
    ]


def test_clean_vault_passes_and_the_report_accounts_every_page(tmp_path: Path) -> None:
    run, vault = _ready(tmp_path)
    report = validate_book(vault)
    assert report.ok
    assert [row.status for row in report.pages] == ["completed", "completed", "failed"]
    assert report.pages[2].reason == "recovery failed"
    assert ("rapidocr", "none", "original", 11) in report.facts.readings
    assert report.facts.scores["rapidocr"].buckets["0.95-1.00"] == 11
    assert any("review queue was not written" in item for item in report.warnings)
    text = _report_text(vault)
    assert "Orphaned overlay entries: 0" in text
    assert "Confidence is diagnostic routing evidence" in text
    _assert_json(vault, run)
    again = validate_book(vault)
    assert again.ok
    assert _report_text(vault) == text


def test_skipped_pages_are_accounted_with_the_configured_range(tmp_path: Path) -> None:
    run, vault = _ready(tmp_path)
    config = read_config(run)
    source = config.source.model_copy(update={"page_count": 5})
    write_config(run, config.model_copy(update={"source": source}))
    report = validate_book(vault)
    assert report.ok
    assert [(row.pdf_page, row.status) for row in report.pages] == [
        (1, "completed"),
        (2, "completed"),
        (3, "failed"),
        (4, "skipped"),
        (5, "skipped"),
    ]
    assert "outside configured pages" in report.pages[3].reason
    assert "skipped | outside configured pages" in _report_text(vault)


def test_an_orphaned_edit_is_counted_and_is_not_a_failure(tmp_path: Path) -> None:
    run, vault = _ready(tmp_path)
    _write_orphan(run, vault)
    report = validate_book(vault)
    assert report.ok
    assert report.facts.orphans == 1
    assert report.facts.orphaned_by_reason["block_changed"] == 1
    assert "Orphaned overlay entries: 1" in _report_text(vault)


def test_a_matching_queue_clears_the_warning_and_an_extra_entry_fails(tmp_path: Path) -> None:
    run, vault = _ready(tmp_path)
    _write_matching_queue(run, vault)
    report = validate_book(vault)
    assert report.ok
    assert not any("review queue was not written" in item for item in report.warnings)
    _add_extra_entry(run)
    failed = validate_book(vault)
    assert not failed.ok
    named = any(
        item.check == MARKER_MISMATCH and "queue.json" in item.file for item in failed.findings
    )
    assert named


@pytest.mark.parametrize("policy", ["symlink", "copy", "reference"])
def test_each_source_policy_validates(tmp_path: Path, policy: str) -> None:
    run = _sample(tmp_path / policy)
    vault = _vault(run, tmp_path / policy / "vault", source_policy=policy)
    assert validate_book(vault).ok


def test_page_fallback_and_an_oversized_split_validate(tmp_path: Path) -> None:
    run = _sample(tmp_path / "fallback")
    vault = _vault(run, tmp_path / "fallback" / "vault", page_fallback=True)
    assert validate_book(vault).ok
    wide = _oversized(tmp_path / "wide")
    assert validate_book(_vault(wide, tmp_path / "wide" / "vault")).ok


@pytest.mark.parametrize(
    ("kind", "needle"),
    [
        (UNACCOUNTED_PAGE, "content/"),
        (UNREACHABLE_CONTENT, "orphan.md"),
        (PREV_NEXT_RECIPROCAL, "content/"),
        (PREV_NEXT_CHAIN, "content/"),
        (BROKEN_RELATIVE, "content/"),
        (BROKEN_WIKILINK, "content/"),
        (BROKEN_ANCHOR, "content/"),
        (BROKEN_EMBED, "content/"),
        (PDF_FRAGMENT, "content/"),
        (MISSING_ASSET, "content/"),
        (UNREFERENCED_ASSET, "assets/extra.png"),
        (BLOCK_WITHOUT_PAGE, "content/"),
        (MARKER_MISMATCH, "content/"),
    ],
)
def test_each_failure_class_names_the_offending_file(
    tmp_path: Path, kind: str, needle: str
) -> None:
    _run, vault = _ready(tmp_path)
    _break(vault, kind)
    report = validate_book(vault)
    assert not report.ok
    assert any(item.check == kind and needle in item.file for item in report.findings)


def test_validate_exits_nonzero_when_a_check_fails(tmp_path: Path) -> None:
    _run, vault = _ready(tmp_path)
    _break(vault, BROKEN_WIKILINK)
    result = runner.invoke(app, ["book-ocr", "validate", str(vault)])
    assert result.exit_code == 1
    assert "validate failed" in result.stdout
    assert "broken-wikilink" in result.stderr


def test_validate_refuses_a_directory_that_is_not_a_vault(tmp_path: Path) -> None:
    result = runner.invoke(app, ["book-ocr", "validate", str(tmp_path)])
    assert result.exit_code == 2
    assert "not a reading vault" in result.stderr


def _ready(tmp_path: Path) -> tuple[Path, Path]:
    run = _sample(tmp_path)
    return run, _vault(run, tmp_path / "vault")


def _report_text(vault: Path) -> str:
    return (vault / METADATA_DIRNAME / PROCESSING_REPORT_FILENAME).read_text(encoding="utf-8")


def _assert_json(vault: Path, run: Path) -> None:
    raw = (vault / METADATA_DIRNAME / "validation.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["source"]["sha256"] == read_config(run).source.sha256
    assert payload["review_by_page_class"] == [
        {"group": "unclassified", "selected_pages": 3, "open_entries": 2}
    ]
    assert payload["review_by_chapter"] == [
        {"group": "book/chapter-1 (PDF 1-3)", "selected_pages": 3, "open_entries": 2}
    ]
    assert str(run.resolve()) not in raw
    manifest_path = vault / METADATA_DIRNAME / RENDER_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_dir"] == str(run.resolve())


def _break(vault: Path, kind: str) -> None:
    notes = sorted((vault / "content").glob("*.md"))
    _MUTATIONS[kind](vault, notes[0], notes[1])


def _drop_page_anchor(vault: Path, first: Path, second: Path) -> None:
    del vault
    holder = second if "^p0002" in second.read_text(encoding="utf-8") else first
    _drop_line(holder, "^p0002")


def _add_orphan_note(vault: Path, first: Path, second: Path) -> None:
    del first, second
    (vault / "content" / "orphan.md").write_text(
        _front("orphan", 1, 1) + "orphan\n", encoding="utf-8"
    )


def _point_previous_at_self(vault: Path, first: Path, second: Path) -> None:
    del vault, first
    _rewrite(second, r"\[Previous\]\([^)]+\)", f"[Previous]({second.name})")


def _remove_next(vault: Path, first: Path, second: Path) -> None:
    del vault, second
    first.write_text(
        first.read_text(encoding="utf-8").replace("[Next]", "[Later]"), encoding="utf-8"
    )


def _break_contents_link(vault: Path, first: Path, second: Path) -> None:
    del vault, second
    _rewrite(first, r"\.\./00_Table_of_Contents\.md", "../missing-contents.md")


def _add_missing_wikilink(vault: Path, first: Path, second: Path) -> None:
    del vault, second
    first.write_text(first.read_text(encoding="utf-8") + "\n[[missing-note]]\n", encoding="utf-8")


def _add_missing_anchor(vault: Path, first: Path, second: Path) -> None:
    del vault, second
    first.write_text(
        first.read_text(encoding="utf-8") + f"[see](./{first.name}#^missing-anchor)\n",
        encoding="utf-8",
    )


def _add_missing_embed(vault: Path, first: Path, second: Path) -> None:
    del vault, second
    first.write_text(
        first.read_text(encoding="utf-8") + "![[review/missing-crop.png]]\n", encoding="utf-8"
    )


def _break_pdf_fragment(vault: Path, first: Path, second: Path) -> None:
    del vault, second
    _rewrite(first, r"#page=1\b", "#page=99")


def _add_missing_asset(vault: Path, first: Path, second: Path) -> None:
    del vault, second
    first.write_text(
        first.read_text(encoding="utf-8") + "[figure](../assets/missing.png)\n", encoding="utf-8"
    )


def _add_unused_asset(vault: Path, first: Path, second: Path) -> None:
    del first, second
    path = vault / "assets" / "extra.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def _drop_page_range(vault: Path, first: Path, second: Path) -> None:
    del vault, second
    kept = [
        line
        for line in first.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("pdf_start:", "pdf_end:"))
    ]
    first.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _rename_marker(vault: Path, first: Path, second: Path) -> None:
    del first, second
    notes = sorted((vault / "content").glob("*.md"))
    holder = next(path for path in notes if "[!low-confidence]" in path.read_text(encoding="utf-8"))
    text = holder.read_text(encoding="utf-8").replace("[!low-confidence]", "[!not-a-marker]")
    holder.write_text(text, encoding="utf-8")


_Mutation = Callable[[Path, Path, Path], None]
_MUTATIONS: dict[str, _Mutation] = {
    UNACCOUNTED_PAGE: _drop_page_anchor,
    UNREACHABLE_CONTENT: _add_orphan_note,
    PREV_NEXT_RECIPROCAL: _point_previous_at_self,
    PREV_NEXT_CHAIN: _remove_next,
    BROKEN_RELATIVE: _break_contents_link,
    BROKEN_WIKILINK: _add_missing_wikilink,
    BROKEN_ANCHOR: _add_missing_anchor,
    BROKEN_EMBED: _add_missing_embed,
    PDF_FRAGMENT: _break_pdf_fragment,
    MISSING_ASSET: _add_missing_asset,
    UNREFERENCED_ASSET: _add_unused_asset,
    BLOCK_WITHOUT_PAGE: _drop_page_range,
    MARKER_MISMATCH: _rename_marker,
}


def _drop_line(path: Path, text: str) -> None:
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() != text]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _rewrite(path: Path, pattern: str, replacement: str) -> None:
    path.write_text(
        re.sub(pattern, replacement, path.read_text(encoding="utf-8"), count=1), encoding="utf-8"
    )


def _front(node: str, start: int, end: int) -> str:
    return (
        "---\n"
        "book: sample\n"
        f"node_id: {node}\n"
        f"pdf_start: {start}\n"
        f"pdf_end: {end}\n"
        "printed_start: null\n"
        "printed_end: null\n"
        "review_count: 0\n"
        "---\n\n"
    )


def _write_orphan(run: Path, vault: Path) -> None:
    marker = next(
        item
        for note in (vault / "content").glob("*.md")
        for item in parse_markers(note.read_text())
    )
    edit = CorrectionEdit(
        edit_id="orphan-1",
        page_id="p0002",
        block_id=marker.block_id,
        anchor=_HEX,
        author="reviewer",
        start=marker.start,
        end=marker.end,
        raw_text="x" * (marker.end - marker.start),
        replacement="1",
        evidence="kept",
        source="test",
    )
    overlay = PageCorrectionOverlay(
        schema_id=OVERLAY_SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        kind=OVERLAY_KIND,
        page_id="p0002",
        pdf_page=2,
        orphaned=[OrphanedEdit(edit=edit, reason="block_changed", requeued=True)],
    )
    write_overlay(run / "pages" / "p0002" / "overlay.json", overlay)


def _write_matching_queue(run: Path, vault: Path) -> None:
    config = read_config(run)
    entries: list[ReviewQueueEntry] = []
    for note in sorted((vault / "content").glob("*.md")):
        for marker in parse_markers(note.read_text(encoding="utf-8")):
            entries.append(_entry(marker, len(entries)))
    write_queue(
        queue_path(run),
        ReviewQueue(
            schema_id=QUEUE_SCHEMA_ID,
            schema_version=REVIEW_SCHEMA_VERSION,
            kind=QUEUE_KIND,
            source_sha256=config.source.sha256,
            entries=entries,
        ),
    )


def _add_extra_entry(run: Path) -> None:
    queue = read_queue(queue_path(run))
    extra = queue.entries[0].model_copy(
        update={"entry_id": "extra", "start": 0, "end": 1, "raw_text": "y"}
    )
    write_queue(queue_path(run), queue.model_copy(update={"entries": [*queue.entries, extra]}))


def _entry(marker: ParsedMarker, ordinal: int) -> ReviewQueueEntry:
    page = int(marker.pdf_link.partition("#page=")[2])
    length = marker.end - marker.start
    text = render_marker(
        marker.kind, marker.pdf_link, marker.block_id, marker.start, marker.end, marker.crop
    )
    return ReviewQueueEntry(
        entry_id=f"{marker.block_id}-{marker.start}",
        page_id=f"p{page:04d}",
        pdf_page=page,
        block_id=marker.block_id,
        ordinal=ordinal,
        anchor=_HEX,
        kind=marker.kind,
        reason="unresolved",
        start=marker.start,
        end=marker.end,
        raw_text="x" * length,
        classes=[],
        marker=text,
        crop=marker.crop,
        pdf_link=marker.pdf_link,
        rendered=text,
        verdict="",
        block_present=True,
    )
