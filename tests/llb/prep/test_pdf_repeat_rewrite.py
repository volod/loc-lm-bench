"""Focused tests split from ``test_pdf_repeats.py``."""

from _pdf_repeats_helpers import (
    FIXTURE_ANCHORED_BLOCKS,
    FIXTURE_BLOCKS,
    FIXTURE_DROPPED_BLOCKS,
    FIXTURE_GROUPS,
    FIXTURE_LARGEST,
    PROCEDURE,
    SUPPORT,
    fixture_text,
)

from llb.prep.pdf.repeats import (
    REPEAT_ANCHOR,
    REPEAT_DROP,
    REPEAT_KEEP,
    heading_breadcrumb,
    rewrite_repeated_blocks,
)
from llb.prep.pdf.repeat_spans import (
    remap_span,
    remap_span_split,
)


def test_census_reports_the_planted_intra_document_repetition() -> None:
    census = rewrite_repeated_blocks(fixture_text(), mode=REPEAT_KEEP).census

    assert census["blocks"] == FIXTURE_BLOCKS
    assert (census["groups"], census["largest_group"]) == (FIXTURE_GROUPS, FIXTURE_LARGEST)
    assert census["handled_groups"] == FIXTURE_GROUPS
    assert census["handled_blocks"] == 0  # `keep` measures, it never rewrites


def test_keep_mode_leaves_the_document_byte_identical() -> None:
    text = fixture_text()

    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_KEEP)

    assert rewrite.text == text
    assert rewrite.edits == []


def test_drop_keeps_the_first_copy_and_removes_the_rest() -> None:
    text = fixture_text()

    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_DROP)

    assert rewrite.census["handled_blocks"] == FIXTURE_DROPPED_BLOCKS
    assert rewrite.text.count(PROCEDURE) == 1
    assert rewrite.text.count(SUPPORT) == 1
    assert len(rewrite.text) < len(text)
    # nothing is rewritten, only removed: every surviving line is still the corpus's own text
    assert all(line in text for line in rewrite.text.splitlines() if line.strip())


def test_drop_leaves_repeated_table_headers_and_headings_alone() -> None:
    """Structure is not furniture: a repeated table header makes the tables under it readable."""
    rewrite = rewrite_repeated_blocks(fixture_text(), mode=REPEAT_DROP)

    assert rewrite.text.count("|**Поле**|**Опис**|") == 2


def test_anchor_keeps_every_copy_and_makes_it_distinct() -> None:
    rewrite = rewrite_repeated_blocks(fixture_text(), mode=REPEAT_ANCHOR)

    assert rewrite.census["handled_blocks"] == FIXTURE_ANCHORED_BLOCKS
    assert rewrite.text.count(PROCEDURE) == FIXTURE_LARGEST
    anchored = [line for line in rewrite.text.splitlines() if line.startswith("> ")]
    assert len(anchored) == FIXTURE_ANCHORED_BLOCKS
    assert len(set(anchored)) == FIXTURE_LARGEST  # one anchor per section, not one per document
    assert "Розділ 2. Переміщення майна" in anchored[2]


def test_heading_breadcrumb_skips_the_rendered_document_title() -> None:
    text = "# Source PDF: a.pdf\n\n## Розділ 1\n\n### Крок\n\nтекст\n"

    assert heading_breadcrumb(text, text.index("текст")) == "Розділ 1 > Крок"
    assert heading_breadcrumb("текст", 0) == ""


def test_remap_span_follows_a_dropped_copy_onto_the_survivor() -> None:
    text = fixture_text()
    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_DROP)
    third = text.index(PROCEDURE, text.index(PROCEDURE, text.index(PROCEDURE) + 1) + 1)

    moved = remap_span(rewrite.edits, third, third + len(PROCEDURE))

    assert moved is not None
    assert rewrite.text[moved[0] : moved[1]] == PROCEDURE
    assert moved[0] == rewrite.text.index(PROCEDURE)  # the one surviving copy


def test_remap_span_refuses_a_span_straddling_a_rewrite() -> None:
    text = fixture_text()
    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_DROP)
    second = text.index(PROCEDURE, text.index(PROCEDURE) + 1)

    assert remap_span(rewrite.edits, second - 40, second + len(PROCEDURE)) is None


def test_remap_span_split_recovers_a_straddler_as_two_pieces() -> None:
    text = fixture_text()
    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_DROP)
    second = text.index(PROCEDURE, text.index(PROCEDURE) + 1)
    start, end = second - 40, second + len(PROCEDURE)  # kept tail + dropped procedure head

    images = remap_span_split(rewrite.edits, start, end)

    assert images is not None and len(images) == 2
    # in order, the two stripped images reconstruct the original span exactly
    assert "".join(rewrite.text[lo:hi] for lo, hi in images) == text[start:end]
    # the second piece landed on the surviving copy of the procedure block
    assert rewrite.text[images[1][0] : images[1][1]] in text[second : second + len(PROCEDURE)]


def test_remap_span_split_is_a_single_piece_for_a_clean_span() -> None:
    text = fixture_text()
    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_DROP)
    unique = "Списання майна виконується комісією"
    start = text.index(unique)

    images = remap_span_split(rewrite.edits, start, start + len(unique))

    assert images is not None and len(images) == 1


def test_every_untouched_span_keeps_its_text_under_both_modes() -> None:
    """The offset map is exact: an unrelated span still reads as itself after the rewrite."""
    text = fixture_text()
    unique = "Списання майна виконується комісією"
    start = text.index(unique)

    for mode in (REPEAT_DROP, REPEAT_ANCHOR):
        rewrite = rewrite_repeated_blocks(text, mode=mode)
        moved = remap_span(rewrite.edits, start, start + len(unique))
        assert moved is not None
        assert rewrite.text[moved[0] : moved[1]] == unique
