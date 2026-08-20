"""Table-aware chunking: row-aligned boundaries, header spans, offset round-trips (CI-safe)."""

import pytest

from llb.core.paths import PROJECT_ROOT
from llb.optimize.tuning_space import EXTENDED_STRATEGIES
from llb.rag.chunking.corpus import chunk_text, summarize
from llb.rag.chunking.dispatch import STRATEGIES, chunk_spans
from llb.rag.chunking.table import TABLE_HEADER_SPAN_KEY, find_tables, table_spans

# A converted-PDF style block: heading + markdown table + a prose paragraph after it.
FIXTURE = PROJECT_ROOT / "samples" / "chunking" / "goods_table_uk.md"
DOC_ID = FIXTURE.name
# The table is 480 chars, so this size forces a row-block split and this one keeps it whole.
SPLIT_SIZE = 200
WHOLE_SIZE = 1000
OVERLAP = 30


def _text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _row_line_bounds(text: str) -> set[int]:
    """Every offset that starts or ends a table ROW line in the fixture."""
    bounds: set[int] = set()
    for table in find_tables(text):
        for start, end in table.rows:
            bounds.update((start, end))
    return bounds


def _row_interiors(text: str) -> set[int]:
    """Offsets strictly INSIDE a table row -- a boundary here would cut the row."""
    interiors: set[int] = set()
    for table in find_tables(text):
        for start, end in table.rows:
            interiors.update(range(start + 1, end))
    return interiors


def test_table_strategy_is_registered_everywhere_a_strategy_is_selectable():
    assert "table" in STRATEGIES
    assert "table" in EXTENDED_STRATEGIES  # `tune --extended-chunkers`


def test_fixture_table_is_longer_than_the_split_size():
    # Guards the fixture: if the table ever shrinks below the cap the split tests go vacuous.
    tables = find_tables(_text())
    assert len(tables) == 1
    assert tables[0].end - tables[0].start > SPLIT_SIZE


@pytest.mark.parametrize("size", (SPLIT_SIZE, WHOLE_SIZE))
def test_chunks_stay_offset_exact_and_within_size(size):
    text = _text()
    chunks = chunk_text(text, DOC_ID, "table", size=size, overlap=OVERLAP)
    assert chunks
    for chunk in chunks:
        assert text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
        assert chunk["char_end"] - chunk["char_start"] <= size
    assert summarize(chunks)["oversize"] == 0


def test_no_boundary_falls_inside_a_table_row():
    text = _text()
    interiors = _row_interiors(text)
    assert interiors  # the fixture really does carry rows
    for start, end, _meta in chunk_spans(text, "table", SPLIT_SIZE, OVERLAP):
        assert start not in interiors
        assert end not in interiors


def test_a_split_table_emits_row_blocks_that_end_on_row_boundaries():
    text = _text()
    bounds = _row_line_bounds(text)
    blocks = [
        span for span in table_spans(text, SPLIT_SIZE, OVERLAP) if TABLE_HEADER_SPAN_KEY in span[2]
    ]
    assert len(blocks) > 1  # the table really did split
    for start, end, _meta in blocks:
        assert start in bounds and end in bounds


def test_a_table_that_fits_stays_one_chunk_with_its_heading_breadcrumb():
    text = _text()
    table = find_tables(text)[0]
    blocks = [
        span for span in table_spans(text, WHOLE_SIZE, OVERLAP) if TABLE_HEADER_SPAN_KEY in span[2]
    ]
    assert len(blocks) == 1
    start, end, meta = blocks[0]
    assert (start, end) == (table.start, table.end)
    assert meta["headers"] == {"h1": "Товарна номенклатура молочної групи"}


def test_every_table_chunk_names_its_header_row():
    text = _text()
    table = find_tables(text)[0]
    blocks = [
        span for span in table_spans(text, SPLIT_SIZE, OVERLAP) if TABLE_HEADER_SPAN_KEY in span[2]
    ]
    for _start, _end, meta in blocks:
        header_start, header_end = meta[TABLE_HEADER_SPAN_KEY]
        assert (header_start, header_end) == table.header
        assert "|" in text[header_start:header_end]  # resolves to the real header row


def test_table_chunks_carry_no_copied_header_text():
    # The header is recorded as OFFSETS precisely so chunk text stays a verbatim corpus slice:
    # only the block that physically contains the header row may show the header text.
    text = _text()
    header_start, header_end = find_tables(text)[0].header
    header_text = text[header_start:header_end]
    for chunk in chunk_text(text, DOC_ID, "table", size=SPLIT_SIZE, overlap=OVERLAP):
        contains_header = chunk["char_start"] <= header_start and chunk["char_end"] >= header_end
        assert (header_text in chunk["text"]) == contains_header


def test_every_non_whitespace_character_reaches_a_chunk():
    text = _text()
    covered = {
        offset
        for chunk in chunk_text(text, DOC_ID, "table", size=SPLIT_SIZE, overlap=OVERLAP)
        for offset in range(chunk["char_start"], chunk["char_end"])
    }
    assert not [i for i, char in enumerate(text) if not char.isspace() and i not in covered]


def test_prose_regions_carry_no_table_metadata():
    text = _text()
    prose = [span for span in table_spans(text, SPLIT_SIZE, OVERLAP) if not span[2]]
    assert prose  # the heading paragraph and the closing paragraph
    assert all("|" not in text[start:end] for start, end, _ in prose)


def test_a_child_slice_records_its_header_span_in_source_coordinates():
    # `parent_child` children re-chunk the PARENT'S text, so a header found there is
    # parent-local and has to move with the child's own offsets.
    from llb.rag.chunking.table import shifted_metadata

    from llb.rag.vector_store.build import _build_children

    parent_start = 5000
    meta = {"headers": {}, TABLE_HEADER_SPAN_KEY: [10, 40]}
    assert shifted_metadata(meta, parent_start)[TABLE_HEADER_SPAN_KEY] == [5010, 5040]
    assert shifted_metadata({"headers": {}}, parent_start) == {"headers": {}}

    text = _text()
    parent = {
        "doc_id": DOC_ID,
        "chunk_id": "p0",
        "char_start": parent_start,
        "char_end": parent_start + len(text),
        "text": text,
        "metadata": {},
    }
    children = _build_children(
        [parent], "table", child_size=SPLIT_SIZE, overlap=OVERLAP, embedder=None
    )
    header_start, header_end = find_tables(text)[0].header
    spans = {
        tuple(c["metadata"][TABLE_HEADER_SPAN_KEY])
        for c in children
        if TABLE_HEADER_SPAN_KEY in c["metadata"]
    }
    assert spans == {(parent_start + header_start, parent_start + header_end)}


# --- table detection ---------------------------------------------------------------------


def test_a_horizontal_rule_after_a_pipe_line_is_not_a_table():
    text = "Ціна | вартість\n---\n\nЗвичайний абзац тут.\n"
    assert find_tables(text) == []


def test_a_document_without_tables_chunks_like_recursive():
    text = "Перший абзац тут.\n\nДругий абзац також тут.\n"
    assert find_tables(text) == []
    assert [(s, e) for s, e, _ in table_spans(text, 1000, 0)] == [(0, len(text.rstrip()))]


def test_tables_are_found_in_source_order_with_their_rows():
    text = "| a | b |\n| --- | --- |\n| 1 | 2 |\n\nТекст.\n\n| c |\n| - |\n| 3 |\n"
    tables = find_tables(text)
    assert len(tables) == 2
    assert [len(table.rows) for table in tables] == [3, 3]
    assert text[tables[0].header[0] : tables[0].header[1]] == "| a | b |"
    assert text[tables[1].start : tables[1].end] == "| c |\n| - |\n| 3 |"


def test_each_table_carries_the_breadcrumb_of_its_own_enclosing_heading():
    text = (
        "| поза розділом |\n| --- |\n| 0 |\n\n"
        "# Розділ\n\n| a |\n| - |\n| 1 |\n\n"
        "## Підрозділ\n\n| b |\n| - |\n| 2 |\n"
    )
    crumbs = [meta["headers"] for _s, _e, meta in table_spans(text, 1000, 0) if meta]
    assert crumbs == [
        {},
        {"h1": "Розділ"},
        {"h1": "Розділ", "h2": "Підрозділ"},
    ]


def test_a_row_longer_than_size_is_the_one_row_the_cap_may_cut():
    # The documented fallback: `size` stays a hard cap even when a single row exceeds it.
    row = "| " + "довгий текст " * 20 + "|"
    text = f"| Код | Назва |\n| --- | --- |\n{row}\n"
    row_start = text.index(row)
    spans = chunk_spans(text, "table", 100, 20)
    assert all(end - start <= 100 for start, end, _ in spans)
    # the long row alone needed more than one chunk, so a boundary DID land inside it
    assert [start for start, _end, _meta in spans if row_start < start < row_start + len(row)]
