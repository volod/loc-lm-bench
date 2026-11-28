"""Index lines read a glued column dump as one printed line per cell."""

from llb.prep.book.render.index_terms import index_lines
from llb.prep.book.render.catalog import Catalog, Target
from llb.prep.book.render.index_links import index_replacements

_DUMP = (
    "[{'type': 'text', 'content': 'orbit, 4'}][{'type': 'text', 'content': 'window .....4-6'}]\n"
)


def test_a_glued_column_is_one_entry_per_cell() -> None:
    lines = index_lines(_DUMP)
    assert [_DUMP[line.term_start : line.term_end] for line in lines] == ["orbit", "window"]
    assert [page.printed for page in lines[0].pages] == ["4"]
    assert [page.printed for page in lines[1].pages] == ["4", "5", "6"]
    assert {_DUMP[page.start : page.end] for page in lines[1].pages} == {"4-6"}


def test_a_line_that_only_mentions_a_cell_stays_whole() -> None:
    text = "see [{'type': 'text', 'content': 'orbit, 4'}] below\n"
    lines = index_lines(text)
    assert len(lines) == 1
    assert text[lines[0].term_start : lines[0].term_end].startswith("see")


def test_index_continuation_has_the_same_printed_offsets_in_both_readings() -> None:
    text = "4\norbit, 5\n6, 7-9\nCalibration record is filed.\n18\n"
    term, prose = index_lines(text)
    assert not prose.pages
    assert [page.printed for page in term.pages] == ["5", "6", "7", "8", "9"]
    assert [text[page.start : page.end] for page in term.pages] == ["5", "6", "7-9", "7-9", "7-9"]
    pages = {str(page): Target(f"p{page:04d}", "note") for page in (*range(4, 10), 18)}
    catalog = Catalog({}, frozenset(), pages, frozenset(), frozenset(), {}, frozenset())
    links = index_replacements(text, [], "index", catalog)
    cited = sorted((cue.start, cue.end) for link in links for cue in link.cited)
    assert [text[start:end] for start, end in cited] == ["5", "6", "7", "9"]
