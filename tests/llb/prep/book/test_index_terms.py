"""Index lines read a glued column dump as one printed line per cell."""

from llb.prep.book.render.index_terms import index_lines

_DUMP = (
    "[{'type': 'text', 'content': 'orbit, 4'}][{'type': 'text', 'content': 'window .....4-6'}]\n"
)


def test_a_glued_column_is_one_entry_per_cell() -> None:
    lines = index_lines(_DUMP)
    assert [_DUMP[line.term_start : line.term_end] for line in lines] == ["orbit", "window"]
    assert lines[0].pages == ("4",)
    assert lines[1].pages == ("4", "5", "6")


def test_a_line_that_only_mentions_a_cell_stays_whole() -> None:
    text = "see [{'type': 'text', 'content': 'orbit, 4'}] below\n"
    lines = index_lines(text)
    assert len(lines) == 1
    assert text[lines[0].term_start : lines[0].term_end].startswith("see")
