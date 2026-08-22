"""table-header-context-restoration -- the prompt-side rule and its character accounting.

Everything here is fixture-driven: the rule is a pure function over chunk dicts plus a document
resolver, so the whole vertical (rule, accounting, retrieve-node wiring) runs with no store, no
backend, and no GPU.
"""

import pytest

from llb.eval import common as eval_common
from llb.eval import graph
from llb.eval.table_headers import (
    HEADER_SEPARATOR,
    corpus_doc_text,
    corpus_header_restorer,
    header_span,
    restore_headers,
    restored_chunk,
)
from llb.rag.chunking.corpus import chunk_text
from llb.rag.chunking.table import TABLE_HEADER_SPAN_KEY

DOC_ID = "goods.md"
HEADER = "| товар | ціна | одиниця |"
DELIMITER = "| --- | --- | --- |"
ROWS = [f"| товар-{i} | {100 + i} | шт |" for i in range(12)]
TABLE_DOC = "\n".join(["# Прайс", "", HEADER, DELIMITER, *ROWS, ""])


def _doc_text(mapping):
    return lambda doc_id: mapping.get(doc_id)


def _chunk(start, end, doc=TABLE_DOC, doc_id=DOC_ID, header=(None, None), **extra):
    metadata = {} if header == (None, None) else {TABLE_HEADER_SPAN_KEY: list(header)}
    return {
        "doc_id": doc_id,
        "char_start": start,
        "char_end": end,
        "text": doc[start:end],
        "metadata": metadata,
        **extra,
    }


def _table_chunks(size=200, overlap=0):
    """Real `table`-strategy chunks of the fixture document, header-carrying ones first."""
    return chunk_text(TABLE_DOC, DOC_ID, "table", size, overlap)


def _header_offsets():
    start = TABLE_DOC.index(HEADER)
    return start, start + len(HEADER)


# --- the rule -----------------------------------------------------------------------------


def test_chunk_without_a_recorded_span_is_left_alone():
    chunk = _chunk(0, 20)
    assert header_span(chunk) is None
    assert restored_chunk(chunk, TABLE_DOC) is None


def test_row_block_gets_its_header_prepended_and_offsets_stay_put():
    span = _header_offsets()
    body_start = TABLE_DOC.index(ROWS[6])
    chunk = _chunk(body_start, body_start + len(ROWS[6]), header=span)
    restored = restored_chunk(chunk, TABLE_DOC)
    assert restored is not None
    assert restored["text"] == f"{HEADER}{HEADER_SEPARATOR}{ROWS[6]}"
    # the stored record is untouched: same offsets, same text, and a different object
    assert restored is not chunk
    assert (restored["char_start"], restored["char_end"]) == (body_start, body_start + len(ROWS[6]))
    assert chunk["text"] == ROWS[6]


def test_block_that_already_carries_the_header_is_skipped():
    span = _header_offsets()
    chunk = _chunk(span[0], TABLE_DOC.index(ROWS[2]), header=span)
    assert HEADER in chunk["text"]
    assert restored_chunk(chunk, TABLE_DOC) is None


def test_header_text_repeated_inside_a_later_block_is_skipped():
    """A repeated header row is already readable, so restoring it would only duplicate it."""
    doc = TABLE_DOC + "\n" + HEADER + "\n"
    span = _header_offsets()
    start = doc.rindex(HEADER)
    chunk = _chunk(start, start + len(HEADER), doc=doc, header=span)
    assert restored_chunk(chunk, doc) is None


def test_document_that_does_not_reproduce_the_chunk_restores_nothing():
    """The guard that makes reading a header by offset safe: a drifted corpus is refused."""
    span = _header_offsets()
    body_start = TABLE_DOC.index(ROWS[6])
    chunk = _chunk(body_start, body_start + len(ROWS[6]), header=span)
    drifted = "x" + TABLE_DOC
    assert restored_chunk(chunk, drifted) is None


def test_blank_recorded_span_restores_nothing():
    blank = TABLE_DOC.index("\n\n")
    body_start = TABLE_DOC.index(ROWS[6])
    chunk = _chunk(body_start, body_start + len(ROWS[6]), header=(blank, blank + 1))
    assert restored_chunk(chunk, TABLE_DOC) is None


@pytest.mark.parametrize("span", [None, "nope", [1], [1, 2, 3], ["a", "b"], [5, 5], [-1, 4]])
def test_malformed_recorded_span_is_not_a_span(span):
    chunk = _chunk(0, 5)
    chunk["metadata"] = {} if span is None else {TABLE_HEADER_SPAN_KEY: span}
    assert header_span(chunk) is None


# --- accounting ---------------------------------------------------------------------------


def test_accounting_counts_only_the_chunks_actually_restored():
    span = _header_offsets()
    header_block = _chunk(span[0], TABLE_DOC.index(ROWS[2]), header=span)
    middle = TABLE_DOC.index(ROWS[6])
    row_block = _chunk(middle, middle + len(ROWS[6]), header=span)
    prose = _chunk(0, 7)
    result = restore_headers([header_block, row_block, prose], _doc_text({DOC_ID: TABLE_DOC}))
    assert result.restored == 1
    assert result.added_chars == len(HEADER) + len(HEADER_SEPARATOR)
    assert [chunk["text"] for chunk in result.chunks] == [
        header_block["text"],
        f"{HEADER}{HEADER_SEPARATOR}{ROWS[6]}",
        prose["text"],
    ]


def test_nothing_restored_returns_the_input_list_itself():
    chunks = [_chunk(0, 7)]
    result = restore_headers(chunks, _doc_text({DOC_ID: TABLE_DOC}))
    assert result.chunks is chunks
    assert (result.restored, result.added_chars) == (0, 0)


def test_unreadable_document_restores_nothing():
    span = _header_offsets()
    middle = TABLE_DOC.index(ROWS[6])
    result = restore_headers([_chunk(middle, middle + len(ROWS[6]), header=span)], _doc_text({}))
    assert (result.restored, result.added_chars) == (0, 0)


def test_a_span_carrying_chunk_after_a_plain_one_still_resolves_its_document():
    """The per-document cache must key on the document, never on the first chunk seen for it."""
    span = _header_offsets()
    middle = TABLE_DOC.index(ROWS[6])
    chunks = [_chunk(0, 7), _chunk(middle, middle + len(ROWS[6]), header=span)]
    result = restore_headers(chunks, _doc_text({DOC_ID: TABLE_DOC}))
    assert result.restored == 1


# --- against the real chunker ---------------------------------------------------------------


def test_real_table_chunks_restore_every_block_but_the_first():
    chunks = _table_chunks()
    with_spans = [chunk for chunk in chunks if header_span(chunk) is not None]
    assert len(with_spans) > 1, "the fixture table must split into several blocks"
    result = restore_headers(chunks, _doc_text({DOC_ID: TABLE_DOC}))
    assert result.restored == len(with_spans) - 1
    for original, prompt in zip(chunks, result.chunks):
        assert prompt["char_start"] == original["char_start"]
        assert prompt["char_end"] == original["char_end"]
        assert HEADER in prompt["text"] or header_span(original) is None


def test_corpus_resolver_reads_each_document_once(tmp_path):
    (tmp_path / DOC_ID).write_text(TABLE_DOC, encoding="utf-8")
    resolve = corpus_doc_text(tmp_path)
    assert resolve(DOC_ID) == TABLE_DOC
    (tmp_path / DOC_ID).unlink()
    assert resolve(DOC_ID) == TABLE_DOC  # cached, not re-read
    assert resolve("missing.md") is None


# --- the retrieve node -----------------------------------------------------------------------


class _Store:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, question, k, chunk_filter=None):
        return self._chunks[:k]


def _retrieve(chunks, restorer=None):
    node = graph.make_retrieve_node(_Store(chunks), k=10, header_restorer=restorer)
    return node({"question": "скільки коштує товар-6?"})


def test_retrieve_node_records_zero_accounting_when_the_step_is_off():
    chunks = _table_chunks()
    state = _retrieve(chunks)
    assert (state["table_headers_restored"], state["table_header_chars"]) == (0, 0)
    assert "prompt_chunks" not in state
    assert state["context"] == eval_common.format_context(chunks)


def test_retrieve_node_restores_the_prompt_only(tmp_path):
    (tmp_path / DOC_ID).write_text(TABLE_DOC, encoding="utf-8")
    chunks = _table_chunks()
    state = _retrieve(chunks, corpus_header_restorer(tmp_path))
    assert state["table_headers_restored"] > 0
    assert state["table_header_chars"] > 0
    # retrieval is untouched: the stored records still reproduce a fresh chunking of the source
    assert state["retrieved"] == chunks
    assert chunks == _table_chunks()
    # the prompt is not
    assert state["context"] != eval_common.format_context(chunks)
    assert state["context"].count(HEADER) > 1
    assert state["prompt_chunks"] is not chunks
