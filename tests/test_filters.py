"""Chunk-metadata filter seam (`llb.rag.filters`) -- doc/heading/page predicates."""

from llb.rag.filters import metadata_filter


def _chunk(doc="a.md", headers=None, pages=None):
    meta = {}
    if headers is not None:
        meta["headers"] = headers
    if pages is not None:
        meta["pages"] = pages
    return {"doc_id": doc, "char_start": 0, "char_end": 10, "text": "t", "metadata": meta}


def test_doc_id_filter():
    accept = metadata_filter(doc_ids={"a.md"})
    assert accept(_chunk("a.md")) is True
    assert accept(_chunk("b.md")) is False


def test_heading_filter_matches_any_breadcrumb_level_casefolded():
    accept = metadata_filter(heading_contains="оплата")
    assert accept(_chunk(headers={"h1": "Розділ", "h2": "ОПЛАТА ПРАЦІ"})) is True
    assert accept(_chunk(headers={"h1": "Інше"})) is False
    assert accept(_chunk()) is False  # no breadcrumb -> fails the heading condition


def test_page_range_filter_overlaps_inclusive():
    accept = metadata_filter(page_range=(3, 5))
    assert accept(_chunk(pages=[5, 7])) is True  # overlaps at page 5
    assert accept(_chunk(pages=[1, 2])) is False
    assert accept(_chunk()) is False  # un-paged chunk never matches a page-scoped query


def test_conditions_combine_with_and():
    accept = metadata_filter(doc_ids={"a.md"}, page_range=(1, 2))
    assert accept(_chunk("a.md", pages=[2, 3])) is True
    assert accept(_chunk("b.md", pages=[2, 3])) is False
    assert accept(_chunk("a.md", pages=[4, 5])) is False
