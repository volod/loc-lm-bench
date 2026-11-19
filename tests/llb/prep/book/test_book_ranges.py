"""Page-range parsing, batching, and deterministic page naming."""

import pytest

from llb.prep.book.ranges import PageRangeError, batches, page_id, parse_page_ranges


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("3", [3]),
        ("2-4", [2, 3, 4]),
        ("8-", [8, 9, 10]),
        ("-2", [1, 2]),
        ("1-", list(range(1, 11))),
        ("9, 1-2 ,5", [1, 2, 5, 9]),
    ],
)
def test_parse_page_ranges_accepts_single_ranges_and_open_ends(spec, expected):
    assert parse_page_ranges(spec, 10) == expected


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ("1-5,4-6", "'1-5' and '4-6' overlap on pages 4-5"),
        ("2,2", "'2' and '2' overlap on page 2"),
        ("1-11", "reaches page 11, past the last page 10"),
        ("12", "reaches page 12, past the last page 10"),
        ("0-3", "starts at 0; pages are 1-based"),
        ("5-3", "is reversed (5 > 3)"),
        ("a-b", "is not N, N-M, N-, or -M"),
        ("-", "is not N, N-M, N-, or -M"),
        ("1,,2", "has an empty part"),
        ("", "has an empty part"),
    ],
)
def test_parse_page_ranges_refuses_by_name(spec, fragment):
    with pytest.raises(PageRangeError) as excinfo:
        parse_page_ranges(spec, 10)
    assert fragment in str(excinfo.value)


def test_batches_are_consecutive_and_bounded():
    assert batches([1, 2, 3, 7, 8], 2) == [[1, 2], [3, 7], [8]]
    with pytest.raises(PageRangeError):
        batches([1], 0)


def test_page_ids_are_zero_padded_and_widen_with_the_document():
    assert page_id(7, 1067) == "p0007"
    assert page_id(1067, 1067) == "p1067"
    assert page_id(3, 12000) == "p00003"
