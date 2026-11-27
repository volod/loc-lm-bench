"""A monotone fit from PDF pages to printed pages.

Observations are the numerals furniture picked. A step is kept when the printed number
advances by at most the number of PDF pages. An interior page with no numeral is a gap
when the printed numbers kept pace, and an unnumbered plate when they did not. Pages
before either series are front matter. The fit does not invent a number past the series
except for the single page touching each end of a pure offset.
"""

from collections import defaultdict
from dataclasses import dataclass

from llb.core.contracts.data_prep.book_structure import NumberScheme, PageStatus, PrintedPage
from llb.prep.book.ranges import page_id
from llb.prep.book.structure.furniture import PageHit
from llb.prep.book.structure.text import int_to_roman


@dataclass(frozen=True)
class _Obs:
    pdf: int
    value: int
    block_id: str


@dataclass(frozen=True)
class _Assigned:
    value: int | None
    status: PageStatus
    block_id: str | None
    fallback: bool


def printed_pages(pages: list[int], hits: list[PageHit], page_count: int) -> list[PrintedPage]:
    """The printed-page mapping for every configured PDF page, in order."""
    arabic = _series([_obs(hit) for hit in hits if hit.scheme == "arabic"], pages)
    roman = _series([_obs(hit) for hit in hits if hit.scheme == "roman"], pages)
    _extend_edges(arabic, roman, pages)
    return [_printed(pdf, arabic, roman, page_count) for pdf in pages]


def ambiguous_folios(hits: list[PageHit]) -> list[str]:
    """Keep duplicate observations even when only one fits the monotone sequence."""
    claimants: dict[tuple[str, int], set[int]] = defaultdict(set)
    for hit in hits:
        claimants[hit.scheme, hit.value].add(hit.pdf)
    return [
        int_to_roman(value) if scheme == "roman" else str(value)
        for (scheme, value), pdfs in sorted(claimants.items())
        if len(pdfs) > 1
    ]


def _obs(hit: PageHit) -> _Obs:
    return _Obs(hit.pdf, hit.value, hit.block_id)


def _series(obs: list[_Obs], pages: list[int]) -> dict[int, _Assigned]:
    chain = _chain(obs)
    assigned = {item.pdf: _Assigned(item.value, "observed", item.block_id, False) for item in chain}
    for left, right in zip(chain, chain[1:]):
        assigned.update(_interval(left, right, pages))
    return assigned


def _chain(obs: list[_Obs]) -> list[_Obs]:
    ordered = sorted(obs, key=lambda item: (item.pdf, item.value))
    count = len(ordered)
    if count == 0:
        return []
    score = [1] * count
    exact = [0] * count
    previous = [-1] * count
    for end in range(count):
        _relax(ordered, score, exact, previous, end)
    best = max(range(count), key=lambda index: (score[index], exact[index], index))
    return _rebuild(ordered, previous, best)


def _relax(
    ordered: list[_Obs],
    score: list[int],
    exact: list[int],
    previous: list[int],
    end: int,
) -> None:
    for start in range(end):
        if ordered[start].pdf >= ordered[end].pdf:
            continue
        d_pdf = ordered[end].pdf - ordered[start].pdf
        d_value = ordered[end].value - ordered[start].value
        if d_value < 1 or d_value > d_pdf:
            continue
        new_score = score[start] + 1
        new_exact = exact[start] + int(d_value == d_pdf)
        if (new_score, new_exact) <= (score[end], exact[end]):
            continue
        score[end] = new_score
        exact[end] = new_exact
        previous[end] = start


def _rebuild(ordered: list[_Obs], previous: list[int], end: int) -> list[_Obs]:
    chain: list[_Obs] = []
    cursor = end
    while cursor >= 0:
        chain.append(ordered[cursor])
        cursor = previous[cursor]
    chain.reverse()
    return chain


def _interval(left: _Obs, right: _Obs, pages: list[int]) -> dict[int, _Assigned]:
    interior = [pdf for pdf in pages if left.pdf < pdf < right.pdf]
    if not interior:
        return {}
    unnumbered = (right.pdf - left.pdf) - (right.value - left.value)
    numbered = (right.pdf - left.pdf - 1) - unnumbered
    if unnumbered == 0:
        return {pdf: _gap(left.value + (pdf - left.pdf)) for pdf in interior}
    if numbered == 0:
        return {pdf: _plate() for pdf in interior}
    return {pdf: _unsure() for pdf in interior}


def _gap(value: int) -> _Assigned:
    return _Assigned(value, "gap", None, True)


def _plate() -> _Assigned:
    return _Assigned(None, "unnumbered", None, False)


def _unsure() -> _Assigned:
    return _Assigned(None, "unnumbered", None, True)


def _extend_edges(
    arabic: dict[int, _Assigned],
    roman: dict[int, _Assigned],
    pages: list[int],
) -> None:
    """One inferred page at each end of a pure offset, never across the other series."""
    first_arabic = min(arabic) if arabic else None
    last_roman = max(roman) if roman else None
    roman_allow = {pdf for pdf in pages if first_arabic is None or pdf < first_arabic}
    arabic_allow = {pdf for pdf in pages if last_roman is None or pdf > last_roman}
    _extend(roman, roman_allow)
    _extend(arabic, arabic_allow)


def _extend(assigned: dict[int, _Assigned], allow: set[int]) -> None:
    observed = [pdf for pdf, item in assigned.items() if item.status == "observed"]
    if len(observed) < 2:
        return
    observed.sort()
    offset = assigned[observed[0]].value
    if offset is None or not _pure(assigned, observed):
        return
    offset -= observed[0]
    for edge in (observed[0] - 1, observed[-1] + 1):
        if edge not in allow or edge in assigned:
            continue
        predicted = edge + offset
        if predicted >= 1:
            assigned[edge] = _gap(predicted)


def _pure(assigned: dict[int, _Assigned], observed: list[int]) -> bool:
    for left, right in zip(observed, observed[1:]):
        left_value = assigned[left].value
        right_value = assigned[right].value
        if left_value is None or right_value is None:
            return False
        if right_value - left_value != right - left:
            return False
    return True


def _printed(
    pdf: int,
    arabic: dict[int, _Assigned],
    roman: dict[int, _Assigned],
    page_count: int,
) -> PrintedPage:
    chosen, series = _choose(pdf, arabic, roman)
    scheme: NumberScheme = series if chosen.status in {"observed", "gap"} else "none"
    return PrintedPage(
        pdf_page=pdf,
        page_id=page_id(pdf, page_count),
        printed=_label(chosen, series),
        scheme=scheme,
        status=chosen.status,
        block_id=chosen.block_id,
        fallback=chosen.fallback,
    )


def _choose(
    pdf: int, arabic: dict[int, _Assigned], roman: dict[int, _Assigned]
) -> tuple[_Assigned, NumberScheme]:
    first_arabic = min(arabic) if arabic else None
    first_roman = min(roman) if roman else None
    last_roman = max(roman) if roman else None
    if first_arabic is not None and pdf >= first_arabic and pdf in arabic:
        return arabic[pdf], "arabic"
    if first_roman is not None and first_roman <= pdf <= (last_roman or pdf) and pdf in roman:
        return roman[pdf], "roman"
    if _before(pdf, first_roman, first_arabic):
        return _Assigned(None, "front-matter", None, False), "none"
    if arabic or roman:
        return _unsure(), "none"
    return _unsure(), "none"


def _before(pdf: int, first_roman: int | None, first_arabic: int | None) -> bool:
    bounds = [bound for bound in (first_roman, first_arabic) if bound is not None]
    return bool(bounds) and pdf < min(bounds)


def _label(assigned: _Assigned, scheme: str) -> str | None:
    if assigned.value is None:
        return None
    if scheme == "roman":
        return int_to_roman(assigned.value)
    return str(assigned.value)
