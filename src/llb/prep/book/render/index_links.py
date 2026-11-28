"""Page tails inside an index note: comma lists and dot leaders.

Imported by ``link`` only while linking an index note, so the two modules do
not import each other at load time.
"""

from llb.prep.book.render.catalog import Catalog
from llb.prep.book.render.grammar import page_token
from llb.prep.book.render.index_tails import IndexSegment, index_line_parts
from llb.prep.book.render.link import (
    Replacement,
    Resolve,
    _blocked,
    _page_resolver,
    _rewrite,
    place_cited,
)

_PAGE_TOKEN = page_token()


def index_replacements(
    text: str, blocked: list[tuple[int, int]], here: str, catalog: Catalog
) -> list[Replacement]:
    """One replacement per page token on an index line."""
    found: list[Replacement] = []
    resolve = _page_resolver(here, catalog, "index")
    for line in index_line_parts(text):
        found.extend(_index_line(line, blocked, resolve))
    return found


def _index_line(
    line: IndexSegment, blocked: list[tuple[int, int]], resolve: Resolve
) -> list[Replacement]:
    found: list[Replacement] = []
    offset = line.offset
    for start, end in line.pages:
        if _blocked(offset + start, offset + end, blocked):
            continue
        rewritten, hits = _rewrite(line.body[start:end], _PAGE_TOKEN, resolve, "index")
        if hits:
            found.append(
                Replacement(
                    offset + start,
                    offset + end,
                    rewritten,
                    tuple(hit.event for hit in hits),
                    place_cited(hits, offset + start),
                )
            )
    return found
