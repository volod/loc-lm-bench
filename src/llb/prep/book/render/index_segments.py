"""Index lines, including a column dump glued into one recognition line.

Some recognizers emit an index column as Python cells
``[{'type': '...', 'content': '...'}]`` joined end to end. Each cell's content
is one printed line. A line that is not only those cells stays one line.
"""

import re

_CELL = re.compile(r"\[\{'type': '(?P<kind>[^']*)', 'content': '(?P<content>(?:\\'|[^'])*)'\}\]")


def index_segments(text: str) -> tuple[tuple[int, str], ...]:
    """``(offset, line)`` pairs. A glued column dump yields one pair per cell."""
    found: list[tuple[int, str]] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        body = raw[:-1] if raw.endswith("\n") else raw
        found.extend(_segments(body, offset))
        offset += len(raw)
    return tuple(found)


def _segments(body: str, offset: int) -> tuple[tuple[int, str], ...]:
    cells = _cells(body, offset)
    if cells is None:
        return ((offset, body),)
    return cells


def _cells(body: str, offset: int) -> tuple[tuple[int, str], ...] | None:
    if "'content':" not in body or not body.lstrip().startswith("[{"):
        return None
    matches = list(_CELL.finditer(body))
    if not matches or _CELL.sub("", body).strip():
        return None
    return tuple((offset + match.start("content"), match.group("content")) for match in matches)
