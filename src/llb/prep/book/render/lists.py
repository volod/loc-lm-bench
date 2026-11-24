"""Lists of figures, tables, and equations, one file per kind.

Each row is one inventory item and links to that item's own anchor. Two
figures that share a number both appear, distinguished by the page they were
read on. A prose mention of that number stays plain: the list is not a guess
about which one the sentence meant.
"""

from llb.prep.book.render.catalog import Catalog, ListRow
from llb.prep.book.render.constants import CONTENT_DIRNAME, LIST_FILENAMES


def render_lists(catalog: Catalog) -> dict[str, str]:
    """Relative path to one Markdown list per figure, table, and equation kind."""
    return {
        filename: _page(title, catalog.rows.get(kind, ()))
        for kind, filename, title in LIST_FILENAMES
    }


def _page(title: str, rows: tuple[ListRow, ...]) -> str:
    body = "None." if not rows else "\n".join(_item(row) for row in rows)
    return f"# {title}\n\n{body}\n"


def _item(row: ListRow) -> str:
    href = f"{CONTENT_DIRNAME}/{row.stem}.md#^{row.anchor}"
    printed = f", printed {row.printed}" if row.printed else ""
    return f"- [{row.number}]({href}){printed}, PDF {row.pdf_page}"
