"""How a formula or a table is written for Obsidian and for the corpus.

A formula is emitted only when the local parser accepts it. A bare alignment
ampersand is wrapped in ``aligned``, which is the form Obsidian's MathJax
renders. A table with one header row and no merged cells is Markdown. Spans
and extra header rows are HTML. A grid HTML cannot hold becomes the page crop.
"""

import html
import re

from llb.core.contracts.data_prep.book_blocks import TableReading
from llb.prep.book.grid import html_holds

_ENV = re.compile(r"\\begin\{([A-Za-z*]+)\}.*?\\end\{\1\}", re.DOTALL)
MARKDOWN = "markdown"
HTML = "html"
IMAGE = "image"


def formula_math(text: str, number: str) -> str:
    """Display LaTeX. The printed equation number is a ``\\tag`` when the text has none."""
    inner = text.strip()
    if inner.startswith("$$") and inner.endswith("$$"):
        inner = inner[2:-2].strip()
    elif inner.startswith("\\[") and inner.endswith("\\]"):
        inner = inner[2:-2].strip()
    if number and "\\tag{" not in inner:
        inner = f"{inner} \\tag{{{number.replace('}', '')}}}"
    return f"$$\n{inner}\n$$"


def table_markdown(text: str) -> str:
    """HTML when the text is already a table, Markdown when it has pipes, else one column."""
    stripped = text.strip()
    lowered = stripped.lower()
    if "<table" in lowered and "</table>" in lowered:
        return stripped
    rows = [row.strip() for row in stripped.splitlines() if row.strip()]
    if rows and all("|" in row for row in rows):
        return "\n".join(rows)
    body = ["| |", "| --- |"]
    for row in rows or [stripped]:
        body.append("| " + row.replace("|", "\\|") + " |")
    return "\n".join(body)


def render_body(latex: str) -> str | None:
    """The math body Obsidian can render, or None when it does not parse.

    A bare alignment ampersand is wrapped before the parse check. The local
    parser does not accept ``&``, so the check sees the body with those
    ampersands removed and the export keeps the wrapped form.
    """
    body = _repair_alignment(_unwrap(latex.strip()))
    if not body or not _braces_balance(body) or _bare_ampersand(body):
        return None
    if not _parses(_without_ampersand(body)):
        return None
    return body


def table_mode(table: TableReading) -> str:
    """``markdown``, ``html``, or ``image`` for one native grid."""
    if not table.faithful or not html_holds(table):
        return IMAGE
    simple = table.header_rows == 1 and _rectangular(table) and _unmerged(table)
    return MARKDOWN if simple else HTML


def render_formula(latex: str, number: str, display: bool, crop: str | None) -> str:
    """Display or inline math. ``crop`` is the page region kept beside an uncertain formula."""
    body = render_body(latex)
    shown = body if body is not None else _unwrap(latex.strip())
    if display:
        math = formula_math(shown, number)
    elif number and "\\tag{" not in shown:
        math = f"${shown}$ ({number})"
    else:
        math = f"${shown}$"
    if not crop:
        return math
    return f"{math}\n\n![[{crop}]]"


def render_table(
    table: TableReading,
    image: str | None,
    evidence: str | None,
    *,
    emit_caption: bool,
) -> str:
    """The table export. An unfaithful grid keeps the image and does not invent cells."""
    parts: list[str] = []
    if emit_caption and table.caption:
        parts.append(table.caption)
    mode = table_mode(table)
    if mode == IMAGE:
        if image:
            parts.append(f"![[{image}]]")
    elif mode == MARKDOWN:
        parts.append(_markdown(table))
    else:
        parts.append(_html(table))
    if evidence:
        parts.append(f"![[{evidence}]]")
    return "\n\n".join(parts)


def _unwrap(text: str) -> str:
    if text.startswith("$$") and text.endswith("$$"):
        return text[2:-2].strip()
    if text.startswith(r"\[") and text.endswith(r"\]"):
        return text[2:-2].strip()
    return text


def _repair_alignment(body: str) -> str:
    if "&" not in body or r"\begin{" in body:
        return body
    if _parses(_without_ampersand(body)):
        return r"\begin{aligned}" + body + r"\end{aligned}"
    return body


def _parses(body: str) -> bool:
    # Imported here so loading this module does not pull the metrics package.
    from llb.prep.book.metrics.latex import latex_parses

    return latex_parses(body)


def _without_ampersand(body: str) -> str:
    return body.replace(r"\&", "\x00").replace("&", "").replace("\x00", r"\&")


def _bare_ampersand(body: str) -> bool:
    return "&" in _ENV.sub(" ", body).replace(r"\&", "")


def _braces_balance(body: str) -> bool:
    depth = 0
    escaped = False
    for char in body:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _rectangular(table: TableReading) -> bool:
    width = len(table.rows[0])
    return all(len(row) == width for row in table.rows)


def _unmerged(table: TableReading) -> bool:
    return all(cell.colspan == 1 and cell.rowspan == 1 for row in table.rows for cell in row)


def _markdown(table: TableReading) -> str:
    width = max(len(row) for row in table.rows)
    lines: list[str] = []
    for index, row in enumerate(table.rows):
        cells = [_pipes(cell.text) for cell in row]
        cells.extend("" for _ in range(width - len(cells)))
        lines.append("| " + " | ".join(cells) + " |")
        if index == 0:
            lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    return "\n".join(lines)


def _html(table: TableReading) -> str:
    rows: list[str] = []
    for index, row in enumerate(table.rows):
        tag = "th" if index < table.header_rows else "td"
        cells = "".join(_html_cell(tag, cell.text, cell.colspan, cell.rowspan) for cell in row)
        rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _html_cell(tag: str, text: str, colspan: int, rowspan: int) -> str:
    attrs = ""
    if colspan > 1:
        attrs += f' colspan="{colspan}"'
    if rowspan > 1:
        attrs += f' rowspan="{rowspan}"'
    return f"<{tag}{attrs}>{html.escape(text)}</{tag}>"


def _pipes(text: str) -> str:
    return text.replace("|", r"\|")
