"""A two-column page with a table, an aside, and a page continuation.

The coordinates are the rendered page. Nothing here is a scan of a real book.
"""

from llb.prep.book.engines.protocol import EngineRequest, Proposed

WIDTH = 800
HEIGHT = 1000

# kind, order, x0, y0, x1, y1, text
Row = tuple[str, int, float, float, float, float, str]

LAYOUT: dict[str, tuple[Row, ...]] = {
    "p0001": (
        ("heading", 0, 40, 30, 760, 70, "Span Title"),
        ("body", 1, 40, 90, 360, 180, "Left alpha begins and holds."),
        ("neighbour-page-bleed", 2, 0, 140, 22, 190, "NEIGHBOUR BLEED"),
        ("body", 3, 40, 200, 360, 300, "Left beta ends with neces-"),
        ("body", 4, 480, 200, 760, 280, "sary word follows."),
        ("body", 5, 480, 270, 760, 360, "Right alpha stays apart."),
        ("aside", 6, 180, 400, 620, 480, "BOXED EXAMPLE stays apart."),
        ("body", 7, 40, 520, 429, 640, "Left gamma holds."),
        ("body", 8, 429, 520, 760, 640, "Right gamma holds."),
        ("table", 9, 40, 680, 760, 760, "speed | 9.81"),
        ("caption", 10, 40, 780, 400, 840, "Table 1-1 Planted caption."),
        ("body", 11, 40, 860, 360, 960, "The discussion continua-"),
        ("body", 12, 20, 80, 780, 980, "INTERLEAVED left and right"),
    ),
    "p0002": (
        ("body", 0, 40, 80, 360, 160, "tion from the next page."),
        ("body", 1, 40, 200, 360, 300, "A fresh paragraph starts."),
    ),
}

LINES: dict[str, tuple[Row, ...]] = {
    "p0001": (
        ("body", 0, 500, 280, 740, 330, "Right alpha stays apart."),
        ("body", 1, 50, 100, 340, 150, "Left alpha begins and holds."),
        ("body", 2, 520, 390, 700, 420, "STRAY LINE"),
        ("body", 3, 50, 210, 340, 280, "Left beta ends with neces-"),
        ("body", 4, 500, 210, 740, 260, "sary word follows."),
        ("unknown", 5, 200, 420, 600, 460, "BOXED EXAMPLE stays apart."),
        ("body", 6, 80, 700, 200, 740, "9.81"),
        ("caption", 7, 50, 790, 380, 830, "Table 1-1 Planted caption."),
        ("heading", 8, 60, 35, 700, 65, "Span Title"),
        ("neighbour-page-bleed", 9, 2, 145, 18, 180, "NEIGHBOUR BLEED"),
        ("body", 10, 50, 880, 340, 940, "The discussion continua-"),
        ("body", 11, 329, 560, 529, 600, "WHICH WAY"),
    ),
    "p0002": (
        ("body", 0, 50, 90, 340, 140, "tion from the next page."),
        ("body", 1, 50, 210, 340, 280, "A fresh paragraph starts."),
        ("body", 2, 360, 940, 400, 970, "2"),
    ),
}


def page_key(request: EngineRequest) -> str:
    """The rendered page the engine was handed, from its path."""
    text = request.image.as_posix()
    return "p0002" if "/p0002/" in text else "p0001"


def proposed(rows: tuple[Row, ...]) -> tuple[Proposed, ...]:
    """Proposals in the planted pixel space."""
    return tuple(
        Proposed(kind, order, x0, y0, x1, y1, text, None)
        for kind, order, x0, y0, x1, y1, text in rows
    )


def scaled(request: EngineRequest, rows: tuple[Row, ...]) -> tuple[Proposed, ...]:
    """The same boxes on the image the stage actually rendered."""
    scale_x = request.width / WIDTH
    scale_y = request.height / HEIGHT
    return tuple(
        Proposed(
            kind,
            order,
            x0 * scale_x,
            y0 * scale_y,
            x1 * scale_x,
            y1 * scale_y,
            text,
            None,
        )
        for kind, order, x0, y0, x1, y1, text in rows
    )
