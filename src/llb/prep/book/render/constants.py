"""Names and limits for the Obsidian reading vault.

The page-link fragment is the one Obsidian documents. The 1.15 help
(obsidian.md/help/pdf and help/embeds) still describes ``#page=N`` on a PDF
link, the form the 0.10.8 release notes introduced on 2021-01-11:
``[[file.pdf#page=3]]`` opens page 3, counted from 1. Confirming that the app
honors the fragment is a later human task.
"""

from typing import Final

VAULT_DIRNAME: Final = "book"
TOC_FILENAME: Final = "00_Table_of_Contents.md"
CONTENT_DIRNAME: Final = "content"
ASSETS_DIRNAME: Final = "assets"
REVIEW_DIRNAME: Final = "review"
METADATA_DIRNAME: Final = "metadata"
SOURCE_DIRNAME: Final = "source"
SOURCE_PAGES_DIRNAME: Final = "pages"
RENDER_MANIFEST_FILENAME: Final = "render.json"
REFERENCE_FILENAME: Final = "reference.txt"
REFERENCE_REPORT_FILENAME: Final = "reference_links.json"
# Generated lists sit beside the contents, outside the reading-order chain.
LIST_FILENAMES: Final = (
    ("figure", "01_List_of_Figures.md", "Figures"),
    ("table", "02_List_of_Tables.md", "Tables"),
    ("equation", "03_List_of_Equations.md", "Equations"),
)

OBSIDIAN_VERSION: Final = "1.15"
PDF_PAGE_FRAGMENT: Final = "page"

# Resolved characters of one section file. A longer section becomes ordered
# page-range parts. The cap counts the resolved text, not the Markdown wrapper.
SECTION_CHAR_CAP: Final = 12000
SECTION_CAP_FLOOR: Final = 200

# A line whose left edge jumps by more than this fraction of the page width is
# the other column, so it does not continue the paragraph.
COLUMN_BREAK: Final = 0.15
# After a sentence, an indent of more than this fraction of the line box starts
# a paragraph. Scan skew is smaller than that.
INDENT_FRACTION: Final = 0.45
# A line whose top has dropped by more than this multiple of the line-box
# height is a new paragraph. Body lines overlap, so the gap between boxes is
# the wrong signal: a footnote sits one line-height below the last body line.
LINE_STEP_BREAK: Final = 1.25

FURNITURE_ROLES: Final = frozenset(
    {"running-header", "page-number", "ornament", "neighbour-page-bleed"}
)
STRUCTURAL_ROLES: Final = frozenset(
    {"formula", "table", "figure", "caption", "footnote", "heading", "page-failed", "missing-block"}
)

HEADING_LEVEL: Final = {
    "part": 1,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "page-range": 2,
    "book": 1,
}
LEVEL_ANCHOR: Final = {
    "part": "part",
    "chapter": "ch",
    "section": "sec",
    "subsection": "sub",
}
ITEM_ANCHOR: Final = {"figure": "fig", "table": "tbl", "equation": "eq"}

FILE_INDEX_WIDTH: Final = 4
ORDINAL_WIDTH: Final = 4
SLUG_LIMIT: Final = 60
FAIL_REASON_LIMIT: Final = 120
