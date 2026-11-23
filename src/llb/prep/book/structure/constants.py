"""Thresholds for inferring book structure from block positions.

The bands are fractions of the rendered page. On a 150 dpi handbook scan the printed
page number and the running header sit in the top 9 percent, and body text starts
below that. A block's kind label is not an input: a line engine marks every block body.
"""

from typing import Final

# Vertical bands, as a fraction of the page. Running headers sit in the top band.
# A printed page number can sit a little lower, still in the outer corner, while the
# first line of body text is already below the header band. One shared cutoff either
# misses those page numbers or reads the first body line as a header.
HEADER_BAND: Final = 0.082
PAGE_NUMBER_BAND: Final = 0.10
FOOTER_BAND: Final = 0.90
OUTER_MARGIN: Final = 0.22
CENTER_LOW: Final = 0.35
CENTER_HIGH: Final = 0.65
NARROW_WIDTH: Final = 0.18

# A header band is real once this many pages carry a short line there. A text that
# repeats on this many pages is a running header even in a shorter band.
HEADER_MIN_PAGES: Final = 3
HEADER_REPEAT: Final = 2
HEADER_MAX_CHARS: Final = 90
HEADER_MAX_WORDS: Final = 14

# Contents lines that share a baseline land within this fraction of the page height.
ROW_Y_FRACTION: Final = 0.014
COLUMN_CUT: Final = 0.50
COLUMN_MIN_LINES: Final = 3
# A bare integer to the right of this fraction of the column is a printed page, not a
# chapter number.
PAGE_NUMBER_X: Final = 0.35
# A following page stays a contents page while at least this fraction of its lines
# still look like contents entries. The last contents line can sit below the footer
# band, so the density count stops only at the bottom edge.
CONTENTS_DENSITY: Final = 0.20
CONTENTS_BOTTOM: Final = 0.96

HEADING_MAX_CHARS: Final = 90
HEADING_MAX_WORDS: Final = 8
# A bare chapter number opens the page. The same integer lower down is a list or a note.
CHAPTER_OPENER_Y: Final = 0.20
# A heading number is a small index (chapter 12, section 10.6). A page number or a
# measurement is larger. The number of a two-line heading also sits on the left.
HEADING_COMPONENT_MAX: Final = 40
HEADING_NUMBER_MAX_X: Final = 0.50
# A short token this far right is a cell or a page number, not the heading itself.
MARGIN_HEADING_X: Final = 0.70
MARGIN_HEADING_WIDTH: Final = 0.20
# The title under a bare number starts at least halfway down that number's box.
HEADING_STACK_FRACTION: Final = 0.50
# An unnumbered heading is a front-matter name near the top of the page. A line that is
# merely taller is not one: on a scan the body boxes vary by more than a heading does.
TYPOGRAPHY_MAX_Y: Final = 0.30
# A numbered heading is a line of words. A table cell is a short token in a narrow box,
# and a one-word title in a narrow box still has a word of this many letters.
HEADING_MIN_WIDTH: Final = 0.12
HEADING_LONG_WORD: Final = 4
HEADING_JOIN_GAP: Final = 0.025
# A page with this many caption lines is a list of figures or tables, not a page of them.
LIST_CAPTION_MIN: Final = 8

# Reconciliation. A number match alone is enough. A title match also needs the page.
TITLE_JACCARD: Final = 0.50
TITLE_MISMATCH: Final = 0.34
MATCH_SCORE: Final = 4
PAGE_TOLERANCE: Final = 1

DOCUMENT_STRUCTURE_FILENAME: Final = "document_structure.json"
BOOK_TITLE: Final = "Book"
