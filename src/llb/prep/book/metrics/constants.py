"""Confusion-class ids and the thresholds the quality harness shares.

The ids are the report keys. A count stays under its id and is never added into character error.
"""

from typing import Final

CLASS_ZERO_O: Final = "0/O"
CLASS_ONE_LI: Final = "1/l/I"
CLASS_FIVE_S: Final = "5/S"
CLASS_MINUS: Final = "minus/hyphen/dash"
CLASS_DECIMAL: Final = "decimal/comma"
CLASS_LATIN_GREEK: Final = "latin/greek"
CLASS_SCRIPT: Final = "script"
CLASS_UNIT: Final = "unit"
CLASS_BRACKET: Final = "bracket"
CLASS_EQUATION: Final = "equation-number"

CONFUSION_CLASSES: tuple[str, ...] = (
    CLASS_ZERO_O,
    CLASS_ONE_LI,
    CLASS_FIVE_S,
    CLASS_MINUS,
    CLASS_DECIMAL,
    CLASS_LATIN_GREEK,
    CLASS_SCRIPT,
    CLASS_UNIT,
    CLASS_BRACKET,
    CLASS_EQUATION,
)

# A block match below this intersection-over-union is a different region.
MATCH_IOU: Final = 0.5

STATUS_SCORED: Final = "scored"
STATUS_ABSENT: Final = "absent"
STATUS_NOT_APPLICABLE: Final = "not_applicable"

LANE_BOOK: Final = "book-ocr"
LANE_EXISTING: Final = "existing-pdf"
ORIGINAL_VARIANT: Final = "original"
REFERENCE_PAGES_DIRNAME: Final = "pages"
