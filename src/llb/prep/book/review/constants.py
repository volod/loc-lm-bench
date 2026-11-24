"""Names, marker kinds, and the routing cut for the book review queue."""

from typing import Final

from llb.core.contracts.data_prep.book_review import MARKER_KINDS, MarkerKind

QUEUE_FILENAME: Final = "queue.json"
QUALITY_MANIFEST_FILENAME: Final = "quality_manifest.json"
CROP_PADDING_PX: Final = 16
# A reported score below this is routed to review. On this scan RapidOCR's ordinary
# lines sit at or above 0.98, and the lines worth a look sit below 0.9. A missing
# score does not route. The number is a cut, not a probability of correctness.
LOW_CONFIDENCE_BELOW: Final = 0.9

FORMULA_TABLE_KINDS: Final = frozenset(("formula", "table"))
# Furniture and figures stay in the ledger. An empty one is not an unread passage.
REVIEW_TEXT_KINDS: Final = frozenset(
    ("body", "heading", "caption", "footnote", "unknown", "formula", "table")
)
MARKER_KIND_ORDER: Final[tuple[MarkerKind, ...]] = MARKER_KINDS

ACCEPT_RAW: Final = "accept-raw"
ACCEPT_CANDIDATE: Final = "accept-candidate"
ENTER_TEXT: Final = "enter-text"
MARK_ILLEGIBLE: Final = "mark-illegible"
CLEAR_DECISION: Final = "clear"
DECISION_TOKENS: Final = frozenset((ACCEPT_RAW, ACCEPT_CANDIDATE, ENTER_TEXT, MARK_ILLEGIBLE))
