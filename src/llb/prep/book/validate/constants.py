"""Check names and the confidence buckets the vault report prints.

A confidence bucket is a routing diagnostic. It is not a probability that the
characters are correct.
"""

from typing import Final

UNACCOUNTED_PAGE: Final = "unaccounted-page"
UNREACHABLE_CONTENT: Final = "unreachable-content"
PREV_NEXT_RECIPROCAL: Final = "prev-next-not-reciprocal"
PREV_NEXT_CHAIN: Final = "prev-next-chain"
BROKEN_RELATIVE: Final = "broken-relative-link"
BROKEN_WIKILINK: Final = "broken-wikilink"
BROKEN_ANCHOR: Final = "broken-block-anchor"
BROKEN_EMBED: Final = "broken-embed"
PDF_FRAGMENT: Final = "pdf-page-fragment"
MISSING_ASSET: Final = "missing-asset"
UNREFERENCED_ASSET: Final = "unreferenced-asset"
BLOCK_WITHOUT_PAGE: Final = "block-without-page"
MARKER_MISMATCH: Final = "marker-queue-mismatch"

VALIDATION_FILENAME: Final = "validation.json"
IMAGE_SUFFIXES: Final = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"})
ASSET_PREFIXES: Final = ("assets/", "review/", "source/pages/")

# Lower bound inclusive, upper bound exclusive, except the last bucket which
# also takes a score of 1 and any score above it.
CONFIDENCE_BUCKETS: Final = (
    ("below-0.50", 0.0, 0.5),
    ("0.50-0.80", 0.5, 0.8),
    ("0.80-0.90", 0.8, 0.9),
    ("0.90-0.95", 0.9, 0.95),
    ("0.95-1.00", 0.95, 1.01),
)
CONFIDENCE_NOTE: Final = (
    "Confidence is diagnostic routing evidence, not a probability of correctness."
)
QUEUE_MISSING: Final = "missing-review-queue"
QUEUE_STALE: Final = "stale-review-queue"
QUEUE_WARNING: Final = "review queue was not written; open markers were checked against the ledger"
QUEUE_DRIFT_WARNING: Final = (
    "the review queue does not match the ledger; markers were checked against the queue"
)
