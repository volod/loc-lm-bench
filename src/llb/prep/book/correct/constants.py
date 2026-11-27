"""Frozen guard limits for evidence-constrained correction.

These values are part of the stage tool identity. A change re-runs correction instead of reusing
a page corrected under the previous guard. The adoption reading is what may change them.
"""

from typing import Final

CORRECT_VERSION: Final = "1"
PROMPT_VERSION: Final = "1"

# Admitted edits per block. Further proposals are refused and marked.
EDIT_BUDGET: Final = 3
# Character operations inside one proposal. A wider diff is a rewrite.
MAX_CHANGES_PER_EDIT: Final = 4

CLASS_DIGIT: Final = "digit"
CLASS_SIGN: Final = "sign"
CLASS_DECIMAL: Final = "decimal"
CLASS_UNIT: Final = "unit"
CLASS_INDEX: Final = "index"
CLASS_LATIN_GREEK: Final = "latin-greek"
CLASS_BRACKET: Final = "bracket"
CLASS_EQUATION: Final = "equation-number"

ADMITTED_CLASSES: Final[tuple[str, ...]] = (
    CLASS_DIGIT,
    CLASS_SIGN,
    CLASS_DECIMAL,
    CLASS_UNIT,
    CLASS_INDEX,
    CLASS_LATIN_GREEK,
    CLASS_BRACKET,
    CLASS_EQUATION,
)

# Neighbour text sent with one block. The crop is only the block's own region.
NEIGHBOUR_CHARS: Final = 240
CROP_PAD_PX: Final = 8

# Kinds that are not the book's text. A figure stays the crop the page shows.
SKIP_KINDS: Final = frozenset(("ornament", "figure", "neighbour-page-bleed"))

CACHE_DIRNAME: Final = "cache"
OLLAMA_NUM_PREDICT: Final = 512
OLLAMA_NUM_CTX: Final = 4096
OLLAMA_TIMEOUT_S: Final = 180.0


def guard_identity() -> str:
    """The tool string a page cache key digests, including the frozen limits."""
    classes = ",".join(ADMITTED_CLASSES)
    return (
        f"llb-book-correct {CORRECT_VERSION} budget={EDIT_BUDGET} "
        f"changes={MAX_CHANGES_PER_EDIT} classes={classes}"
    )
