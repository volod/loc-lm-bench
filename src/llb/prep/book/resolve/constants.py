"""Names for the resolution stage, its files, and the decisions it reads."""

from typing import Final

RESOLVE_STAGE: Final = "resolve"
CORRECT_STAGE: Final = "correct"
RESOLVE_VERSION: Final = "1"
RESOLVE_TOOL: Final = "llb-book-resolve 1"

OVERLAY_FILENAME: Final = "overlay.json"
RESOLVED_FILENAME: Final = "resolved.json"
CORRECTIONS_FILENAME: Final = "corrections.json"
DECISIONS_FILENAME: Final = "decisions.json"
REVIEW_DIRNAME: Final = "review"

OVERLAY_SCHEMA_ID: Final = "llb.book-ocr-correction-overlay"
RESOLVED_SCHEMA_ID: Final = "llb.book-ocr-resolved-page"
MODEL_SCHEMA_ID: Final = "llb.book-ocr-model-corrections"
REVIEWER_SCHEMA_ID: Final = "llb.book-ocr-reviewer-decisions"
SCHEMA_VERSION: Final = "1.0.0"

OVERLAY_KIND: Final = "book-ocr-correction-overlay"
RESOLVED_KIND: Final = "book-ocr-resolved-page"
MODEL_KIND: Final = "book-ocr-model-corrections"
REVIEWER_KIND: Final = "book-ocr-reviewer-decisions"

# Cache-key input name for the run-level reviewer file. A stage that reads those decisions
# includes this digest; the digest covers that page's decisions and no other page.
REVIEWER_INPUT: Final = "reviewer"
AGREEMENT_SOURCE: Final = "agreement"
AGREEMENT_EVIDENCE: Final = "every reading agrees"
AUTHOR_RANK: Final = {"reviewer": 0, "model": 1, "agreement": 2}
