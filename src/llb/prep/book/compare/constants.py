"""Ids and the decision constants for the pilot pipeline comparison.

The margin that selects an engine is character error. Dangerous substitutions stay
beside that margin and are never added into it. Pilot numbers are selection-set
numbers.
"""

from typing import Final

KIND: Final = "book-ocr-pilot-comparison"
SCHEMA_VERSION: Final = "1.0.0"
BASELINE_ENGINE: Final = "docling"
ROLE_PAGE: Final = "page-reader"
ROLE_VARIANT: Final = "variant-policy"
STATUS_MEASURED: Final = "measured"
STATUS_OVER_VRAM: Final = "over-vram"
STATUS_UNAVAILABLE: Final = "unavailable"
STATUS_GAIN: Final = "gain"
STATUS_LOSS: Final = "loss"
STATUS_INCONCLUSIVE: Final = "inconclusive"
# Survey peaks below, converted from the published gigabytes at 1024 MiB each.
# The dry run compares these with the device total, which is also in MiB.
BASELINE_VRAM_MIB: Final = round(2.9 * 1024)
REOPEN_WHEN: Final = (
    "Reopen when a measured candidate's character-error margin on a pilot class "
    "has a page-bootstrap interval entirely above zero and its dangerous-substitution "
    "rate is not higher than the existing lane."
)
