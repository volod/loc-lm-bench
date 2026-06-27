"""Public facade for the guarded verified-category composite."""

from llb.scoring.composite_builder import (
    build_category_composite_rows,
    normalized_composite_weights,
)
from llb.scoring.composite_format import format_composite_issues, format_composite_rows
from llb.scoring.composite_types import (
    CATEGORY_COMPOSITE_RAW_WEIGHTS,
    CATEGORY_COMPOSITE_REQUIRED_TIERS,
    CompositeComponent,
    CompositeIssue,
)

__all__ = [
    "CATEGORY_COMPOSITE_RAW_WEIGHTS",
    "CATEGORY_COMPOSITE_REQUIRED_TIERS",
    "CompositeComponent",
    "CompositeIssue",
    "build_category_composite_rows",
    "format_composite_issues",
    "format_composite_rows",
    "normalized_composite_weights",
]
