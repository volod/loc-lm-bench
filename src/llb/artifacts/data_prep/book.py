"""The registered scanned-book recovery run families.

All four are new with this capability, so each has one version and no legacy read form: nothing
was written before the registry knew them.
"""

from llb.artifacts.definitions import ContractDefinition
from llb.core.contracts.artifact_catalog import FormatBinding
from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookPageManifest, BookStageRecord
from llb.core.contracts.data_prep.book_blocks import PageBlockLedger

_JSON_DOCUMENT = (
    FormatBinding(format="json", media_type="application/json", granularity="document"),
)
_ONLY_FORM = "Version 1 is the only form; a resume refuses any other."


def book_definitions() -> tuple[ContractDefinition, ...]:
    """Every `book-ocr` run family, in schema-id order."""
    return (
        ContractDefinition(
            schema_id="llb.book-ocr-block-ledger",
            description="One scanned page's blocks, raw engine readings, and uncertainty spans.",
            current_version="1.0.0",
            models={"1.0.0": PageBlockLedger},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
        ContractDefinition(
            schema_id="llb.book-ocr-page-manifest",
            description="Per-page, per-stage ledger of one scanned-book recovery run.",
            current_version="1.0.0",
            models={"1.0.0": BookPageManifest},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
        ContractDefinition(
            schema_id="llb.book-ocr-run-config",
            description="What one scanned-book recovery run is: source, pages, stages, models.",
            current_version="1.0.0",
            models={"1.0.0": BookOcrRunConfig},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
        ContractDefinition(
            schema_id="llb.book-ocr-stage-record",
            description="One stage's outcome on one page and the cache key it was produced under.",
            current_version="1.0.0",
            models={"1.0.0": BookStageRecord},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
    )
