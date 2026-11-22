"""The registered scanned-book recovery run families.

Each has one version and no legacy read form: nothing was written before the registry knew it.
"""

from llb.artifacts.definitions import ContractDefinition
from llb.core.contracts.artifact_catalog import FormatBinding
from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookPageManifest, BookStageRecord
from llb.core.contracts.data_prep.book_blocks import PageBlockLedger
from llb.core.contracts.data_prep.book_classes import PageClassRecord, PilotSelection
from llb.core.contracts.data_prep.book_preprocess import PageSignals, VariantManifest
from llb.core.contracts.data_prep.book_resolve import (
    ModelCorrections,
    PageCorrectionOverlay,
    PageResolvedText,
    ReviewerDecisions,
)

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
            schema_id="llb.book-ocr-correction-overlay",
            description="Edits applied to one scanned page, and edits a rebuilt ledger orphaned.",
            current_version="1.0.0",
            models={"1.0.0": PageCorrectionOverlay},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
        ContractDefinition(
            schema_id="llb.book-ocr-model-corrections",
            description="Model edits for one scanned page, read by resolution and not produced by it.",
            current_version="1.0.0",
            models={"1.0.0": ModelCorrections},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
        ContractDefinition(
            schema_id="llb.book-ocr-page-classes",
            description="Every class one scanned page matches, and the class an engine stage routes on.",
            current_version="1.0.0",
            models={"1.0.0": PageClassRecord},
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
            schema_id="llb.book-ocr-page-signals",
            description="One scanned page's measured signal vector.",
            current_version="1.0.0",
            models={"1.0.0": PageSignals},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
        ContractDefinition(
            schema_id="llb.book-ocr-pilot-selection",
            description="The seeded pilot pages, the class each one represents, and the signals that placed it.",
            current_version="1.0.0",
            models={"1.0.0": PilotSelection},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
        ContractDefinition(
            schema_id="llb.book-ocr-resolved-page",
            description="One scanned page's resolved text, with the provenance of every span.",
            current_version="1.0.0",
            models={"1.0.0": PageResolvedText},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
        ContractDefinition(
            schema_id="llb.book-ocr-reviewer-decisions",
            description="Reviewer edits for a scanned-book run, kept outside the stage cache.",
            current_version="1.0.0",
            models={"1.0.0": ReviewerDecisions},
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
        ContractDefinition(
            schema_id="llb.book-ocr-variant-manifest",
            description="The image variants generated for one scanned page, and the maps back.",
            current_version="1.0.0",
            models={"1.0.0": VariantManifest},
            bindings=_JSON_DOCUMENT,
            deprecation_policy=_ONLY_FORM,
        ),
    )
