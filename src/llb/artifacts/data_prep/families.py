"""The registered data-preparation contract families.

Every family declares its current version, the older versions this build still reads, and how an
old record reaches the current one. `legacy_version` is what lets the migration cover the files
already on disk: this project wrote them before the registry existed, so they carry no identity at
all and a reader that knows which family it opened supplies it.
"""

from collections.abc import Mapping

from pydantic import BaseModel

from llb.artifacts.data_prep.migrations import (
    gold_item_v1_to_v2,
    linkage_settings_v1_to_v2,
    ontology_provenance_v1_to_v2,
    stage_inputs_to_current,
)
from llb.artifacts.definitions import ContractDefinition, MigrationStep
from llb.core.contracts.artifact_catalog import FormatBinding
from llb.core.contracts.data_prep.conflicts import (
    ConflictOverlay,
    ConflictStageInputs,
    ConflictStageInputsV1,
    ConflictStageInputsV2,
    ConflictStageInputsV3,
    ConflictStageInputsV4,
    ConflictStageInputsV5,
    ConflictStageInputsV6,
    STAGE_INPUTS_SCHEMA_ID,
)
from llb.core.contracts.data_prep.corpus import CorpusManifest, PdfCitations, PdfCorpusManifest
from llb.core.contracts.data_prep.external_draft import (
    ExternalDraftItemRow,
    ExternalDraftProvenance,
)
from llb.core.contracts.data_prep.goldset import (
    GoldChainRecord,
    GoldItemRecord,
    GoldItemRecordV1,
    NeedleItemRecord,
)
from llb.core.contracts.data_prep.linkage import LinkageSettings, LinkageSettingsV1
from llb.core.contracts.data_prep.ontology import (
    OntologyDocument,
    OntologyExtractionRow,
    OntologyProvenance,
    OntologyProvenanceV1,
)
from llb.core.contracts.data_prep.review import VerificationWorksheetRow

JSON_DOCUMENT = (
    FormatBinding(format="json", media_type="application/json", granularity="document"),
)
JSONL_ROWS = (FormatBinding(format="jsonl", media_type="application/x-ndjson", granularity="row"),)
CSV_ROWS = (FormatBinding(format="csv", media_type="text/csv", granularity="row"),)

# The stage-inputs record keeps the compact integer version it has always written inside the
# bundle; the registry names the same form semantically. The two are one encoding of one version,
# which is why the mapping lives here rather than being inferred at either end.
_STAGE_INPUTS_MODELS: Mapping[str, type[BaseModel]] = {
    "1.0.0": ConflictStageInputsV1,
    "2.0.0": ConflictStageInputsV2,
    "3.0.0": ConflictStageInputsV3,
    "4.0.0": ConflictStageInputsV4,
    "5.0.0": ConflictStageInputsV5,
    "6.0.0": ConflictStageInputsV6,
    "7.0.0": ConflictStageInputs,
}
STAGE_INPUTS_CURRENT_VERSION = "7.0.0"


def contract_version(local_version: int) -> str:
    """The registry version naming the bundle's own integer stage-inputs schema."""
    version = f"{int(local_version)}.0.0"
    if version not in _STAGE_INPUTS_MODELS:
        raise ValueError(
            f"conflict stage-inputs schema {local_version} is not a registered contract version; "
            f"registered: {sorted(_STAGE_INPUTS_MODELS)}"
        )
    return version


def local_stage_inputs_version(version: str) -> int:
    """The bundle's integer schema for a registry version, the inverse of `contract_version`."""
    return int(version.split(".", 1)[0])


def _stage_inputs_migrations() -> tuple[MigrationStep, ...]:
    return tuple(
        MigrationStep(
            from_version=version,
            to_version=STAGE_INPUTS_CURRENT_VERSION,
            description=(
                f"Re-encode a schema-{local_stage_inputs_version(version)} record at the current "
                "form through the bundle reader that already understands it."
            ),
            transform=stage_inputs_to_current,
        )
        for version in sorted(_STAGE_INPUTS_MODELS)
        if version != STAGE_INPUTS_CURRENT_VERSION
    )


def data_prep_definitions() -> tuple[ContractDefinition, ...]:
    """Every data-preparation family, in schema-id order."""
    return (
        ContractDefinition(
            schema_id="llb.conflict-overlay",
            description="Applied conflict overlay read by corpus chunking and fingerprints.",
            current_version="1.0.0",
            models={"1.0.0": ConflictOverlay},
            bindings=JSON_DOCUMENT,
            deprecation_policy="Overlay version 1 is the only form; a newer one refuses a build.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id=STAGE_INPUTS_SCHEMA_ID,
            description="Per-document record a finished conflict audit answers its readings from.",
            current_version=STAGE_INPUTS_CURRENT_VERSION,
            models=_STAGE_INPUTS_MODELS,
            bindings=JSON_DOCUMENT,
            deprecation_policy=(
                "Every earlier form stays readable: a finished run's readings outlive its store."
            ),
            migrations=_stage_inputs_migrations(),
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.corpus-manifest",
            description="Mixed-corpus ingestion manifest: every source, its status, governance.",
            current_version="1.0.0",
            models={"1.0.0": CorpusManifest},
            bindings=JSON_DOCUMENT,
            deprecation_policy="Manifest version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.external-draft-item",
            description="Per-item labels an external-draft import records beside the bundle.",
            current_version="1.0.0",
            models={"1.0.0": ExternalDraftItemRow},
            bindings=JSONL_ROWS,
            deprecation_policy="Row version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.external-draft-provenance",
            description="What an external-draft import loaded, under which classification.",
            current_version="1.0.0",
            models={"1.0.0": ExternalDraftProvenance},
            bindings=JSON_DOCUMENT,
            deprecation_policy="Sidecar version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.gold-chain",
            description="One ordered chain-of-questions gold item.",
            current_version="1.0.0",
            models={"1.0.0": GoldChainRecord},
            bindings=JSONL_ROWS,
            deprecation_policy="Chain version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.gold-item",
            description="One gold item: a question, its reference answer, and its source spans.",
            current_version="2.0.0",
            models={"1.0.0": GoldItemRecordV1, "2.0.0": GoldItemRecord},
            bindings=JSONL_ROWS,
            deprecation_policy=(
                "Version 1 is read-and-migrate: it leaves lang and verified to a reader's default."
            ),
            migrations=(
                MigrationStep(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    description="State lang and verified instead of leaving them to a default.",
                    transform=gold_item_v1_to_v2,
                ),
            ),
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.linkage-settings",
            description="The whole linkage decision a replay re-scores from.",
            current_version="2.0.0",
            models={"1.0.0": LinkageSettingsV1, "2.0.0": LinkageSettings},
            bindings=JSON_DOCUMENT,
            deprecation_policy=(
                "Version 1 is read-and-migrate: its tuning knobs came from the reader's build."
            ),
            migrations=(
                MigrationStep(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    description="State every tuning knob the specification left to a default.",
                    transform=linkage_settings_v1_to_v2,
                ),
            ),
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.needle-item",
            description="A citation-valid gold row with its drafting labels and retrieval rank.",
            current_version="1.0.0",
            models={"1.0.0": NeedleItemRecord},
            bindings=JSONL_ROWS,
            deprecation_policy="Row version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.ontology",
            description="The constrained candidate ontology induced over a corpus.",
            current_version="1.0.0",
            models={"1.0.0": OntologyDocument},
            bindings=JSON_DOCUMENT,
            deprecation_policy="Ontology version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.ontology-extraction",
            description="Everything extracted from one document, every span exact-grounded.",
            current_version="1.0.0",
            models={"1.0.0": OntologyExtractionRow},
            bindings=JSONL_ROWS,
            deprecation_policy="Row version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.ontology-provenance",
            description="What produced one draft bundle, from which corpus version.",
            current_version="2.0.0",
            models={"1.0.0": OntologyProvenanceV1, "2.0.0": OntologyProvenance},
            bindings=JSON_DOCUMENT,
            deprecation_policy=(
                "Version 1 is read-and-migrate: it predates the corpus version binding."
            ),
            migrations=(
                MigrationStep(
                    from_version="1.0.0",
                    to_version="2.0.0",
                    description="State the corpus binding and per-document acquisition as absent.",
                    transform=ontology_provenance_v1_to_v2,
                ),
            ),
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.pdf-citations",
            description="Which PDF page a staged character offset came from.",
            current_version="1.0.0",
            models={"1.0.0": PdfCitations},
            bindings=JSON_DOCUMENT,
            deprecation_policy="Sidecar version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.pdf-corpus-manifest",
            description="PDF conversion manifest and its quality report.",
            current_version="1.0.0",
            models={"1.0.0": PdfCorpusManifest},
            bindings=JSON_DOCUMENT,
            deprecation_policy="Manifest version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
        ContractDefinition(
            schema_id="llb.verification-worksheet-row",
            description="One sampled item as the verification worksheet carries it.",
            current_version="1.0.0",
            models={"1.0.0": VerificationWorksheetRow},
            bindings=CSV_ROWS,
            deprecation_policy="Worksheet version 1 remains readable for this release line.",
            legacy_version="1.0.0",
        ),
    )
