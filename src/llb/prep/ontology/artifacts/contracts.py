"""The ontology and extraction records as their registered contracts.

Two producers write these files: `prepare-goldset` writes them into a draft bundle, and
`build-graph` persists the same pair beside a graph store so a later refresh can chain from them.
One place builds the record, so a graph store's inputs carry the same identity a bundle's do.
"""

from llb.core.contracts.data_prep.ontology import OntologyDocument, OntologyExtractionRow
from llb.prep.ontology.constants import (
    EXTRACTION_SCHEMA_ID,
    EXTRACTION_SCHEMA_VERSION,
    ONTOLOGY_SCHEMA_ID,
    ONTOLOGY_SCHEMA_VERSION,
)
from llb.prep.ontology.models import DocExtraction, OntologyCandidate


def ontology_record(ontology: OntologyCandidate) -> OntologyDocument:
    """The induced ontology as its registered contract."""
    return OntologyDocument.model_validate(
        {
            "schema_id": ONTOLOGY_SCHEMA_ID,
            "schema_version": ONTOLOGY_SCHEMA_VERSION,
            **ontology.model_dump(),
        }
    )


def extraction_record(extraction: DocExtraction) -> OntologyExtractionRow:
    """One extraction row as its registered contract."""
    return OntologyExtractionRow.model_validate(
        {
            "schema_id": EXTRACTION_SCHEMA_ID,
            "schema_version": EXTRACTION_SCHEMA_VERSION,
            **extraction.model_dump(),
        }
    )
