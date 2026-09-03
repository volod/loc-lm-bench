"""Load the ontology-assisted drafting extraction artifacts that feed the graph build (GraphRAG backend construction).

The graph REUSES the ontology-assisted drafting extraction; this module reads it back. The primary input is a
`prepare-goldset` draft bundle (its `extraction.jsonl` + `corpus/`), but explicit paths are also
supported, and a corpus with no prior extraction can be extracted fresh through the same ontology-assisted drafting
endpoint adapter. Kept separate from the CLI so the loading is unit-testable.
"""

import json
import logging
from pathlib import Path

from llb.artifacts.records import decode
from llb.prep.ontology.constants import (
    CORPUS_DIRNAME,
    EXTRACTION_FILENAME,
    EXTRACTION_SCHEMA_ID,
    ONTOLOGY_FILENAME,
    ONTOLOGY_SCHEMA_ID,
)
from llb.prep.ontology.coverage.inventory import inventory_corpus
from llb.prep.ontology.models import DocExtraction, DocRecord, OntologyCandidate

_LOG = logging.getLogger(__name__)


def load_extractions(path: Path | str) -> list[DocExtraction]:
    """Read an `extraction.jsonl` back into typed records, through its registered contract.

    A bundle drafted by a newer build is refused here rather than silently dropping the fields
    this build's `DocExtraction` does not name -- which is exactly what ignoring extras would do.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"extraction file not found: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    extractions = [
        DocExtraction.model_validate(
            decode(EXTRACTION_SCHEMA_ID, json.loads(line), source=f"{path}#record-{index}")
        )
        for index, line in enumerate(lines, start=1)
    ]
    if not extractions:
        raise SystemExit(f"no extractions in {path}")
    return extractions


def load_ontology(path: Path | str) -> OntologyCandidate | None:
    """Read an induced `ontology.json` if present (carries the type confidences onto nodes)."""
    path = Path(path)
    if not path.exists():
        return None
    record = decode(
        ONTOLOGY_SCHEMA_ID, json.loads(path.read_text(encoding="utf-8")), source=str(path)
    )
    return OntologyCandidate.model_validate(record)


def load_bundle(
    bundle_dir: Path | str,
) -> tuple[list[DocExtraction], list[DocRecord], OntologyCandidate | None]:
    """Load (extractions, docs, ontology) from a `prepare-goldset` draft bundle directory."""
    bundle_dir = Path(bundle_dir)
    extractions = load_extractions(bundle_dir / EXTRACTION_FILENAME)
    docs = inventory_corpus(bundle_dir / CORPUS_DIRNAME)
    ontology = load_ontology(bundle_dir / ONTOLOGY_FILENAME)
    _LOG.info(
        "[graph] loaded bundle %s: %d extractions, %d docs", bundle_dir, len(extractions), len(docs)
    )
    return extractions, docs, ontology
