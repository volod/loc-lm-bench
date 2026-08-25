"""Shared paths and loaders for the ontology axiom-layer tests."""

import json
from pathlib import Path

import pytest

from llb.core.paths import PROJECT_ROOT
from llb.prep.ontology.axioms.ledger import Ledger
from llb.prep.ontology.axioms.loader import load_axioms
from llb.prep.ontology.axioms.models import AxiomSet
from llb.prep.ontology.models import DocExtraction

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "ontology"
FIXTURE_LEDGER = FIXTURE_DIR / "axiom_fixture_extraction.jsonl"
FIXTURE_DOC = FIXTURE_DIR / "axiom_fixture_uk.md"
CANDIDATE_TURTLE = PROJECT_ROOT / "samples" / "ontology" / "axioms_uk_v1.ttl"
CANDIDATE_JSON = PROJECT_ROOT / "samples" / "ontology" / "axioms_uk_v1.json"


@pytest.fixture(scope="session")
def axiom_set() -> AxiomSet:
    """The committed candidate constraint set, read from its Turtle source."""
    return load_axioms(CANDIDATE_TURTLE)


@pytest.fixture(scope="session")
def fixture_extractions() -> list[DocExtraction]:
    """The planted-violation ledger: one violation per axiom class, all spans exact."""
    return [
        DocExtraction.model_validate(json.loads(line))
        for line in FIXTURE_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="session")
def fixture_ledger(fixture_extractions: list[DocExtraction]) -> Ledger:
    return Ledger(fixture_extractions)


@pytest.fixture(scope="session")
def fixture_doc_text() -> str:
    return FIXTURE_DOC.read_text(encoding="utf-8")


def path_of(name: str) -> Path:
    return FIXTURE_DIR / name
