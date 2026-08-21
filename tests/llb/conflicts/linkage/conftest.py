"""The planted edition corpus and the pieces every edition-linkage test reads it through."""

from pathlib import Path

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.constants import REL_DUPLICATE, REL_SUBSUMED_BY, TIER_LEXICAL
from llb.conflicts.corpus import load_corpus_docs
from llb.conflicts.tiers.hash import detect_hash_duplicates
from llb.core.paths import PROJECT_ROOT

EDITIONS_CORPUS = PROJECT_ROOT / "samples" / "corpora" / "editions_uk_v1" / "corpus"
# The seven-document conflict fixture, which is BELOW the lane's document floor -- the corpus the
# decline is asserted on.
SMALL_CORPUS = PROJECT_ROOT / "samples" / "corpora" / "conflicts_uk_v1" / "corpus"

# What the plant is (see the corpus README): six families of editions, and the two notes a longer
# document absorbed whole.
PLANTED_EDITION_FAMILIES = {
    "appeals": (
        "appeals/polozhennia-2019.md",
        "appeals/polozhennia-2022.md",
        "appeals/polozhennia-2022-copy.md",
        "appeals/polozhennia-2022-sharepoint.md",
    ),
    "archive": (
        "archive/instruktsiia-2018.md",
        "archive/instruktsiia-2021.md",
        "archive/instruktsiia-2021-copy.md",
    ),
    "edoc": (
        "edoc/rehlament-2020.md",
        "edoc/rehlament-2023.md",
        "edoc/rehlament-2023-portal.md",
    ),
    "travel": ("travel/poriadok-2021.md", "travel/poriadok-2024.md"),
    "privacy": ("privacy/polityka-2022.md", "privacy/polityka-2022-copy.md"),
    "hr": ("hr/polozhennia-2020.md", "hr/polozhennia-2023.md"),
}
PLANTED_NOTES = (
    "appeals/pamiatka-stroky-rozghliadu.md",
    "privacy/dovidka-zghoda-subiekta.md",
)


@pytest.fixture(scope="session")
def edition_docs():
    return load_corpus_docs(EDITIONS_CORPUS)


@pytest.fixture(scope="session")
def edition_audit():
    """The audit the lane reads: the two model-free tiers over the planted corpus."""
    return run_audit(EDITIONS_CORPUS, AuditParams(effort=TIER_LEXICAL))


@pytest.fixture(scope="session")
def settled_pairs(edition_docs):
    _, settled = detect_hash_duplicates(list(edition_docs))
    return settled


@pytest.fixture(scope="session")
def edition_lane(edition_docs, edition_audit, settled_pairs):
    """One real fit over the planted corpus, shared by every heavy test in this directory."""
    pytest.importorskip("splink")
    pytest.importorskip("duckdb")
    from llb.conflicts.linkage.run import run_edition_linkage

    return run_edition_linkage(
        list(edition_docs),
        edition_audit.findings,
        settled_pairs,
        jaccard_threshold=AuditParams(effort=TIER_LEXICAL).jaccard_threshold,
        containment_threshold=AuditParams(effort=TIER_LEXICAL).containment_threshold,
    )


def relations_of(findings, relation: str) -> set[tuple[str, str]]:
    return {finding.doc_pair() for finding in findings if finding.relation == relation}


def planted_relations(findings) -> tuple[set, set]:
    return relations_of(findings, REL_DUPLICATE), relations_of(findings, REL_SUBSUMED_BY)


def family_of(doc_id: str) -> str:
    for name, members in PLANTED_EDITION_FAMILIES.items():
        if doc_id in members:
            return name
    return Path(doc_id).parent.as_posix()
