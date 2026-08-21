"""One real fit over the planted edition corpus: what it recovers, groups, and refuses to run on.

Heavy only in the sense of the `linkage` extra -- no GPU, no network, one in-memory DuckDB. The
plant's answer is known (six edition families and two absorbed notes, see the corpus README), so
each assertion below is against a corpus fact rather than against a number this fit happened to
produce.
"""

import json

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.constants import REL_DUPLICATE, REL_SUBSUMED_BY, TIER_LEXICAL
from llb.conflicts.linkage.artifacts import write_edition_linkage
from llb.conflicts.linkage.constants import EDITIONS_FILE, MIN_LINKAGE_DOCUMENTS, RECORDS_FILE
from llb.conflicts.linkage.constants import SUMMARY_FILE as EDITION_SUMMARY_FILE
from llb.conflicts.linkage.report import console_lines, report_section
from llb.conflicts.linkage.run import VERDICT_RANKED
from llb.conflicts.report.render import write_audit
from llb.conflicts.tiers.lexical import candidate_pairs, shingles
from llb.linkage.artifacts import linkage_dir
from llb.linkage.constants import CLUSTERS_FILE, MODEL_FILE, PAIRS_FILE, SETTINGS_FILE

from tests.llb.conflicts.linkage.conftest import (
    EDITIONS_CORPUS,
    PLANTED_EDITION_FAMILIES,
    PLANTED_NOTES,
    SMALL_CORPUS,
    family_of,
)

pytestmark = pytest.mark.heavy_env


def test_the_fit_scores_every_candidate_pair_the_lexical_tier_generates(edition_lane, edition_docs):
    """The exploding blocking rule and `candidate_pairs` must agree, in the engine and not only
    in Python: a fit that scored a different candidate list would be ranking a different corpus."""
    expected = candidate_pairs([shingles(doc.body) for doc in edition_docs])
    assert edition_lane.summary["linkage"]["n_scored_pairs"] == len(expected)


def test_every_relation_the_thresholds_recover_is_scored_and_ranked_above_the_rest(edition_lane):
    recovered = edition_lane.summary["recovery"]
    assert recovered["relations"] == recovered["scored"] > 0
    assert edition_lane.summary["verdict"] == VERDICT_RANKED


def test_every_duplicate_the_thresholds_report_is_merged_into_one_edition(edition_lane):
    decided = edition_lane.summary["decisions"]
    assert decided["thresholds_only"] == []
    assert decided["agreed"] == decided["threshold_duplicates"] > 0


def test_the_edition_groups_are_the_planted_families(edition_lane):
    """Six families, each whole, and no member of one family merged into another."""
    grouped = {frozenset(group.doc_ids) for group in edition_lane.groups}
    planted = {frozenset(members) for members in PLANTED_EDITION_FAMILIES.values()}
    assert grouped == planted
    for members in grouped:
        assert len({family_of(doc_id) for doc_id in members}) == 1


def test_an_absorbed_note_is_ranked_but_never_merged_into_the_document_that_absorbed_it(
    edition_lane,
):
    """Subsumption is not identity. The note must be scored and ranked, and stay its own document."""
    rows = edition_lane.summary["recovery"]["rows"]
    subsumptions = [row for row in rows if row["relation"] == REL_SUBSUMED_BY]
    assert subsumptions, "the plant carries absorbed notes"
    assert all(row["scored"] and not row["co_clustered"] for row in subsumptions)
    grouped = {doc for group in edition_lane.groups for doc in group.doc_ids}
    assert all(note not in grouped for note in PLANTED_NOTES)


def test_the_duplicates_rank_above_every_subsumption(edition_lane):
    """The order the tier's two incomparable measures cannot express, and the fit can."""
    rows = edition_lane.summary["recovery"]["rows"]
    duplicates = [row["rank"] for row in rows if row["relation"] == REL_DUPLICATE]
    subsumptions = [row["rank"] for row in rows if row["relation"] == REL_SUBSUMED_BY]
    assert max(duplicates) < min(subsumptions)


def test_the_prior_is_measured_from_the_hash_tier_not_defaulted(edition_lane):
    prior = edition_lane.summary["prior"]
    assert prior["settled_pairs"] > 0
    assert prior["random_match_probability"] > 0.01


def test_two_fits_of_the_same_corpus_publish_identical_pairs(
    edition_lane, edition_docs, edition_audit, settled_pairs
):
    from llb.conflicts.linkage.run import run_edition_linkage

    again = run_edition_linkage(
        list(edition_docs),
        edition_audit.findings,
        settled_pairs,
        jaccard_threshold=AuditParams(effort=TIER_LEXICAL).jaccard_threshold,
        containment_threshold=AuditParams(effort=TIER_LEXICAL).containment_threshold,
    )
    assert [pair.payload() for pair in again.result.pairs] == [
        pair.payload() for pair in edition_lane.result.pairs
    ]
    assert again.summary["cut"] == edition_lane.summary["cut"]


def test_a_corpus_below_the_document_floor_declines_with_its_reason(tmp_path):
    result = run_audit(SMALL_CORPUS, AuditParams(effort=TIER_LEXICAL, linkage=True))
    summary = result.edition_linkage
    assert summary["declined"] is True
    assert str(MIN_LINKAGE_DOCUMENTS) in summary["reason"]
    paths = write_edition_linkage(tmp_path, result.edition_linkage_run)
    assert paths["edition_summary"].exists()
    assert not (linkage_dir(tmp_path) / MODEL_FILE).exists()
    assert console_lines(summary) == [f"[conflicts] edition linkage not run: {summary['reason']}"]


def test_the_bundle_holds_every_documented_artifact(tmp_path):
    result = run_audit(EDITIONS_CORPUS, AuditParams(effort=TIER_LEXICAL, linkage=True))
    write_audit(tmp_path, result)
    bundle = linkage_dir(tmp_path)
    for name in (
        SETTINGS_FILE,
        MODEL_FILE,
        PAIRS_FILE,
        CLUSTERS_FILE,
        RECORDS_FILE,
        EDITIONS_FILE,
        EDITION_SUMMARY_FILE,
    ):
        assert (bundle / name).is_file(), name
    settings = json.loads((bundle / SETTINGS_FILE).read_text(encoding="utf-8"))
    assert settings["metadata"]["mode"] == "corpus-edition-linkage"


def test_the_record_table_records_shingle_counts_not_the_shingles(tmp_path):
    result = run_audit(EDITIONS_CORPUS, AuditParams(effort=TIER_LEXICAL, linkage=True))
    write_audit(tmp_path, result)
    rows = [
        json.loads(line)
        for line in (linkage_dir(tmp_path) / RECORDS_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert rows and all(isinstance(row["shingles"], int) for row in rows)


def test_the_audit_findings_are_the_same_with_the_lane_on_and_off():
    """The lane is additive: it may not add, drop, or re-score a single finding."""
    without = run_audit(EDITIONS_CORPUS, AuditParams(effort=TIER_LEXICAL))
    with_lane = run_audit(EDITIONS_CORPUS, AuditParams(effort=TIER_LEXICAL, linkage=True))
    assert with_lane.rows() == without.rows()
    assert without.edition_linkage == {}


def test_the_decision_groups_sidecar_names_the_edition_groups_only_when_the_lane_ran(tmp_path):
    with_lane = run_audit(EDITIONS_CORPUS, AuditParams(effort=TIER_LEXICAL, linkage=True))
    write_audit(tmp_path / "on", with_lane)
    named = json.loads((tmp_path / "on" / "groups.json").read_text(encoding="utf-8"))
    assert any(group.get("edition_groups") for group in named["groups"])

    without = run_audit(EDITIONS_CORPUS, AuditParams(effort=TIER_LEXICAL))
    write_audit(tmp_path / "off", without)
    plain = json.loads((tmp_path / "off" / "groups.json").read_text(encoding="utf-8"))
    assert all("edition_groups" not in group for group in plain["groups"])


def test_the_report_section_never_calls_a_match_probability_a_conflict(edition_lane):
    text = "\n".join(report_section(edition_lane.summary))
    assert "same document" in text and "contradict" in text
    assert "Edition linkage" in text
