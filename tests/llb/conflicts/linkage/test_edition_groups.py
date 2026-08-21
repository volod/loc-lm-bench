"""Edition groups, the prior, the provisional cut, and the two-ranking comparison.

All of it is arithmetic over data a fit produced, so all of it runs in the base install against
hand-built clusters and pairs -- the fit itself is exercised in `test_edition_lane.py`.
"""

from llb.conflicts.constants import REL_DUPLICATE, REL_SUBSUMED_BY, TIER_HASH, TIER_LEXICAL
from llb.conflicts.linkage.constants import DUPLICATE_RELATIONS
from llb.conflicts.linkage.editions import edition_groups, editions_by_document
from llb.conflicts.linkage.rankings import (
    ReportedRelation,
    decisions,
    duplicate_probabilities,
    ordering,
    recovery,
    reported_relations,
)
from llb.conflicts.linkage.run import (
    PRIOR_FROM_DEFAULT,
    PRIOR_FROM_HASH_TIER,
    hash_prior,
    provisional_cut,
)
from llb.conflicts.models import ClaimRef, Finding
from llb.linkage.constants import DEFAULT_MATCH_THRESHOLD, DEFAULT_RANDOM_MATCH_PROBABILITY
from llb.linkage.model import LinkageCluster, LinkagePair


def _cluster(*doc_ids: str) -> LinkageCluster:
    return LinkageCluster(cluster_id=min(doc_ids), record_ids=tuple(sorted(doc_ids)))


def _pair(left: str, right: str, probability: float, weight: float = 0.0) -> LinkagePair:
    return LinkagePair(
        left_id=left, right_id=right, match_probability=probability, match_weight=weight
    )


def _finding(relation: str, tier: str, left: str, right: str, score: float) -> Finding:
    return Finding(
        relation=relation,
        tier=tier,
        a=ClaimRef(doc_id=left, char_start=0, char_end=1, text=""),
        b=ClaimRef(doc_id=right, char_start=0, char_end=1, text=""),
        score=score,
        evidence="test",
    )


DATED = {
    "old.md": {"effective_date": "2019-01-01"},
    "new.md": {"effective_date": "2023-01-01"},
    "newest.md": {"effective_date": "2024-01-01"},
    "twin-a.md": {"effective_date": "2024-01-01"},
    "undated-a.md": {},
    "undated-b.md": {},
    "versioned-a.md": {"version": "1.0"},
    "versioned-b.md": {"version": "2.0"},
}


def test_a_group_lists_its_members_oldest_first_and_names_the_current_edition():
    groups = edition_groups([_cluster("newest.md", "old.md", "new.md")], DATED)
    assert len(groups) == 1
    assert groups[0].doc_ids == ("old.md", "new.md", "newest.md")
    assert groups[0].current == ("newest.md",)
    assert groups[0].basis == "effective_date"


def test_two_copies_carrying_one_date_are_both_current():
    """Naming either one would invent a precedence the governance fields do not record."""
    groups = edition_groups([_cluster("old.md", "newest.md", "twin-a.md")], DATED)
    assert groups[0].current == ("newest.md", "twin-a.md")
    assert groups[0].basis == "effective_date"


def test_a_group_nothing_can_order_says_so_rather_than_picking_a_member():
    groups = edition_groups([_cluster("undated-a.md", "undated-b.md")], DATED)
    assert groups[0].basis is None
    assert set(groups[0].current) == {"undated-a.md", "undated-b.md"}


def test_version_orders_a_group_with_no_dates():
    groups = edition_groups([_cluster("versioned-a.md", "versioned-b.md")], DATED)
    assert groups[0].current == ("versioned-b.md",)
    assert groups[0].basis == "version"


def test_singleton_clusters_are_not_edition_groups():
    groups = edition_groups([_cluster("old.md"), _cluster("new.md", "newest.md")], DATED)
    assert [group.size for group in groups] == [2]
    assert editions_by_document(groups) == {"new.md": "E1", "newest.md": "E1"}


def test_groups_are_numbered_largest_first():
    clusters = [_cluster("old.md", "new.md", "newest.md"), _cluster("undated-a.md", "undated-b.md")]
    groups = edition_groups(clusters, DATED)
    assert [(group.edition_id, group.size) for group in groups] == [("E1", 3), ("E2", 2)]


def test_the_prior_is_measured_from_the_hash_tiers_settled_pairs():
    prior = hash_prior(settled_pairs=6, n_docs=26)
    assert prior["total_document_pairs"] == 325
    assert prior["source"] == PRIOR_FROM_HASH_TIER
    assert prior["random_match_probability"] == 6 / (325 * prior["assumed_hash_recall"])


def test_a_corpus_the_hash_tier_settled_nothing_in_falls_back_to_the_seam_default():
    prior = hash_prior(settled_pairs=0, n_docs=26)
    assert prior["source"] == PRIOR_FROM_DEFAULT
    assert prior["random_match_probability"] == DEFAULT_RANDOM_MATCH_PROBABILITY


def test_the_cut_is_the_seam_default_when_it_already_keeps_every_duplicate():
    assert provisional_cut([0.99, 0.95])["cut"] == DEFAULT_MATCH_THRESHOLD


def test_the_cut_drops_to_the_lowest_duplicate_the_thresholds_report():
    decided = provisional_cut([0.99, 0.8054, 0.95])
    assert decided["cut"] == 0.8054


def test_reported_relations_include_the_pairs_a_chained_hash_group_implies():
    """The hash tier reports a chain and settles a closure; all of the closure is recovered."""
    findings = [_finding(REL_DUPLICATE, TIER_HASH, "a.md", "b.md", 1.0)]
    settled = {("a.md", "b.md"), ("a.md", "c.md"), ("b.md", "c.md")}
    relations = reported_relations(findings, settled)
    assert {relation.doc_pair for relation in relations} == settled
    assert all(relation.relation == REL_DUPLICATE for relation in relations)


def test_only_the_model_free_tiers_are_compared_against():
    """A semantic or claim finding is not a duplicate decision this lane can price."""
    findings = [
        _finding(REL_DUPLICATE, TIER_LEXICAL, "a.md", "b.md", 0.9),
        _finding("contradicts", "claim", "a.md", "c.md", 0.8),
    ]
    relations = reported_relations(findings, set())
    assert {relation.doc_pair for relation in relations} == {("a.md", "b.md")}


def _reported():
    return (
        ReportedRelation(("a.md", "b.md"), REL_DUPLICATE, TIER_HASH, 1.0),
        ReportedRelation(("c.md", "d.md"), REL_SUBSUMED_BY, TIER_LEXICAL, 1.0),
        ReportedRelation(("e.md", "f.md"), REL_DUPLICATE, TIER_LEXICAL, 0.85),
    )


def _pairs():
    return (
        _pair("a.md", "b.md", 0.99, 10.0),
        _pair("e.md", "f.md", 0.91, 5.0),
        _pair("x.md", "y.md", 0.4, -2.0),
        _pair("c.md", "d.md", 0.01, -20.0),
    )


def test_recovery_separates_being_scored_from_clearing_the_cut():
    recovered = recovery(_reported(), _pairs(), [_cluster("a.md", "b.md")], 0.9)
    assert recovered["relations"] == 3
    assert recovered["scored"] == 3
    assert recovered["above_cut"] == 2
    assert recovered["co_clustered"] == 1
    subsumption = next(row for row in recovered["rows"] if row["relation"] == REL_SUBSUMED_BY)
    assert subsumption["scored"] is True and subsumption["above_cut"] is False


def test_recovery_counts_an_unreported_pair_that_outranks_a_reported_one():
    recovered = recovery(_reported(), _pairs(), [], 0.9)
    assert recovered["unreported_pairs_ranked_higher"] == 1


def test_a_relation_no_blocking_rule_generated_is_recorded_as_unscored():
    reported = (
        *_reported(),
        ReportedRelation(("g.md", "h.md"), REL_DUPLICATE, TIER_LEXICAL, 0.9),
    )
    recovered = recovery(reported, _pairs(), [], 0.9)
    assert recovered["scored"] == 3 and recovered["relations"] == 4
    missing = next(row for row in recovered["rows"] if row["doc_pair"] == ["g.md", "h.md"])
    assert missing["rank"] is None and missing["match_probability"] is None


def test_the_orderings_disagree_where_a_containment_outscores_a_jaccard():
    """The subsumption carries a tier score of 1.0 and the fit ranks it last.

    The two duplicates keep their relative order, and the two relations the tier scores equally are
    excluded rather than counted as agreement -- so one comparable ordering agrees and one, the one
    that puts a containment against a Jaccard, does not.
    """
    measured = ordering(_reported(), _pairs())
    assert measured["ranked_relations"] == 3
    assert measured["comparable_orderings"] == 2
    assert measured["discordant_orderings"] == 1
    assert measured["kendall_tau"] == 0.0


def test_decisions_compare_the_cut_against_the_duplicate_relations_only():
    decided = decisions(_reported(), _pairs(), 0.9, DUPLICATE_RELATIONS)
    assert decided["threshold_duplicates"] == 2
    assert decided["agreed"] == 2
    assert decided["fit_only"] == [] and decided["thresholds_only"] == []


def test_duplicate_probabilities_are_unrounded_and_skip_unscored_pairs():
    values = duplicate_probabilities(_reported(), _pairs(), DUPLICATE_RELATIONS)
    assert sorted(values) == [0.91, 0.99]
