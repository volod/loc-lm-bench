"""Compare the fitted ranking against the ranking the current thresholds produce.

The lexical tier hands a reviewer a list ordered by whichever measure fired: a Jaccard for a
`duplicate`, a containment for a `subsumed_by`. Those two numbers are not on one scale -- a
containment of 0.95 and a Jaccard of 0.95 are different amounts of evidence -- so the list has no
usable order ACROSS relations, which is exactly what a single probability supplies.

This module measures the difference rather than asserting it: what the fit does with every relation
the thresholds report, how the two orderings disagree over the pairs both rank, and which pairs the
two decisions part company on.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from llb.conflicts.constants import TIER_HASH, TIER_LEXICAL
from llb.conflicts.models import Finding
from llb.core.contracts.common import JsonObject
from llb.linkage.model import LinkageCluster, LinkagePair, pairs_co_clustered

DocPair = tuple[str, str]


@dataclass(frozen=True)
class ReportedRelation:
    """One relation the current thresholds return, with the number they ranked it by."""

    doc_pair: DocPair
    relation: str
    tier: str
    score: float


def _key(left: str, right: str) -> DocPair:
    return (left, right) if left <= right else (right, left)


def reported_relations(
    findings: Sequence[Finding], settled: set[DocPair]
) -> tuple[ReportedRelation, ...]:
    """Every duplicate or subsumption the model-free tiers return, plus the pairs they IMPLY.

    The hash tier chains a duplicate group rather than listing its closure, so a group of three
    reports two pairs and settles three. All three are recovered by the current thresholds, and a
    fit that scores only the two reported ones has not reproduced the tier -- so the implied pair
    joins the list, scored at the 1.0 an identical document pair carries.
    """
    relations: list[ReportedRelation] = []
    seen: set[DocPair] = set()
    for finding in sorted(findings, key=lambda f: (f.tier, f.relation, f.doc_pair())):
        pair = finding.doc_pair()
        if finding.tier not in (TIER_HASH, TIER_LEXICAL) or pair in seen:
            continue
        seen.add(pair)
        relations.append(
            ReportedRelation(
                doc_pair=pair, relation=finding.relation, tier=finding.tier, score=finding.score
            )
        )
    for pair in sorted(settled - seen):
        relations.append(
            ReportedRelation(doc_pair=pair, relation="duplicate", tier=TIER_HASH, score=1.0)
        )
    return tuple(relations)


def duplicate_probabilities(
    reported: Sequence[ReportedRelation],
    pairs: Sequence[LinkagePair],
    duplicate_relations: Sequence[str],
) -> list[float]:
    """What the fit scored every duplicate the current thresholds report, unrounded.

    Unrounded because a cut is READ OFF these numbers: a probability rounded up before it becomes
    a threshold excludes the very pair it was derived from.
    """
    ranks = _ranked(pairs)
    return [
        ranks[relation.doc_pair][1].match_probability
        for relation in reported
        if relation.relation in duplicate_relations and relation.doc_pair in ranks
    ]


def _ranked(pairs: Sequence[LinkagePair]) -> dict[DocPair, tuple[int, LinkagePair]]:
    """Every scored pair by document pair, with its 1-based rank in the published order."""
    return {
        _key(pair.left_id, pair.right_id): (rank, pair) for rank, pair in enumerate(pairs, start=1)
    }


def _recovery_row(
    relation: ReportedRelation,
    found: tuple[int, LinkagePair] | None,
    merged: set[DocPair],
    threshold: float,
) -> JsonObject:
    """One relation's row: was it scored, where does it rank, and what was decided about it."""
    pair = found[1] if found else None
    return {
        "doc_pair": list(relation.doc_pair),
        "relation": relation.relation,
        "tier": relation.tier,
        "tier_score": round(relation.score, 4),
        "scored": found is not None,
        "rank": found[0] if found else None,
        "match_probability": round(pair.match_probability, 6) if pair else None,
        "match_weight": round(pair.match_weight, 3) if pair else None,
        "above_cut": bool(pair and pair.match_probability >= threshold),
        "co_clustered": relation.doc_pair in merged,
    }


def recovery(
    reported: Sequence[ReportedRelation],
    pairs: Sequence[LinkagePair],
    clusters: Sequence[LinkageCluster],
    threshold: float,
) -> JsonObject:
    """What the fit did with every relation the current thresholds recover.

    Four questions per relation, and they are deliberately separate: was the pair SCORED at all (a
    pair no blocking rule generated is a merge never proposed), where does it RANK, does it clear
    the run's cut, and did the clustering put its two documents in one identity. A relation can be
    ranked at the very top of the list and still sit below the cut -- which is the honest answer for
    a subsumed note, because a note absorbed into a regulation is not the same document as it.
    """
    ranks = _ranked(pairs)
    merged = pairs_co_clustered(clusters)
    rows = [
        _recovery_row(relation, ranks.get(relation.doc_pair), merged, threshold)
        for relation in reported
    ]
    ranked_rows = [row for row in rows if row["rank"] is not None]
    lowest = max((int(row["rank"]) for row in ranked_rows), default=0)
    reported_keys = {relation.doc_pair for relation in reported}
    intruders = sum(
        1 for pair, (rank, _) in ranks.items() if rank < lowest and pair not in reported_keys
    )
    return {
        "relations": len(rows),
        "scored": len(ranked_rows),
        "above_cut": sum(1 for row in rows if row["above_cut"]),
        "co_clustered": sum(1 for row in rows if row["co_clustered"]),
        # The one number that says whether the ranking SEPARATES what the thresholds report from
        # what they do not: pairs the thresholds return nothing for that outrank a reported one.
        "unreported_pairs_ranked_higher": intruders,
        "rows": rows,
    }


def ordering(reported: Sequence[ReportedRelation], pairs: Sequence[LinkagePair]) -> JsonObject:
    """How often the two rankings order one pair of relations the opposite way.

    Kendall's discordance over the relations both sides rank: every pair of relations where each
    side has a strict preference, counted as agreeing or disagreeing. Ties on either side are
    excluded from the denominator rather than counted as agreement -- a tier that scores two
    relations equally has expressed no order to agree with.
    """
    ranks = _ranked(pairs)
    scored = [
        (relation.score, ranks[relation.doc_pair][1].match_weight)
        for relation in reported
        if relation.doc_pair in ranks
    ]
    concordant = discordant = 0
    for index, (tier_left, fit_left) in enumerate(scored):
        for tier_right, fit_right in scored[index + 1 :]:
            if tier_left == tier_right or fit_left == fit_right:
                continue
            same = (tier_left > tier_right) == (fit_left > fit_right)
            concordant += same
            discordant += not same
    comparable = concordant + discordant
    return {
        "ranked_relations": len(scored),
        "comparable_orderings": comparable,
        "discordant_orderings": discordant,
        "kendall_tau": round((concordant - discordant) / comparable, 4) if comparable else None,
    }


def decisions(
    reported: Sequence[ReportedRelation],
    pairs: Sequence[LinkagePair],
    threshold: float,
    duplicate_relations: Sequence[str],
) -> JsonObject:
    """Where the fit's cut and the thresholds' cutoffs decide differently, pair by pair.

    Compared against the DUPLICATE relations only, because that is what a match probability is a
    probability of. A subsumption is a relation between two different documents and is not a
    disagreement when the fit leaves it below the cut.
    """
    duplicates = {
        relation.doc_pair for relation in reported if relation.relation in duplicate_relations
    }
    above = {
        _key(pair.left_id, pair.right_id) for pair in pairs if pair.match_probability >= threshold
    }
    return {
        "threshold": threshold,
        "threshold_duplicates": len(duplicates),
        "fit_matches": len(above),
        "agreed": len(duplicates & above),
        "fit_only": [list(pair) for pair in sorted(above - duplicates)],
        "thresholds_only": [list(pair) for pair in sorted(duplicates - above)],
    }
