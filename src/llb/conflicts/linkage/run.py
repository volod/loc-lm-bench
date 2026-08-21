"""Run the edition-linkage lane beside the audit's model-free tiers.

The lane is additive by construction. It reads the documents the hash and lexical tiers already
read, prices their duplicate and subsumption evidence as one match probability, and clusters the
result into edition groups. It changes no finding, no relation, and no threshold: what it publishes
is a RANKING and a GROUPING beside the tiers' own answer, plus the measurement of how the two
differ.

The prior is where the hash tier earns its keep twice. `probability_two_random_records_match` moves
every published probability, and on a corpus of a few dozen documents the seam's generic default is
orders of magnitude off. The pairs the hash tier settled are duplicates nobody had to judge, so
their share of the corpus's document pairs -- divided by the share of duplicates a byte-and-
normalization test can be expected to catch -- is the prior, measured rather than assumed.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec

from llb.conflicts.constants import MAX_SHINGLE_DOC_FREQUENCY
from llb.conflicts.corpus import CorpusDoc
from llb.conflicts.linkage import rankings
from llb.conflicts.linkage.constants import (
    DUPLICATE_RELATIONS,
    HASH_TIER_ASSUMED_RECALL,
    LINKAGE_MODE,
    MAX_EXPLODED_SHINGLES,
    MAX_LINKAGE_DOCUMENTS,
    MIN_LINKAGE_DOCUMENTS,
)
from llb.conflicts.linkage.editions import EditionGroup, edition_groups, editions_by_document
from llb.conflicts.linkage.records import build_edition_spec, build_records, discriminative
from llb.conflicts.models import Finding
from llb.conflicts.tiers.lexical import shingles
from llb.core.contracts.common import JsonObject
from llb.linkage.clustering import cluster_pairs
from llb.linkage.constants import DEFAULT_MATCH_THRESHOLD, DEFAULT_RANDOM_MATCH_PROBABILITY
from llb.linkage.model import LinkageResult

_LOG = logging.getLogger(__name__)

_REQUIRED_PACKAGES = ("splink", "duckdb")
PRIOR_FROM_HASH_TIER = "hash tier settled pairs"
PRIOR_FROM_DEFAULT = "seam default (the hash tier settled nothing to measure one from)"
VERDICT_RANKED = "ranked"
VERDICT_NEGATIVE = "negative"


@dataclass(frozen=True)
class EditionLinkageRun:
    """One edition-linkage reading: what was fitted, what it grouped, and how it compares."""

    summary: JsonObject
    records: tuple[JsonObject, ...] = ()
    result: LinkageResult | None = None
    groups: tuple[EditionGroup, ...] = ()
    editions_of: dict[str, str] = field(default_factory=dict)

    @property
    def declined(self) -> bool:
        return bool(self.summary.get("declined"))


def declined_reason(docs: Sequence[CorpusDoc], exploded: int) -> str | None:
    """Why the lane is not running, in the words the summary will carry (None = it runs)."""
    missing = [name for name in _REQUIRED_PACKAGES if find_spec(name) is None]
    if missing:
        return f"the linkage extra is not installed (missing: {', '.join(missing)})"
    if len(docs) < MIN_LINKAGE_DOCUMENTS:
        return (
            f"{len(docs)} documents are below the {MIN_LINKAGE_DOCUMENTS}-document floor a "
            "Fellegi-Sunter fit needs: a corpus this small has too few pairs to estimate the "
            "non-match parameters from, and the levels no pair exhibits would not train at all"
        )
    if len(docs) > MAX_LINKAGE_DOCUMENTS:
        return (
            f"{len(docs)} documents exceed the {MAX_LINKAGE_DOCUMENTS}-document cap of the "
            "exploded shingle blocking rule"
        )
    if exploded > MAX_EXPLODED_SHINGLES:
        return (
            f"{exploded} discriminative shingles exceed the {MAX_EXPLODED_SHINGLES} cap of the "
            "exploded blocking join"
        )
    return None


def hash_prior(settled_pairs: int, n_docs: int) -> JsonObject:
    """The prior that two random documents are the same document, measured off the hash tier."""
    total = n_docs * (n_docs - 1) // 2
    measured = settled_pairs / (total * HASH_TIER_ASSUMED_RECALL) if settled_pairs and total else 0
    prior = min(max(measured, DEFAULT_RANDOM_MATCH_PROBABILITY), 0.5)
    return {
        "settled_pairs": settled_pairs,
        "total_document_pairs": total,
        "assumed_hash_recall": HASH_TIER_ASSUMED_RECALL,
        "random_match_probability": prior,
        "source": PRIOR_FROM_HASH_TIER if measured else PRIOR_FROM_DEFAULT,
    }


def provisional_cut(scored: Sequence[float]) -> JsonObject:
    """The cut this run groups editions at: the seam default, or the tightest one that keeps them.

    The seam's 0.9 whenever it already merges every duplicate the current thresholds report, and
    otherwise the lowest probability among those duplicates -- the tightest cut that preserves
    every decision the corpus is audited under today. Deliberately not the midpoint between the
    classes: a non-match probability of a well-separated fit is far below anything an operator
    would adopt, and it moves with every unrelated pair the corpus gains.
    """
    lowest = min(scored, default=DEFAULT_MATCH_THRESHOLD)
    if lowest >= DEFAULT_MATCH_THRESHOLD:
        return {"cut": DEFAULT_MATCH_THRESHOLD, "source": "the seam default already keeps them all"}
    return {"cut": lowest, "source": "the lowest-scoring duplicate the current thresholds report"}


def _verdict(
    recovered: JsonObject, decided: JsonObject, groups: Sequence[EditionGroup]
) -> JsonObject:
    """Whether the fit RECOVERS what the current thresholds recover, or falls short of it.

    Recall, and only recall: a relation the blocking never generated is lost, and a duplicate the
    thresholds report that the cut does not merge is lost. Pairs the fit ranks that the thresholds
    reject are counted and reported but are not losses -- they are the candidates a ranking adds,
    and whether any of them is worth a merge is a reviewer's call, not this run's.
    """
    losses: list[str] = []
    unscored = int(recovered["relations"]) - int(recovered["scored"])
    if unscored:
        losses.append(f"{unscored} reported relation(s) no blocking rule generated")
    if decided["thresholds_only"]:
        losses.append(
            f"{len(decided['thresholds_only'])} duplicate(s) the thresholds report that the cut "
            "does not merge"
        )
    added = int(recovered["unreported_pairs_ranked_higher"])
    return {
        "verdict": VERDICT_NEGATIVE if losses else VERDICT_RANKED,
        "losses": losses,
        "statement": (
            "the fit scores every relation the current thresholds recover and merges every "
            f"duplicate among them, grouping the corpus into {len(groups)} edition group(s); "
            f"{added} pair(s) the thresholds reject rank among them, which is the candidate "
            "surface a ranking adds and not a decision this run takes"
            if not losses
            else "the fit recovers less than the current thresholds do -- recorded, not adopted: "
            + "; ".join(losses)
        ),
    }


def run_edition_linkage(
    docs: Sequence[CorpusDoc],
    findings: Sequence[Finding],
    settled: set[tuple[str, str]],
    *,
    jaccard_threshold: float,
    containment_threshold: float,
) -> EditionLinkageRun:
    """Fit the document-edition linkage, cluster it into editions, and compare the two rankings."""
    doc_shingles = [shingles(doc.body) for doc in docs]
    exploded = sum(
        len(subset) for subset in discriminative(doc_shingles, MAX_SHINGLE_DOC_FREQUENCY)
    )
    reason = declined_reason(docs, exploded)
    if reason is not None:
        _LOG.info("[conflicts] linkage lane not run: %s", reason)
        return EditionLinkageRun(
            summary={
                "mode": LINKAGE_MODE,
                "declined": True,
                "reason": reason,
                "n_documents": len(docs),
            }
        )

    from llb.linkage.engine import run_linkage

    prior = hash_prior(len(settled), len(docs))
    records = build_records(docs, doc_shingles, MAX_SHINGLE_DOC_FREQUENCY)
    spec = build_edition_spec(
        jaccard_threshold=jaccard_threshold,
        containment_threshold=containment_threshold,
        match_threshold=DEFAULT_MATCH_THRESHOLD,
        random_match_probability=float(prior["random_match_probability"]),
    )
    result = run_linkage(records, spec)

    # One cut, then everything else read at it: the fit resolves the seam's own threshold, and the
    # edition grouping, the recovery table, and the decision comparison all speak about this run's
    # provisional cut instead of two numbers a reader would have to keep apart.
    reported = rankings.reported_relations(findings, settled)
    cut = provisional_cut(
        rankings.duplicate_probabilities(reported, result.pairs, DUPLICATE_RELATIONS)
    )
    doc_ids = [doc.doc_id for doc in docs]
    clusters = cluster_pairs(doc_ids, result.pairs, float(cut["cut"]))
    governance = {doc.doc_id: doc.governance for doc in docs}
    groups = edition_groups(clusters, governance)
    recovered = rankings.recovery(reported, result.pairs, clusters, float(cut["cut"]))
    decided = rankings.decisions(reported, result.pairs, float(cut["cut"]), DUPLICATE_RELATIONS)
    summary = {
        "mode": LINKAGE_MODE,
        "n_documents": len(docs),
        "n_exploded_shingles": exploded,
        "prior": prior,
        "linkage": {
            **result.summary(),
            "untrained_levels": [
                f"{level.comparison}/{level.level}" for level in result.untrained_levels
            ],
        },
        "cut": cut,
        "editions": _edition_counts(groups),
        "edition_groups": [group.payload() for group in groups],
        "recovery": recovered,
        "ordering": rankings.ordering(reported, result.pairs),
        "decisions": decided,
    }
    summary.update(_verdict(recovered, decided, groups))
    return EditionLinkageRun(
        summary=summary,
        records=tuple(records),
        result=result,
        groups=groups,
        editions_of=editions_by_document(groups),
    )


def compact_summary(summary: JsonObject) -> JsonObject:
    """The lane's summary without its two per-row lists, for the audit's own `summary.json`.

    Both lists have a home of their own in `linkage/` -- the relation rows in
    `edition_summary.json`, the groups in `editions.jsonl` -- and repeating them in the audit
    summary would grow a file every consumer reads with rows only a linkage reader wants.
    """
    if summary.get("declined"):
        return dict(summary)
    recovered = {key: value for key, value in summary["recovery"].items() if key != "rows"}
    return {
        **{key: value for key, value in summary.items() if key != "edition_groups"},
        "recovery": recovered,
    }


def _edition_counts(groups: Sequence[EditionGroup]) -> JsonObject:
    return {
        "groups": len(groups),
        "documents_grouped": sum(group.size for group in groups),
        "largest_group": max((group.size for group in groups), default=0),
        # A group whose members the governance fields could ORDER; a group of two undated copies
        # names both as current and is not one of these.
        "with_current_edition": sum(1 for group in groups if group.basis),
    }
