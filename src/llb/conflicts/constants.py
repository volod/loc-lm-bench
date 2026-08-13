"""Effort tiers, claim-relation vocabulary, and thresholds for corpus-conflict detection.

The relation vocabulary is deliberately claim-level: a relation describes how ONE claim relates
to ONE other claim, never how two documents relate as wholes. A revision that deprecates part of
an older document while restating knowledge that is still current therefore produces several
records with different relations instead of a single whole-document verdict.
"""

from collections.abc import Mapping

from llb.prep.ontology.constants import NEAR_DUP_COSINE_THRESHOLD

# --- effort tiers (cheapest first; each tier runs every tier below it) ------------------------

TIER_HASH = "hash"
TIER_LEXICAL = "lexical"
TIER_SEMANTIC = "semantic"
TIER_CLAIM = "claim"
TIERS = (TIER_HASH, TIER_LEXICAL, TIER_SEMANTIC, TIER_CLAIM)


def tier_rank(tier: str) -> int:
    """Position of `tier` in the cumulative escalation order."""
    try:
        return TIERS.index(tier)
    except ValueError:
        raise ValueError(f"unknown effort tier {tier!r}; choose one of {TIERS}") from None


def tiers_up_to(tier: str) -> tuple[str, ...]:
    """Every tier that runs when the operator asks for `tier` (tiers are cumulative)."""
    return TIERS[: tier_rank(tier) + 1]


# --- claim relations --------------------------------------------------------------------------

REL_DUPLICATE = "duplicate"
REL_SUBSUMES = "subsumes"
REL_SUBSUMED_BY = "subsumed_by"
REL_CONTRADICTS = "contradicts"
REL_SUPERSEDED_BY = "superseded_by"
REL_COMPLEMENTARY = "complementary"
RELATIONS = (
    REL_DUPLICATE,
    REL_SUBSUMES,
    REL_SUBSUMED_BY,
    REL_CONTRADICTS,
    REL_SUPERSEDED_BY,
    REL_COMPLEMENTARY,
)


def is_actionable(relation: str | None) -> bool:
    """Whether a relation means someone must DECIDE, rather than two facts coexisting.

    One definition, because the audit counts and reads the same rows: the claim-tier precision
    block measures the share of the list that is actionable, and the report promotes actionable
    rows to the head of the list. When those two sets differ, the row an operator must act on can
    sort below the rows they can ignore -- which is exactly what a precision figure is quoted to
    prevent. Anything the vocabulary does not know is actionable too: an unrecognized verdict is
    someone's problem, not a coexisting fact.
    """
    return relation is not None and relation != REL_COMPLEMENTARY


# --- the two counts of one decision group's work ------------------------------------------------
#
# A decision group carries TWO counts of "work", and they are NOT interchangeable:
#
# - TO DECIDE (`decide_rows`) is RELATION-based: `is_actionable` over the group's rows. It exists
#   the moment detection finishes, it is the set the claim-tier precision block measures, and it is
#   the only one of the two the audit can compute -- a report is written before any resolution
#   policy has been chosen.
# - TO REVIEW (`review_rows`) is POLICY-based: the rows whose resolved status is `review_required`.
#   It does not exist until an operator picks a policy, and it is the number they actually fund,
#   because it counts human decisions rather than rows the vocabulary calls work.
#
# They diverge in BOTH directions. An accepted `drop_duplicate` is to-decide and not to-review: the
# policy already settled it. Every semantic-tier duplicate is to-review under every policy, because
# that tier has no deletion authority, so a group of 100 semantic duplicates is 100 to-decide AND
# 100 to-review -- which is exactly where the two numbers look identical and an operator learns the
# wrong lesson from them. Neither number can serve both roles, so both are printed wherever both
# exist, they never share a name, and each ranking states which one it used.
DECIDE_LABEL = "to decide"
REVIEW_LABEL = "to review"


def decide_count(relation_counts: Mapping[str, int]) -> int:
    """The TO DECIDE count over a relation census -- one implementation for every consumer."""
    return sum(count for relation, count in relation_counts.items() if is_actionable(relation))


# Relations the model may return; `superseded_by` is derived, never asked for, because it needs
# the governance dates the model does not see.
MODEL_RELATIONS = (
    REL_DUPLICATE,
    REL_SUBSUMES,
    REL_SUBSUMED_BY,
    REL_CONTRADICTS,
    REL_COMPLEMENTARY,
)

# --- thresholds -------------------------------------------------------------------------------

# Word-shingle width for the lexical tier (5-grams: long enough that ordinary Ukrainian phrasing
# does not collide, short enough to survive small edits).
SHINGLE_SIZE = 5
# Skip shingles occurring in more than this share of the corpus when blocking: they are
# boilerplate, so they pair almost every document with every other while carrying no evidence.
MAX_SHINGLE_DOC_FREQUENCY = 0.8
# Jaccard at or above which two documents are near-duplicates of each other.
DEFAULT_JACCARD_THRESHOLD = 0.8
# Containment (|A and B| / |A|) at or above which the smaller document is subsumed by the larger.
DEFAULT_CONTAINMENT_THRESHOLD = 0.9
# Cosine at or above which two chunks are semantically the same claim. Shared with the ontology
# drafting dedup so "near-duplicate" means one thing across the project.
DEFAULT_COSINE_THRESHOLD = NEAR_DUP_COSINE_THRESHOLD
# Semantic prefix tree: split a node until it holds at most this many chunks.
DEFAULT_LEAF_SIZE = 32
# A chunk needs at least this many content tokens to carry a claim worth comparing. Below it the
# chunk is PDF-conversion residue -- a page marker, a `<!-- source_pdf ... -->` comment, a bare
# heading -- and matching such chunks to each other reports conversion artifacts, not knowledge.
MIN_CLAIM_TOKENS = 25
# Repeated publication/registry blocks are keyed by a normalized structural heading, then
# confirmed without a language-specific vocabulary: they must share enough corpus-wide tokens
# and each block must be dominated by variable numeric fields (dates, issue/page numbers, codes).
MIN_METADATA_BLOCK_DOCUMENTS = 2
MIN_METADATA_SHARED_TOKENS = 4
MIN_METADATA_SHARED_COVERAGE = 0.35
MIN_METADATA_NUMERIC_TOKEN_FRACTION = 0.25
# Centering estimates the corpus mean direction, which needs enough vectors to be a real estimate.
# Below this the "mean" is dominated by whichever few documents happen to be present, so centering
# would distort similarities rather than correct them, and the raw space is used instead.
MIN_CENTERING_VECTORS = 50
# Bisecting-k-means refinement passes per split (deterministic; converges well before this).
SPLIT_ITERATIONS = 8
# Floating-point slack on exact metric-tree pruning bounds.
TREE_BOUND_EPSILON = 1e-12

# --- independent-null research ------------------------------------------------------------------

# Which candidate matrix `research-conflict-nulls` runs. Each generation is motivated by the
# failure modes of the one before it, and every generation stays runnable so a later claim can be
# compared against the evidence that produced the earlier verdict.
RESEARCH_GENERATION_INITIAL = "initial"
RESEARCH_GENERATION_NEXT = "next"
RESEARCH_GENERATION_THIRD = "third"
RESEARCH_GENERATION_FOURTH = "fourth"
RESEARCH_GENERATIONS = (
    RESEARCH_GENERATION_INITIAL,
    RESEARCH_GENERATION_NEXT,
    RESEARCH_GENERATION_THIRD,
    RESEARCH_GENERATION_FOURTH,
)
# Ranked candidate rows adjudicated per corpus when measuring claim-tier precision, and traced
# control edits per corpus and edit type sent to the relation verifier. Both bound model cost.
DEFAULT_ADJUDICATION_BUDGET = 50
DEFAULT_ROLE_SAMPLES_PER_TYPE = 12
# Fourth generation. Control claims GENERATED per source document (each one costs a generation call
# plus a verification call), and how many top-cosine rows per corpus the cross-encoder re-scores.
DEFAULT_SYNTHESIS_PER_DOCUMENT = 3
DEFAULT_CROSS_ENCODER_ROWS = 50

# --- artifact names ---------------------------------------------------------------------------

CONFLICTS_METHOD = "corpus-conflicts"
# The two `summary.json` keys a RE-READ of a finished bundle addresses: the coverage the stage
# attribution rides in, and the per-document record that attribution can be recomputed from. Named
# here rather than at either reader, so the writer and the re-reader cannot part company over them.
COVERAGE_FIELD = "governance_coverage"
STAGE_INPUTS_FIELD = "stage_attribution_inputs"
FINDINGS_FILE = "findings.jsonl"
GROUPS_FILE = "groups.json"
REPORT_FILE = "report.md"
TREE_META_FILE = "tree_meta.json"
SUMMARY_FILE = "summary.json"
RESOLUTION_PLAN_FILE = "plan.json"
CONFLICT_OVERLAY_FILE = "conflict_overlay.json"
REVIEW_RECORDS_FILE = "resolution_review.jsonl"
EFFECT_REPORT_FILE = "effect.md"
EFFECT_DATA_FILE = "effect.json"
APPLIED_OVERLAY_DIR = ".llb"
APPLIED_OVERLAY_FILE = "conflict_overlay.json"
TREE_DIR = "semantic_tree"
TREE_FILE = "tree.json"
PROJECTION_FILE = "projection.json"
