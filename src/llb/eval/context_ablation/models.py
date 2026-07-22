"""Types and vocabulary of the RAG-versus-long-context ablation.

A leaderboard row says how well a model answers WITH retrieval; it never says how much of that
score retrieval paid for. Three lanes over the identical item set answer that: `closed_book` (no
context at all -- what the weights already know), `rag` (the run configuration as-is), and
`long_context` (the item's whole source document laid into the prompt). Two derived numbers make
the question explicit -- retrieval uplift (`rag - closed_book`) and the long-context delta
(`long_context - rag`) -- and a per-item contamination flag names the items the model answers
without any evidence at all.

The lanes are DIAGNOSTIC. `rag` stays the leaderboard row; nothing here changes a ranking policy.
"""

from typing_extensions import TypedDict

from llb.rag.fusion_evidence.slices import SliceReport
from llb.rag.fusion_evidence.stats import PairedComparison

# The three context lanes; each label is also the `RunConfig.context_strategy` it selects, so a
# lane's numbers are reproducible by re-running `run-eval --context-strategy <label>`.
LANE_CLOSED_BOOK = "closed_book"
LANE_RAG = "rag"
LANE_LONG_CONTEXT = "long_context"
LANES = (LANE_CLOSED_BOOK, LANE_RAG, LANE_LONG_CONTEXT)

# Per-case columns compared across lanes, all present on every `scores.jsonl` row. `retrieval_hit`
# is reported because it is the lanes' own sanity check: it is 0.0 by construction under
# `closed_book` and 1.0 by construction under `long_context` (the gold document IS the context).
METRIC_OBJECTIVE = "objective_score"
METRIC_TOKEN_F1 = "token_f1"
METRIC_RETRIEVAL_HIT = "retrieval_hit"
METRICS = (METRIC_OBJECTIVE, METRIC_TOKEN_F1, METRIC_RETRIEVAL_HIT)

# Answer-side columns the contamination flag reads: the closed-book answer "already matches the
# reference" when the normalized strings are identical, or when every reference token appears in
# it. Both are canonical `run-eval` columns (`llb.scoring.correctness`).
CONTAMINATION_COLUMNS = ("exact", "contains")

# The derived numbers the report is really about, each a paired candidate-minus-reference delta.
DERIVED_RETRIEVAL_UPLIFT = "retrieval_uplift"
DERIVED_LONG_CONTEXT_DELTA = "long_context_delta"
# The long-context delta restricted to items no lane skipped; emitted only when something WAS
# skipped, since a skipped item scores zero and would otherwise be read as a long-context loss.
DERIVED_LONG_CONTEXT_DELTA_FITTING = "long_context_delta_fitting"

# Verdicts, in the order `decide` checks them.
VERDICT_LONG_CONTEXT_WINS = "long_context_wins"
VERDICT_RAG_PAYS_OFF = "rag_pays_off"
VERDICT_RETRIEVAL_INCONCLUSIVE = "retrieval_inconclusive"
VERDICT_NO_RETRIEVAL_GAIN = "no_retrieval_gain"
VERDICT_NO_EVIDENCE = "no_evidence"


class LaneReport(TypedDict):
    """One scored lane: its run bundles, overall metrics, every question-type slice, and skips."""

    label: str
    run_dirs: list[str]
    overall: SliceReport
    slices: dict[str, SliceReport]
    skipped_item_ids: list[str]


class DerivedComparison(TypedDict):
    """One candidate-minus-reference delta over a named item population."""

    label: str
    candidate: str
    reference: str
    metric: str
    n: int
    population: str
    paired: PairedComparison


class ContaminationReport(TypedDict):
    """Items the closed-book lane already answers -- parametric knowledge, or corpus leakage."""

    lane: str
    n: int
    n_contaminated: int
    rate: float
    item_ids: list[str]


class ItemOutcome(TypedDict):
    """Item-level paired outcome across every lane -- the small-n reviewer view."""

    item_id: str
    question_type: str | None
    contaminated: bool
    lanes: dict[str, dict[str, float]]


class ContextAblationVerdict(TypedDict):
    """Whether retrieval pays for itself on this corpus, and whether stuffing beats chunking."""

    baseline: str
    n: int
    decision: str
    reason: str
    contamination_rate: float
    skipped: dict[str, int]


class ContextAblationReport(TypedDict):
    """The full lane artifact: per-lane slices, derived deltas, contamination, item ledger."""

    n: int
    baseline: str
    metrics: list[str]
    resamples: int
    confidence: float
    seed: int
    item_ids: list[str]
    lanes: dict[str, LaneReport]
    derived: list[DerivedComparison]
    contamination: ContaminationReport
    items: list[ItemOutcome]
    verdict: ContextAblationVerdict
