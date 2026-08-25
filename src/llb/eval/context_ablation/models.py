"""Types and vocabulary of the RAG-versus-long-context ablation.

A leaderboard row says how well a model answers WITH retrieval; it never says how much of that
score retrieval paid for. Four lanes over the identical item set answer that: `closed_book` (no
context at all -- what the weights already know), `rag` (the run configuration as-is),
`retrieved_document` (retrieve as configured, then send the whole document the top-ranked chunk
came from), and `long_context` (the item's whole GOLD source document laid into the prompt).

The derived numbers make the question explicit: retrieval uplift (`rag - closed_book`), the
long-context delta (`long_context - rag`), and then the SPLIT of that oracle gap into the part an
operator can capture without a gold label (`retrieved_document - rag`) and the part that was pure
oracle advantage (`long_context - retrieved_document`). A per-item contamination flag names the
items the model answers without any evidence at all.

`closed_book` and `long_context` are DIAGNOSTIC -- `long_context` is oracle-grounded and can only
size a ceiling. `retrieved_document` is the one lane here that an operator could ship, so it
carries its own adopt-or-reject verdict; `rag` stays the leaderboard row either way.
"""

from typing_extensions import NotRequired, TypedDict

from llb.rag.fusion_evidence.slices import SliceReport
from llb.rag.fusion_evidence.paired import PairedComparison
from llb.rag.fusion_evidence.spread import ValueSpread

# The four context lanes, ordered by how much context each one is entitled to see; each label is
# also the `RunConfig.context_strategy` it selects, so a lane's numbers are reproducible by
# re-running `run-eval --context-strategy <label>`.
LANE_CLOSED_BOOK = "closed_book"
LANE_RAG = "rag"
LANE_RETRIEVED_DOCUMENT = "retrieved_document"
LANE_LONG_CONTEXT = "long_context"
LANES = (LANE_CLOSED_BOOK, LANE_RAG, LANE_RETRIEVED_DOCUMENT, LANE_LONG_CONTEXT)

# Per-case columns compared across lanes, all present on every `scores.jsonl` row. `retrieval_hit`
# is reported because it is the lanes' own sanity check: it is 0.0 by construction under
# `closed_book` and 1.0 by construction under `long_context` (the gold document IS the context).
# Under `retrieved_document` it is the one number in the table that is NOT by construction -- the
# prompt carries whole documents, so recall@k reads DOCUMENT-level: how often the lane's selection
# rule picked a document that actually holds the answer. That is the lane's own ceiling.
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
# The capturable half of the oracle gap: the same document-sized context, chosen by RETRIEVAL
# instead of by the gold label. This is the only derived row an operator can act on directly.
DERIVED_RETRIEVED_DOCUMENT_DELTA = "retrieved_document_delta"
# The residual half: what the gold label was still worth once the unit of context already matched.
DERIVED_ORACLE_DOCUMENT_GAP = "oracle_document_gap"
# Each document-lane delta restricted to items the pair did not skip; emitted only when something
# WAS skipped, since a skipped item scores zero and would otherwise read as a document-lane loss.
DERIVED_LONG_CONTEXT_DELTA_FITTING = "long_context_delta_fitting"
DERIVED_RETRIEVED_DOCUMENT_DELTA_FITTING = "retrieved_document_delta_fitting"
DERIVED_ORACLE_DOCUMENT_GAP_FITTING = "oracle_document_gap_fitting"

# Verdicts, in the order `decide` checks them.
VERDICT_LONG_CONTEXT_WINS = "long_context_wins"
VERDICT_RAG_PAYS_OFF = "rag_pays_off"
VERDICT_RETRIEVAL_INCONCLUSIVE = "retrieval_inconclusive"
VERDICT_NO_RETRIEVAL_GAIN = "no_retrieval_gain"
VERDICT_NO_EVIDENCE = "no_evidence"

# The adopt-or-reject call on the one lane here that is a shippable configuration rather than a
# diagnostic. It is reported BESIDE the ablation verdict, never folded into it: the ablation asks
# what retrieval is worth, this asks whether to widen the unit of retrieval from chunk to document.
ADOPT_RETRIEVED_DOCUMENT = "adopt_retrieved_document"
REJECT_RETRIEVED_DOCUMENT = "reject_retrieved_document"
RETRIEVED_DOCUMENT_INCONCLUSIVE = "retrieved_document_inconclusive"
RETRIEVED_DOCUMENT_NOT_MEASURED = "retrieved_document_not_measured"

# Decoding stability: what a lane's own numbers do when the IDENTICAL configuration is scored
# again on the identical items. `reproducible` is every measured band being exactly zero; anything
# else is `drifts`, and the report then states how wide the drift is per lane.
STABILITY_REPRODUCIBLE = "reproducible"
STABILITY_DRIFTS = "drifts"

POWER_RESOLUTION_SEPARATED = "separated"
POWER_RESOLUTION_FLAT = "flat"
POWER_RESOLUTION_UNDECIDABLE = "undecidable"


class ContextWindowBinding(TypedDict):
    """Which window a document lane's skips were measured against, and which side bound it.

    A skip count is only readable next to this. The DECLARED window (host planner cap, model card,
    `max_model_len`, `context_budget`) is what a config says the model can take; the SERVED one is
    what the backend answered when asked -- Ollama serves `num_ctx` 4096 however large a window the
    GGUF advertises. The smaller of the two is what actually truncates, so it is what decides a
    skip, and `budget_source` names which side that was.
    """

    declared_max_model_len: int | None
    served_max_model_len: int | None
    budget_source: str


class LaneReport(TypedDict):
    """One scored lane: its run bundles, overall metrics, every question-type slice, and skips."""

    label: str
    run_dirs: list[str]
    overall: SliceReport
    slices: dict[str, SliceReport]
    skipped_item_ids: list[str]
    # None for a lane that never checked a document against a window (`closed_book`, `rag`), and
    # for a bundle written before the binding was recorded.
    context_window: ContextWindowBinding | None


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


class RetrievedDocumentVerdict(TypedDict):
    """Adopt or reject "retrieve the chunk, send the document" as a shippable configuration."""

    decision: str
    reason: str
    n: int
    delta: float
    captured_share: float | None
    skipped: int


class ContextAblationVerdict(TypedDict):
    """Whether retrieval pays for itself on this corpus, and whether stuffing beats chunking."""

    baseline: str
    n: int
    decision: str
    reason: str
    contamination_rate: float
    skipped: dict[str, int]
    retrieved_document: NotRequired[RetrievedDocumentVerdict]


class SliceReading(TypedDict):
    """One question type's own derived deltas, contamination, and ablation reading.

    Diagnostic by construction: it is decided on the items of a single question type, so it names
    the slices retrieval fails to pay for without ever becoming the corpus decision. The
    `retrieved_document` adoption call is absent here on purpose -- see `per_slice.py`.
    """

    slice: str
    n: int
    derived: list[DerivedComparison]
    contamination: ContaminationReport
    verdict: ContextAblationVerdict


class LongContextPowerAnalysis(TypedDict):
    """Predeclared sensitivity target plus the reading reached by the new item set."""

    method: str
    reference_artifact: str
    reference_n: int
    reference_mean: float
    reference_sample_sd: float
    minimum_detectable_delta: float
    target_power: float
    alpha: float
    required_n: int
    planned_n: int
    target_reached: bool
    selector: dict[str, str]
    variance_required_n: int
    evidence_floor_n: int | None
    binding_floor: str
    planned_target_reached: NotRequired[bool]
    realized_n: NotRequired[int]
    realized_mean: NotRequired[float]
    realized_sample_sd: NotRequired[float]
    realized_required_n: NotRequired[int]
    realized_evidence_floor_n: NotRequired[int | None]
    realized_binding_floor: NotRequired[str]
    resolvable_mde: NotRequired[float]
    realized_sd_exceeds_plan: NotRequired[bool]
    resolution: NotRequired[str]
    direction: NotRequired[str]
    reason: NotRequired[str]


class LaneDecodingSpread(TypedDict):
    """One lane's between-repeat band: how far re-running it moves its own numbers.

    `divergent_items` counts items whose objective changed at all, which is what MOVES the mean;
    `answer_divergent_items` counts items whose recorded answer text changed, which is what caused
    it. The second is read off the persisted answer PREVIEW, so a divergence past that cut is not
    counted and the number is a lower bound, and an answer can be reworded without moving its
    score at all.

    `outcome_groups` is the shape of the drift rather than its size: the sizes of the groups of
    repeats that produced the IDENTICAL per-item objective vector, in first-appearance order. `[4]`
    is a lane that reproduced throughout; `[1, 3]` is one that answered differently once and then
    settled; `[1, 1, 1, 1]` is one that never repeated itself. The three are the same half-width
    and completely different findings.
    """

    lane: str
    grounded: bool  # False for the baseline lane, whose prompt carries no evidence at all
    run_dirs: list[str]  # every bundle this lane was scored in, repeat by repeat, split by split
    objective: ValueSpread
    token_f1: ValueSpread
    match_rate: ValueSpread  # `exact`/`contains` hit rate -- the contamination rate on the baseline
    divergent_items: int
    answer_divergent_items: int
    outcome_groups: list[int]


class DecodingFloorMargin(TypedDict):
    """One derived delta read against the decoding floor of the two lanes it is taken over.

    The floor is the SUM of the two lanes' half-widths rather than their quadrature: the repeats
    are few, so the conservative bound is the honest one, and a delta that clears it is not decode
    noise.
    """

    label: str
    n: int
    delta: float
    floor: float
    clears_floor: bool
    floor_multiple: float | None  # `|delta| / floor`; null at a zero floor


class DecodingStabilityReport(TypedDict):
    """Between-repeat spread of every lane, and what it does to the derived deltas.

    The paired bootstrap prices the item sample; this prices the DECODE. A closed-book prompt
    leaves a much flatter next-token distribution than a grounded one, so the two are not equally
    reproducible and a contamination rate quoted to one decimal place has to say which it is.
    """

    repeats: int
    n: int
    baseline: str
    lanes: dict[str, LaneDecodingSpread]
    baseline_floor: float  # half-width of the baseline lane's own mean objective
    grounded_floor: float  # the widest half-width any grounded lane showed
    noise_multiple: float | None  # `baseline_floor / grounded_floor`; null at a zero grounded floor
    deltas: list[DecodingFloorMargin]
    reading: str
    reason: str


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
    slice_readings: list[SliceReading]
    contamination: ContaminationReport
    items: list[ItemOutcome]
    verdict: ContextAblationVerdict
    power_analysis: NotRequired[LongContextPowerAnalysis]
    decoding_stability: NotRequired[DecodingStabilityReport]
