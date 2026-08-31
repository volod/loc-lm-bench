"""Re-read a finished audit bundle: its own record and its own rows, with no store.

The record (`bundle_record.py`) is what a run wrote down; this is what a reader does with it. Two
questions, both answered from `summary.json` plus `findings.jsonl` alone -- no store, no corpus, and
no model call, so a bundle written on another host months ago answers exactly as it did the day it
was written:

- WHICH STAGE lost an orderable document pair, re-run under THIS build's rule. That is what lets a
  rule change be scored over an archive: the bundles where the two readings part are the evidence.
- WHAT A SMALLER CANDIDATE BUDGET would have cost. The budget is a rank cutoff into the store's own
  cosine ordering, so a smaller one returns a PREFIX of the recorded ranking -- exactly answerable,
  and refused rather than approximated past the capped prefix.

A bundle that predates a record answers "not recomputable" and keeps what its run recorded. That is
the whole point of reading the record rather than the store: no answer is a correct answer, and a
reading recomputed against a store rebuilt since the run is a wrong one.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from llb.conflicts.bundle.readings import bundle_readings
from llb.conflicts.bundle.record import (
    CHUNKS_KEY,
    documents_of,
    readable_record,
    recorded_inputs,
)
from llb.conflicts.bundle.candidate_record import CandidateRecord
from llb.conflicts.constants import COVERAGE_FIELD, TIER_CLAIM, TIER_SEMANTIC
from llb.conflicts.governance.stage import (
    LOST_PAIR_FIELD,
    LOST_PAIR_COUNTS_FIELD,
    attribution_over_returned,
    returned_doc_pairs,
)

from llb.core.contracts.common import JsonObject

# The tiers whose rows came out of the ranked candidate list, and therefore the only rows a budget
# change can take away. Hash and lexical rows compare whole documents and are budget-independent.
CANDIDATE_TIERS = (TIER_SEMANTIC, TIER_CLAIM)

# Why a bundle cannot answer at a DIFFERENT budget, though it answers its own reading fine. The
# first two are the same silence for different reasons and must not be collapsed: an older bundle is
# fixed by re-running the audit, and a run below the semantic tier by raising `--effort`.
NO_CANDIDATE_RECORD_REASON = "no ranked candidate list: this bundle predates the candidate record"
NO_STORE_BUDGET_REASON = "no ranked candidate list: this run read no store, so none was ever built"
BUDGET_BEYOND_RECORD_REASON = (
    "budget {budget} is past rank {covered}, the deepest the capped candidate record reaches "
    "({cap})"
)


@dataclass(frozen=True)
class StageReplay:
    """A bundle's attribution as its own record gives it, or why the record cannot give one.

    `attribution` carries the same `{LOST_PAIR_FIELD: ...}` shape the run recorded, so a replay is
    compared with the run by equality rather than field by field. Empty means two different things
    and `recomputable` is what tells them apart: nothing was lost (the run agrees), or nothing can
    be said (the bundle predates the record).
    """

    recomputable: bool
    attribution: JsonObject = field(default_factory=dict)
    reason: str = ""

    @property
    def lost(self) -> JsonObject | None:
        """The named pair and its stage, or None when the run lost nothing to attribute."""
        lost = self.attribution.get(LOST_PAIR_FIELD)
        return lost if isinstance(lost, dict) else None

    @property
    def stage(self) -> str | None:
        lost = self.lost
        return str(lost["stage"]) if lost else None


def recorded_attribution(summary: JsonObject) -> JsonObject:
    """The attribution the RUN wrote, in the shape a replay produces -- the thing to agree with."""
    coverage = summary.get(COVERAGE_FIELD)
    lost = coverage.get(LOST_PAIR_FIELD) if isinstance(coverage, dict) else None
    return {LOST_PAIR_FIELD: dict(lost)} if isinstance(lost, dict) else {}


def _headline(lost: JsonObject | None) -> JsonObject | None:
    """The named pair diagnosis, excluding the additive census beside it.

    Older bundles did not record the census, and a budget can resize it without moving the pair.
    Neither is a changed attribution: agreement and movement continue to mean the named pair,
    stage, reason, or knob changed.
    """
    if lost is None:
        return None
    return {key: value for key, value in lost.items() if key != LOST_PAIR_COUNTS_FIELD}


def returned_pairs_at_budget(
    rows: Sequence[JsonObject], candidates: CandidateRecord, budget: int
) -> set[tuple[str, str]]:
    """The document pairs the run would have returned had its candidate budget been `budget`.

    Hash- and lexical-tier rows are kept whatever the budget is -- they compare whole documents and
    never entered the ranking -- and the candidate rows are replaced by the recorded prefix.
    """
    kept = [row for row in rows if str(row.get("tier")) not in CANDIDATE_TIERS]
    return returned_doc_pairs(kept) | candidates.doc_pairs_within(budget)


def _pairs_at_budget(
    record: JsonObject,
    candidates: CandidateRecord | None,
    rows: Sequence[JsonObject],
    budget: int,
) -> tuple[set[tuple[str, str]] | None, str]:
    """The pairs this bundle would have returned at `budget`, or why it cannot say."""
    if candidates is None:
        # Two silences that must not be collapsed: an older bundle is fixed by re-running the
        # audit, a run below the semantic tier by raising `--effort`.
        missing = NO_STORE_BUDGET_REASON if CHUNKS_KEY not in record else NO_CANDIDATE_RECORD_REASON
        return None, missing
    if not candidates.covers(budget):
        # The refusal names the knob, because "past the record" is otherwise indistinguishable
        # from "past the corpus": one is re-runnable at a larger cap, the other is not.
        return None, BUDGET_BEYOND_RECORD_REASON.format(
            budget=budget,
            covered=candidates.covered_to_rank,
            cap=candidates.cap_phrase(),
        )
    return returned_pairs_at_budget(rows, candidates, budget), ""


def replay_attribution(
    summary: JsonObject, rows: Sequence[JsonObject], *, budget: int | None = None
) -> StageReplay:
    """Re-run the stage rule over one bundle: its `summary.json` record and its own rows.

    `budget` re-asks the same question at a different candidate budget, which is the one knob that
    changes only which pairs came back. It is refused rather than approximated when the bundle
    recorded no ranking, or when the budget reaches past the capped record.
    """
    record, refusal = readable_record(summary)
    if record is None:
        return StageReplay(False, reason=refusal)
    inputs = recorded_inputs(record)
    if budget is None:
        returned = returned_doc_pairs(rows)
    else:
        at_budget, refused = _pairs_at_budget(record, inputs.candidates, rows, budget)
        if at_budget is None:
            return StageReplay(False, reason=refused)
        returned = at_budget
    return StageReplay(
        True,
        attribution=attribution_over_returned(
            summary[COVERAGE_FIELD],
            documents_of(record),
            returned,
            inputs.chunks,
            inputs.exclusions,
        ),
    )


def budget_entry(
    summary: JsonObject, rows: Sequence[JsonObject], replay: StageReplay, budget: int
) -> JsonObject:
    """The same bundle asked what a smaller candidate budget would have cost it.

    Two answers, because the attribution alone under-reports the budget: it names ONE pair, and on a
    corpus whose corpus-first lost pair is lost at every budget the name never moves however many
    pairs the budget takes away. `document_pairs` is what actually moved -- the pairs that would
    have come back -- and `changes` is whether the named pair moved with them. `changes` is None
    unless BOTH readings exist: a budget answer that cannot be compared with the run's own answer
    says nothing about the budget.
    """
    at_budget = replay_attribution(summary, rows, budget=budget)
    comparable = at_budget.recomputable and replay.recomputable
    entry: JsonObject = {
        "budget": budget,
        "recomputable": at_budget.recomputable,
        "reason": at_budget.reason,
        "recomputed": at_budget.lost,
        "changes": _headline(at_budget.lost) != _headline(replay.lost) if comparable else None,
        "document_pairs": None,
        "run_document_pairs": len(returned_doc_pairs(rows)),
    }
    record, _ = readable_record(summary)
    candidates = recorded_inputs(record).candidates if record is not None else None
    if candidates is not None and candidates.covers(budget):
        entry["document_pairs"] = len(returned_pairs_at_budget(rows, candidates, budget))
    return entry


def replay_entry(
    label: str,
    source: str,
    summary: JsonObject,
    rows: Sequence[JsonObject],
    *,
    budget: int | None = None,
) -> JsonObject:
    """One bundle re-read, beside what it recorded -- the unit a whole archive is scored in.

    `agrees` is the question the record exists to answer, and it is None rather than False on a
    bundle that cannot be re-read: an unanswerable bundle disagrees with nothing. `readings` is the
    per-question verdict (`bundle_readings.py`), carried on every entry so an operator reads what
    this bundle can be asked instead of discovering it one refusal at a time.
    """
    replay = replay_attribution(summary, rows)
    recorded = recorded_attribution(summary)
    entry: JsonObject = {
        "label": label,
        "source": source,
        "recomputable": replay.recomputable,
        "reason": replay.reason,
        "recorded": recorded.get(LOST_PAIR_FIELD),
        "recomputed": replay.lost,
        "agrees": (
            _headline(replay.lost) == _headline(recorded.get(LOST_PAIR_FIELD))
            if replay.recomputable
            else None
        ),
        "readings": [reading.payload() for reading in bundle_readings(summary)],
    }
    if budget is not None:
        entry["at_budget"] = budget_entry(summary, rows, replay, budget)
    return entry
