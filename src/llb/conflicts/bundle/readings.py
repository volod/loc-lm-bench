"""Which questions a finished bundle can answer alone, and which still need the store rebuilt.

A bundle is small and a store is not, so the temptation is to record everything a re-read might
want. The cost is not symmetric though: some inputs are a handful of counters per DOCUMENT, one is
bounded by a run parameter, and the rest are per CHUNK and grow with the corpus. This module is the
verdict, per question, kept in ONE place so an operator reads it instead of inferring it -- and so a
question that is refused is refused loudly rather than answered from a store that has moved.

Recorded, because they pay for themselves:

- the STAGE that lost an orderable pair: the corpus's ordering fields (one entry per document) plus
  the per-document chunk accounting;
- WHY a document is not comparable and at what floor it returns: three counters and one integer per
  document;
- what a SMALLER CANDIDATE BUDGET would have returned: the ranked candidate list collapsed to
  document pairs and capped by the run's own budget.

Refused, with the boundary stated rather than left to be discovered:

- WHICH CHUNK a lost pair would have matched on. The answer is a chunk ordinal and a similarity per
  pair, which is bounded by neither DOCUMENTS nor any run parameter -- it is the store, re-recorded.
- what a LOWER `--min-claim-tokens` would have excluded corpus-wide. Repeated-metadata detection
  runs OVER the candidate set the floor produces, so a lower floor can re-group other chunks out of
  comparison; an exact answer needs a token count per chunk and a re-run of the grouping. The
  per-document `recovery_floor` above is the necessary condition, and is quoted as one.

Both refusals have the same fix and it is the honest one: rebuild the store and re-run the audit.
"""

from dataclasses import dataclass

from llb.conflicts.bundle.record import (
    CANDIDATES_KEY,
    CHUNKS_KEY,
    EXCLUSIONS_KEY,
    RunInputs,
    readable_record,
    recorded_inputs,
)
from llb.core.contracts.common import JsonObject

READING_STAGE = "stage_attribution"
READING_EXCLUSIONS = "exclusion_reasons"
READING_BUDGET = "candidate_budget"
READING_CHUNK_MATCH = "chunk_match"
READING_FLOOR_SWEEP = "floor_sweep"

QUESTIONS = {
    READING_STAGE: "Which stage lost an orderable document pair?",
    READING_EXCLUSIONS: "Why is a document not comparable, and at what `--min-claim-tokens` "
    "does it come back?",
    READING_BUDGET: "What would a smaller candidate budget have returned?",
    READING_CHUNK_MATCH: "Which chunk would a lost pair have matched on?",
    READING_FLOOR_SWEEP: "What would a different `--min-claim-tokens` have excluded, corpus-wide?",
}

# The two questions no bundle answers at any schema, and why the record stops there. Stated as
# standing facts rather than as a bundle's own refusal: a newer run does not make them answerable.
OUT_OF_SCOPE = {
    READING_CHUNK_MATCH: (
        "not recordable: a chunk ordinal and a similarity per pair is the store re-recorded, "
        "bounded by neither DOCUMENTS nor a run parameter -- rebuild the store and re-run"
    ),
    READING_FLOOR_SWEEP: (
        "not recordable: repeated-metadata detection runs over the candidate set the floor "
        f"produces, so an exact sweep needs a token count per chunk -- `{READING_EXCLUSIONS}` "
        "carries the per-document floor, which is a necessary condition and not a sweep"
    ),
}

_MISSING = {
    READING_EXCLUSIONS: "this bundle predates the exclusion record",
    READING_BUDGET: "this bundle predates the candidate record",
}
# A run below the semantic tier has no exclusions and no candidate list to record, so its silence
# is not the archive's silence: re-running the bundle answers nothing, raising `--effort` does.
NO_STORE = "this run read no store -- it stopped below the semantic tier; raise `--effort`"


@dataclass(frozen=True)
class Reading:
    """One question, whether this bundle answers it, and the detail an operator acts on."""

    name: str
    question: str
    available: bool
    detail: str = ""

    def payload(self) -> JsonObject:
        return {
            "name": self.name,
            "question": self.question,
            "available": self.available,
            "detail": self.detail,
        }


def _budget_detail(inputs: RunInputs) -> str:
    """How deep into the ranking the recorded prefix reaches, which is the budget ceiling.

    A prefix that stops short says what stopped it. Without that the two ways a record can end --
    the corpus ranked no more, or this run declined to write more down -- read identically, and
    only the second has a knob.
    """
    candidates = inputs.candidates
    if candidates is None:
        return ""
    if candidates.covered_to_rank >= candidates.total_pairs:
        reach = "the whole candidate list"
    else:
        depth = f"rank {candidates.covered_to_rank} of {candidates.total_pairs}"
        reach = f"{depth}, {candidates.cap_phrase()}"
    return f"answers any budget up to {reach} ({len(candidates.entries)} document pairs recorded)"


def bundle_readings(summary: JsonObject) -> list[Reading]:
    """Every question this bundle is asked, answered available-or-not with its reason.

    The list is the same length whatever the bundle is, including one that carries no record at
    all: a question missing from the table reads as a question nobody asks, and the point of the
    table is that these are asked of every finished run.
    """
    record, refusal = readable_record(summary)
    if record is None:
        return [
            Reading(name, QUESTIONS[name], False, OUT_OF_SCOPE.get(name, refusal))
            for name in QUESTIONS
        ]
    inputs = recorded_inputs(record)
    below_store = CHUNKS_KEY not in record
    present = {READING_EXCLUSIONS: EXCLUSIONS_KEY, READING_BUDGET: CANDIDATES_KEY}
    readings = [Reading(READING_STAGE, QUESTIONS[READING_STAGE], True)]
    for name, key in present.items():
        available = key in record
        detail = _budget_detail(inputs) if name == READING_BUDGET and available else ""
        missing = NO_STORE if below_store else _MISSING[name]
        readings.append(Reading(name, QUESTIONS[name], available, detail if available else missing))
    readings += [
        Reading(name, QUESTIONS[name], False, reason) for name, reason in OUT_OF_SCOPE.items()
    ]
    return readings
