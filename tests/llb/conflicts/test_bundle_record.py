"""The two readings a finished bundle gained, and the boundary it keeps.

A bundle already re-reads its STAGE attribution without the store that run held. Two more questions
were asked of a finished run and still reached for that store, and a store rebuilt since answers
about itself:

1. WHY a document is not comparable, and at what `--min-claim-tokens` it comes back. Recorded as
   three counters and one floor per document, and pinned here by RE-RUNNING at the floor the record
   names -- a floor that does not recover the document is a floor nobody should act on.
2. What a SMALLER CANDIDATE BUDGET would have returned. Recorded as the ranked candidate list
   collapsed to document pairs, and pinned here against the rows the run actually wrote.

And the boundary: the questions whose input is per CHUNK are refused at every schema, said out loud
in the readings table rather than discovered when an answer turns out wrong.

Everything goes through a JSON round-trip, because the record's whole purpose is to be read off
disk months later.
"""

import json
from pathlib import Path

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.bundle_readings import (
    OUT_OF_SCOPE,
    READING_BUDGET,
    READING_CHUNK_MATCH,
    READING_EXCLUSIONS,
    READING_FLOOR_SWEEP,
    READING_STAGE,
    bundle_readings,
)
from llb.conflicts.candidate_record import (
    CAP_KEY,
    DEFAULT_CANDIDATE_RECORD_PAIRS,
    UNRECORDED_CAP,
    UNRECORDED_CAP_PHRASE,
    CandidateRecord,
)
from llb.conflicts.constants import STAGE_INPUTS_FIELD, TIER_HASH, TIER_SEMANTIC
from llb.conflicts.document_chunks import DocumentChunks
from llb.conflicts.document_exclusions import DocumentExclusions
from llb.conflicts.governance_stage import (
    LOST_PAIR_FIELD,
    STAGE_CANDIDATES,
    STAGE_CLAIM_FLOOR,
    attribution_over_returned,
)
from llb.conflicts.models import AuditResult
from llb.conflicts.report_stage_replay import budget_line, replay_report
from llb.conflicts.bundle_record import (
    CANDIDATES_KEY,
    CHUNKS_KEY,
    EXCLUSIONS_KEY,
    SCHEMA_KEY,
)
from llb.conflicts.stage_replay import (
    CANDIDATE_TIERS,
    NO_CANDIDATE_RECORD_REASON,
    NO_STORE_BUDGET_REASON,
    replay_attribution,
    replay_entry,
)

from conflict_helpers import FAKE_COS_THRESHOLD, dated_body, dated_corpus, store_over

SHORT = "коротка примітка"

# Two documents a store holds and the comparison drops, for the two DIFFERENT reasons the floor
# knob can and cannot move: one is under the floor, the other is nothing but front matter.
BELOW_THE_FLOOR = {
    "old.md": ("2021-01-01", dated_body(0)),
    "new.md": ("2024-01-01", SHORT),
}

# Three dated documents whose candidate ranking is a chain: `a` matches `b` strongly and `c` weakly,
# and `b` and `c` share nothing. Every pair is orderable, so which one the attribution names is
# decided purely by how deep into the ranking the budget reaches.
RANKED = {
    "a-first.md": ("2021-01-01", f"{dated_body(0, 30)} {dated_body(100, 10)}"),
    "b-second.md": ("2022-01-01", f"{dated_body(0, 30)} {dated_body(200, 10)}"),
    "c-third.md": ("2023-01-01", f"{dated_body(100, 10)} {dated_body(300, 30)}"),
}
# Between the b/c pair's cosine and the a/c pair's, so the ranking has a tail the budget cuts.
RANKED_COS = 0.89


def _audit(
    root: Path,
    documents: dict[str, tuple[str, str]],
    *,
    effort: str = TIER_SEMANTIC,
    cos_threshold: float = FAKE_COS_THRESHOLD,
    min_claim_tokens: int | None = None,
    candidate_budget: int | None = None,
    record_cap: int | None = None,
) -> AuditResult:
    """One real audit over a throwaway dated corpus, with a store only from the semantic tier up."""
    corpus = dated_corpus(root, documents)
    params = AuditParams(
        effort=effort,
        cos_threshold=cos_threshold,
        max_candidate_pairs=candidate_budget,
        max_candidate_record_pairs=record_cap,
    )
    if min_claim_tokens is not None:
        params.min_claim_tokens = min_claim_tokens
    store = store_over(corpus) if effort == TIER_SEMANTIC else None
    return run_audit(corpus, params, store=store)


def _bundle(result: AuditResult) -> tuple[dict, list]:
    """The two files a replay reads, through JSON exactly as `write_audit` puts them on disk."""
    return (
        json.loads(json.dumps(result.summary(), ensure_ascii=False)),
        json.loads(json.dumps(result.rows(), ensure_ascii=False)),
    )


# --- reading 1: why a document is not comparable, and the floor that returns it ------------------


def test_the_floor_the_record_names_is_the_floor_that_recovers_the_document(tmp_path):
    """The acceptance test for a recorded floor: re-run at it and the pair must come back.

    A knob printed as a VALUE is only worth more than "lower it" if the value works, so this is
    asserted against a second live run rather than against the record that produced it.
    """
    result = _audit(tmp_path / "corpus", BELOW_THE_FLOOR)
    summary, _ = _bundle(result)
    exclusions = DocumentExclusions.from_payload(summary[STAGE_INPUTS_FIELD][EXCLUSIONS_KEY])
    floor = exclusions.floor_for("new.md")

    assert result.governance_coverage[LOST_PAIR_FIELD]["stage"] == STAGE_CLAIM_FLOOR
    assert floor is not None and floor < result.params["min_claim_tokens"]

    recovered = _audit(tmp_path / "recovered", BELOW_THE_FLOOR, min_claim_tokens=floor)

    assert recovered.stage_inputs[CHUNKS_KEY]["comparable"].get("new.md")
    assert recovered.governance_coverage.get(LOST_PAIR_FIELD, {}).get("stage") != STAGE_CLAIM_FLOOR


def test_the_exclusion_counts_are_disjoint_and_account_for_every_dropped_chunk(tmp_path):
    """One reason per excluded chunk: the three counters must reconstruct `stored - comparable`."""
    result = _audit(tmp_path / "corpus", BELOW_THE_FLOOR)
    summary, _ = _bundle(result)
    record = summary[STAGE_INPUTS_FIELD]
    exclusions = DocumentExclusions.from_payload(record[EXCLUSIONS_KEY])
    chunks = record[CHUNKS_KEY]

    for doc_id, stored in chunks["stored"].items():
        excluded = sum(count for _, count in exclusions.reasons_for(doc_id))
        assert stored - chunks["comparable"].get(doc_id, 0) == excluded, doc_id


def test_a_document_no_floor_can_recover_is_told_so_rather_than_given_a_value():
    """Front matter is excluded BEFORE the floor is consulted, so the floor knob cannot move it."""
    exclusions = DocumentExclusions(front_matter={"cover.md": 2}, min_claim_tokens=25)
    chunks = DocumentChunks(stored={"cover.md": 2, "body.md": 1}, comparable={"body.md": 1})
    documents = [
        ("body.md", {"effective_date": "2021-01-01"}),
        ("cover.md", {"effective_date": "2024-01-01"}),
    ]

    lost = attribution_over_returned(
        {"orderable_document_pairs": 1}, documents, set(), chunks, exclusions
    )[LOST_PAIR_FIELD]

    assert exclusions.floor_for("cover.md") is None
    assert (lost["stage"], exclusions.phrase_for("cover.md")) == (
        STAGE_CLAIM_FLOOR,
        "2 front matter",
    )
    assert "2 front matter" in lost["reason"]
    assert "no `--min-claim-tokens` value returns this pair" in lost["knob"]


def test_the_record_stays_linear_in_documents(tmp_path):
    """The size bound: past the cap, twice the corpus is about twice the record, never its pair space.

    Every part of the record is either one entry per DOCUMENT or a prefix capped by a run
    parameter, so doubling a corpus whose candidate list saturates must not square the bytes.
    """

    def recorded(count: int) -> tuple[int, CandidateRecord]:
        documents = {
            f"d{index:03d}.md": (f"20{index + 10:02d}-01-01", dated_body(index * 5, 40))
            for index in range(count)
        }
        result = _audit(tmp_path / f"corpus{count}", documents, cos_threshold=0.0)
        return (
            len(json.dumps(result.stage_inputs, ensure_ascii=False)),
            CandidateRecord.from_payload(result.stage_inputs[CANDIDATES_KEY]),
        )

    small, small_candidates = recorded(25)
    large, large_candidates = recorded(50)

    assert small_candidates.total_pairs > DEFAULT_CANDIDATE_RECORD_PAIRS, "the cap has to bite"
    assert len(small_candidates.entries) == len(large_candidates.entries)
    assert large_candidates.total_pairs > small_candidates.total_pairs, "the pair space grew"
    # Quadratic growth would be ~4x; the cap and the per-document entries keep it under 2x.
    assert large < small * 2, f"{small} -> {large} bytes"


def test_the_exclusion_record_round_trips_through_json():
    """`payload` / `from_payload` is how the reading survives the store, so it is pinned directly."""
    exclusions = DocumentExclusions(
        front_matter={"a.md": 1},
        low_content={"a.md": 2, "b.md": 1},
        metadata_block={"b.md": 3},
        recovery_floor={"a.md": 6, "b.md": 9},
        min_claim_tokens=25,
    )

    assert (
        DocumentExclusions.from_payload(json.loads(json.dumps(exclusions.payload()))) == exclusions
    )


# --- reading 2: what a smaller candidate budget would have returned ------------------------------


def test_the_recorded_candidate_pairs_are_the_pairs_the_run_returned(tmp_path):
    """At its own budget the record must reproduce the run: same document pairs, no more, no less."""
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS)
    summary, rows = _bundle(result)
    candidates = CandidateRecord.from_payload(summary[STAGE_INPUTS_FIELD][CANDIDATES_KEY])
    returned = {
        tuple(sorted([row["a"]["doc_id"], row["b"]["doc_id"]]))
        for row in rows
        if row["tier"] in CANDIDATE_TIERS
    }

    assert candidates.total_pairs >= 2, "the corpus has to rank two pairs for a budget to matter"
    assert candidates.doc_pairs_within(candidates.total_pairs) == returned


def test_a_smaller_budget_moves_the_attribution_to_the_pair_it_drops(tmp_path):
    """The reading itself: the same bundle, asked what a shallower candidate list would have cost."""
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS)
    summary, rows = _bundle(result)

    at_full = replay_attribution(summary, rows)
    at_one = replay_attribution(summary, rows, budget=1)

    assert at_full.recomputable and at_one.recomputable
    assert at_full.stage == at_one.stage == STAGE_CANDIDATES
    assert at_full.lost["documents"] != at_one.lost["documents"], "the dropped pair is now named"
    assert at_one.lost["documents"] == ["a-first.md", "c-third.md"]


def test_the_budget_reading_counts_the_pairs_the_named_pair_cannot_show(tmp_path):
    """A budget that takes pairs away without moving the NAME still has to report the loss."""
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS)
    summary, rows = _bundle(result)

    at_budget = replay_entry("ranked", "s.json", summary, rows, budget=1)["at_budget"]

    assert at_budget["run_document_pairs"] == 2
    assert at_budget["document_pairs"] == 1, "the budget-1 prefix returns only the top pair"
    assert "1 of the run's 2 document pairs return" in budget_line(
        {"label": "ranked", "at_budget": at_budget}
    )


def test_a_run_below_the_semantic_tier_refuses_the_budget_by_naming_the_store(tmp_path):
    """Never "predates the record": the fix for an `--effort hash` run is `--effort`, not a re-run."""
    result = _audit(tmp_path / "corpus", RANKED, effort=TIER_HASH)
    summary, rows = _bundle(result)

    refused = replay_attribution(summary, rows, budget=1)

    assert (refused.recomputable, refused.reason) == (False, NO_STORE_BUDGET_REASON)


def test_a_budget_past_the_recorded_prefix_is_refused_rather_than_guessed():
    """A truncated record answers about the ranks it holds and says so about the ranks it does not."""
    capped = CandidateRecord(entries=((1, "a.md", "b.md", 0.9),), total_pairs=50, covered_to_rank=3)

    assert capped.covers(3) and not capped.covers(4)
    assert capped.doc_pairs_within(3) == {("a.md", "b.md")}


def test_the_candidate_record_stops_at_its_cap_and_says_how_far_it_reaches(tmp_path):
    """The size bound: a record capped at N document pairs never records the N+1st."""
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS)
    summary, _ = _bundle(result)
    full = CandidateRecord.from_payload(summary[STAGE_INPUTS_FIELD][CANDIDATES_KEY])

    assert full.covers(full.total_pairs), "uncapped, the record answers for the whole ranking"

    store = store_over(dated_corpus(tmp_path / "corpus", RANKED))
    pairs = [(0, 1, 0.9), (0, 2, 0.8), (1, 2, 0.7)]
    capped = CandidateRecord.of(pairs, store.chunks, limit=1)

    assert len(capped.entries) == 1
    assert capped.total_pairs == 3
    assert capped.covered_to_rank == 1 and not capped.covers(2)


def test_the_cap_is_a_run_parameter_recorded_beside_the_prefix_it_produced(tmp_path):
    """A truncated prefix and a short ranking look identical, so the record says which it is.

    The cap is what an operator who re-reads deeply raises, and a bundle that does not carry it
    leaves them guessing whether the corpus ran out or the run stopped writing.
    """
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS, record_cap=1)
    summary, rows = _bundle(result)
    capped = CandidateRecord.from_payload(summary[STAGE_INPUTS_FIELD][CANDIDATES_KEY])

    assert (capped.cap, len(capped.entries)) == (1, 1)
    assert result.params["max_candidate_record_pairs"] == 1
    assert capped.covers(1) and not capped.covers(2)

    refused = replay_attribution(summary, rows, budget=2)

    assert not refused.recomputable
    assert "capped at 1 document pairs" in refused.reason
    assert "`--max-candidate-record-pairs`" in refused.reason


def test_the_record_cap_overrides_the_runs_own_candidate_budget(tmp_path):
    """The two knobs are different questions: how many pairs the RUN returns, and how many it records.

    Left alone the record follows the run's budget, since a re-read only ever asks downward. Set,
    it wins -- a corpus that is re-read deeper pays for the depth rather than losing the reading.
    """
    followed = _audit(tmp_path / "followed", RANKED, cos_threshold=RANKED_COS, candidate_budget=1)
    overridden = _audit(
        tmp_path / "overridden",
        RANKED,
        cos_threshold=RANKED_COS,
        candidate_budget=1,
        record_cap=50,
    )

    assert CandidateRecord.from_payload(followed.stage_inputs[CANDIDATES_KEY]).cap == 1
    assert CandidateRecord.from_payload(overridden.stage_inputs[CANDIDATES_KEY]).cap == 50


def test_a_schema_two_bundle_answers_inside_its_prefix_and_says_the_cap_is_unknown(tmp_path):
    """The seam: the cap is additive, so an older bundle keeps every answer it already gave."""
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS, record_cap=1)
    summary, rows = _bundle(result)
    older = summary[STAGE_INPUTS_FIELD]
    older[SCHEMA_KEY] = 2
    older[CANDIDATES_KEY].pop(CAP_KEY)

    assert replay_attribution(summary, rows, budget=1).recomputable
    refused = replay_attribution(summary, rows, budget=2)
    assert not refused.recomputable and UNRECORDED_CAP_PHRASE in refused.reason
    assert CandidateRecord.from_payload(older[CANDIDATES_KEY]).cap == UNRECORDED_CAP


def test_a_bundle_without_a_candidate_record_refuses_the_budget_question(tmp_path):
    """A bundle from before the record answers its stage and refuses the budget -- never both."""
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS)
    summary, rows = _bundle(result)
    summary[STAGE_INPUTS_FIELD].pop(CANDIDATES_KEY)

    assert replay_attribution(summary, rows).recomputable, "the stage is unaffected"
    refused = replay_attribution(summary, rows, budget=1)
    assert (refused.recomputable, refused.reason) == (False, NO_CANDIDATE_RECORD_REASON)


# --- the older schema, and the boundary ---------------------------------------------------------


def test_a_schema_one_bundle_still_replays_its_stage(tmp_path):
    """The migration seam: the new keys are additive, so an older bundle loses no reading it had."""
    result = _audit(tmp_path / "corpus", BELOW_THE_FLOOR)
    summary, rows = _bundle(result)
    older = summary[STAGE_INPUTS_FIELD]
    older[SCHEMA_KEY] = 1
    older.pop(EXCLUSIONS_KEY)
    older.pop(CANDIDATES_KEY)

    replay = replay_attribution(summary, rows)
    named = {reading.name: reading for reading in bundle_readings(summary)}

    assert replay.recomputable and replay.stage == STAGE_CLAIM_FLOOR
    # Without the exclusion record the floor knob falls back to the disjunction, which is the
    # older reading rather than a wrong one.
    assert replay.lost["knob"] == (
        "lower `--min-claim-tokens`, or re-chunk so the claim lands in a longer chunk"
    )
    assert named[READING_STAGE].available
    assert not named[READING_EXCLUSIONS].available and not named[READING_BUDGET].available


@pytest.mark.parametrize("name", [READING_CHUNK_MATCH, READING_FLOOR_SWEEP])
def test_the_per_chunk_questions_are_refused_at_every_schema(tmp_path, name):
    """The stated boundary: a newer run does not make a per-chunk reading recordable."""
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS)
    summary, _ = _bundle(result)

    named = {reading.name: reading for reading in bundle_readings(summary)}

    assert not named[name].available
    assert named[name].detail == OUT_OF_SCOPE[name]
    assert "rebuild the store" in named[name].detail or "per chunk" in named[name].detail


def test_a_run_below_the_semantic_tier_says_it_read_no_store(tmp_path):
    """The two newer readings are absent there for a reason the operator can act on: `--effort`."""
    result = _audit(tmp_path / "corpus", BELOW_THE_FLOOR, effort=TIER_HASH)
    summary, _ = _bundle(result)

    named = {reading.name: reading for reading in bundle_readings(summary)}

    assert named[READING_STAGE].available
    assert not named[READING_EXCLUSIONS].available
    assert "read no store" in named[READING_EXCLUSIONS].detail
    assert "`--effort`" in named[READING_BUDGET].detail


# --- the operator's read ------------------------------------------------------------------------


def test_the_report_states_per_question_what_the_archive_can_answer(tmp_path):
    """The user-visible outcome: a per-question verdict, including the questions nobody can ask."""
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS)
    summary, rows = _bundle(result)
    entries = [replay_entry("ranked", "a/summary.json", summary, rows, budget=1)]

    report = replay_report(entries)

    assert "## What a bundle can answer alone" in report
    assert "| Which stage lost an orderable document pair? | 1 of 1 | -- |" in report
    assert "| Which chunk would a lost pair have matched on? | 0 of 1 |" in report
    assert "## The same bundles at candidate budget 1" in report
    assert "- attribution moves at this budget: 1 of 1" in report
    assert "MOVES the attribution" in budget_line(entries[0])


def test_a_refused_budget_names_its_knob_in_the_report_not_only_in_the_echo(tmp_path):
    """The table says "not recomputable"; the reason is what tells an operator which knob to raise."""
    result = _audit(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS, record_cap=1)
    summary, rows = _bundle(result)
    entries = [replay_entry("capped", "a/summary.json", summary, rows, budget=2)]

    report = replay_report(entries)

    assert "- refused (no record, or past the recorded prefix): 1 of 1" in report
    assert "capped at 1 document pairs" in report
    assert "`--max-candidate-record-pairs`" in report
