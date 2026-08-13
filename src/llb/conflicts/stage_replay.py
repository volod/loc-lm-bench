"""Recompute a finished audit's stage attribution from the bundle it wrote, with no store.

The attribution (`governance_stage.py`) is decided from three things: the corpus documents that
carry an ordering field, the document pairs the run RETURNED, and the per-document chunk accounting
the semantic tier folded (`DocumentChunks`). Only the second of those survives a finished run --
`findings.jsonl` is on disk and immutable. The other two are re-derived today by reading the corpus
and rebuilding the store, and both of those move: a store rebuilt since the run collapses a
different duplicate or chunks a document differently, a re-ingested corpus carries a new
`effective_date`, and either one answers a DIFFERENT question while looking like the same recompute.

So the run writes both down, beside the coverage they explain:

- `documents`: every corpus document in corpus order, with the `effective_date` / `version` it was
  audited under. Corpus order is data here, not presentation -- it is what picks between two pairs
  lost at the same stage.
- `chunks`: what the store held per document, what the tier compared per document, and the copies
  the hash tier settled. ABSENT, never empty, on a run below the semantic tier: that absence is
  what the `effort` reading is read from, and an empty accounting would say the opposite (a store
  that held nothing).

What is deliberately NOT recorded is chunk text and chunk ordinals: the record says what each
document REACHED, never what it said, so it stays a handful of small maps on a corpus of thousands.

A bundle written before the record simply cannot answer -- and says so. That is the whole point of
reading the record rather than the store: "not recomputable" is a correct answer, and a stage
recomputed against a store rebuilt since the run is a wrong one.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from llb.conflicts.constants import COVERAGE_FIELD, STAGE_INPUTS_FIELD
from llb.conflicts.document_chunks import DocumentChunks
from llb.conflicts.governance import ORDERING_FIELDS
from llb.conflicts.governance_stage import LOST_PAIR_FIELD, lost_pair_attribution
from llb.core.contracts.common import JsonObject

# 1 records the corpus documents with their ordering fields plus the per-document chunk accounting.
STAGE_INPUTS_SCHEMA_VERSION = 1
SCHEMA_KEY = "schema_version"
DOCUMENTS_KEY = "documents"
CHUNKS_KEY = "chunks"
DOC_ID_KEY = "doc_id"

# Why a bundle cannot be re-read. All three are refusals rather than fallbacks: the store that
# would answer instead has been rebuilt since, and an answer from it is not this run's answer. Kept
# SHORT, because an archive sweep prints one per bundle -- the standing explanation belongs in the
# report's own prose (`report_stage_replay.py`), which says it once.
NO_RECORD_REASON = "no per-document record: this bundle predates the record"
NO_COVERAGE_REASON = "no governance coverage: no orderable-pair count to attribute"
NEWER_SCHEMA_REASON = "per-document record is schema {version}, newer than this build's {known}"


def _document_entry(doc_id: str, governance: JsonObject) -> JsonObject:
    """One document as the record carries it: its id, plus the ordering fields it actually has.

    The values are recorded RAW, exactly as `compare_editions` received them, so a replay orders
    the pair the same way the run did -- including the values it could not order.
    """
    entry: JsonObject = {DOC_ID_KEY: doc_id}
    for name in ORDERING_FIELDS:
        value = governance.get(name)
        if isinstance(value, str) and value:
            entry[name] = value
    return entry


def stage_attribution_inputs(
    documents: Sequence[tuple[str, JsonObject]], chunks: DocumentChunks | None
) -> JsonObject:
    """Everything `lost_pair_attribution` reads that a finished bundle would otherwise lose."""
    record: JsonObject = {
        SCHEMA_KEY: STAGE_INPUTS_SCHEMA_VERSION,
        DOCUMENTS_KEY: [_document_entry(doc_id, governance) for doc_id, governance in documents],
    }
    if chunks is not None:
        record[CHUNKS_KEY] = chunks.payload()
    return record


def _documents_of(record: JsonObject) -> list[tuple[str, JsonObject]]:
    """The recorded documents in the order they were recorded, which is corpus order."""
    entries = record.get(DOCUMENTS_KEY)
    if not isinstance(entries, list):
        return []
    return [
        (
            str(entry[DOC_ID_KEY]),
            {name: entry[name] for name in ORDERING_FIELDS if isinstance(entry.get(name), str)},
        )
        for entry in entries
        if isinstance(entry, dict) and DOC_ID_KEY in entry
    ]


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


def replay_attribution(summary: JsonObject, rows: Sequence[JsonObject]) -> StageReplay:
    """Re-run the stage rule over one bundle: its `summary.json` record and its own rows.

    No store, no corpus, and no model -- so a rule change can be scored over an archive of runs,
    and a run made on another host months ago answers exactly as it did the day it was written.
    """
    record = summary.get(STAGE_INPUTS_FIELD)
    if not isinstance(record, dict) or not isinstance(record.get(DOCUMENTS_KEY), list):
        return StageReplay(False, reason=NO_RECORD_REASON)
    recorded_version = record.get(SCHEMA_KEY)
    version = recorded_version if isinstance(recorded_version, int) else 0
    if version > STAGE_INPUTS_SCHEMA_VERSION:
        return StageReplay(
            False,
            reason=NEWER_SCHEMA_REASON.format(version=version, known=STAGE_INPUTS_SCHEMA_VERSION),
        )
    coverage = summary.get(COVERAGE_FIELD)
    if not isinstance(coverage, dict):
        return StageReplay(False, reason=NO_COVERAGE_REASON)
    chunk_record = record.get(CHUNKS_KEY)
    # Absent is the `effort` reading and empty is a store that held nothing; they are not the same
    # answer, so the None below is read off the KEY rather than off the counts under it.
    chunks = DocumentChunks.from_payload(chunk_record) if isinstance(chunk_record, dict) else None
    return StageReplay(
        True,
        attribution=lost_pair_attribution(coverage, _documents_of(record), rows, chunks),
    )


def replay_entry(
    label: str, source: str, summary: JsonObject, rows: Sequence[JsonObject]
) -> JsonObject:
    """One bundle re-read, beside what it recorded -- the unit a whole archive is scored in.

    `agrees` is the question the record exists to answer, and it is None rather than False on a
    bundle that cannot be re-read: an unanswerable bundle disagrees with nothing.
    """
    replay = replay_attribution(summary, rows)
    recorded = recorded_attribution(summary)
    return {
        "label": label,
        "source": source,
        "recomputable": replay.recomputable,
        "reason": replay.reason,
        "recorded": recorded.get(LOST_PAIR_FIELD),
        "recomputed": replay.lost,
        "agrees": replay.attribution == recorded if replay.recomputable else None,
    }
