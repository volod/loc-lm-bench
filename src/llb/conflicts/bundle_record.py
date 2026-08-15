"""What a finished audit writes down so its own readings survive the store it read.

An audit run's store moves on: the next `make build-index` collapses a different duplicate, chunks a
document differently, or is one ingest ahead of the corpus the run saw. Every question asked of the
finished run afterwards therefore has two answers -- the one that run would have given and the one
today's store gives -- and they look identical from the outside. So the run records what its
readings were read FROM, beside the coverage they explain:

- `documents`: every corpus document in corpus order, with the `effective_date` / `version` it was
  audited under. Corpus order is data here, not presentation -- it is what picks between two pairs
  lost at the same stage.
- `chunks`: what the store held per document, what the tier compared per document, and the copies
  the hash tier settled (`document_chunks.py`).
- `exclusions`: the exclusion reason per document, plus the floor each one would return at
  (`document_exclusions.py`).
- `candidates`: the ranked candidate list collapsed to document pairs, capped by the run's own
  budget or by a constant (`candidate_record.py`).

`documents` is also the record's ID TABLE: the last three name a document by its position in it
rather than by its id, so every id is written exactly once however many maps mention it
(`document_index.py`). That is the whole of schema 4, and a bundle at any earlier schema keys on the
id itself and reads exactly as it did.

The last three are ABSENT below the semantic tier, never empty: a run that read no store built no
accounting, no exclusion pass, and no ranking, and an empty one would say the opposite (a store that
held nothing). Chunk text and chunk ordinals are deliberately not recorded either -- the record says
what each document REACHED, never what it said, so it stays a handful of small maps on a corpus of
thousands, and the readings that would need an ordinal are refused (`bundle_readings.py` states that
boundary rather than leaving it to be discovered).

This module is the record itself: what a run writes down, and what a reader gets back. Reading a
finished BUNDLE with it -- replaying the stage attribution, re-asking it at a different candidate
budget -- is `stage_replay.py`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from llb.conflicts.candidate_record import CandidateRecord
from llb.conflicts.constants import COVERAGE_FIELD, STAGE_INPUTS_FIELD
from llb.conflicts.document_chunks import DocumentChunks
from llb.conflicts.document_exclusions import DocumentExclusions
from llb.conflicts.document_index import (
    EXTRA_IDS_KEY,
    DocumentInterner,
    DocumentNaming,
)
from llb.conflicts.governance import ORDERING_FIELDS
from llb.core.contracts.common import JsonObject

# 1 records the corpus documents with their ordering fields plus the per-document chunk accounting.
# 2 adds the per-document exclusion reasons and the ranked candidate list. 3 adds the cap the
# candidate prefix was written at. 1-3 are additive, so a schema-1 bundle still replays its stage
# (it answers the two newer questions with a refusal) and a schema-2 bundle still answers a budget
# inside its prefix -- it just cannot say what truncated that prefix.
#
# 4 is the first version that CHANGES an existing shape rather than adding to it: every document
# outside `documents` is named by its corpus position instead of by its id (`document_index.py`).
# Nothing is lost or gained by the change -- both forms resolve to the same document ids and every
# reading replays identically through either -- so the version is the only thing that tells them
# apart, and the reader keeps both.
STAGE_INPUTS_SCHEMA_VERSION = 4
INTERNED_IDS_SCHEMA_VERSION = 4
SCHEMA_KEY = "schema_version"
DOCUMENTS_KEY = "documents"
CHUNKS_KEY = "chunks"
EXCLUSIONS_KEY = "exclusions"
CANDIDATES_KEY = "candidates"
DOC_ID_KEY = "doc_id"

# Why nothing can be read from a bundle at all. Refusals rather than fallbacks: the store that would
# answer instead has been rebuilt since, and an answer from it is not this run's answer. Kept SHORT,
# because an archive sweep prints one per bundle -- the standing explanation belongs in the report's
# own prose (`report_stage_replay.py`), which says it once.
NO_RECORD_REASON = "no per-document record: this bundle predates the record"
NO_COVERAGE_REASON = "no governance coverage: no orderable-pair count to attribute"
NEWER_SCHEMA_REASON = "per-document record is schema {version}, newer than this build's {known}"


@dataclass(frozen=True)
class RunInputs:
    """What one run's semantic pass knew that the bundle would otherwise lose.

    Every field is None below the semantic tier, and that is not a default -- it is the `effort`
    reading, which is why the audit passes the whole container rather than three loose arguments
    that could each be forgotten separately.
    """

    chunks: DocumentChunks | None = None
    exclusions: DocumentExclusions | None = None
    candidates: CandidateRecord | None = None


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
    documents: Sequence[tuple[str, JsonObject]], inputs: RunInputs
) -> JsonObject:
    """Everything the bundle's readings need that a finished run would otherwise lose.

    `documents` is written first because it IS the id table the rest of the record keys on; the
    extras it did not cover are only known once every other part has been written, so they are
    appended last.
    """
    interner = DocumentInterner([doc_id for doc_id, _ in documents])
    record: JsonObject = {
        SCHEMA_KEY: STAGE_INPUTS_SCHEMA_VERSION,
        DOCUMENTS_KEY: [_document_entry(doc_id, governance) for doc_id, governance in documents],
    }
    for key, part in (
        (CHUNKS_KEY, inputs.chunks),
        (EXCLUSIONS_KEY, inputs.exclusions),
        (CANDIDATES_KEY, inputs.candidates),
    ):
        if part is not None:
            record[key] = part.payload(interner)
    if interner.extras:
        record[EXTRA_IDS_KEY] = interner.extras
    return record


def documents_of(record: JsonObject) -> list[tuple[str, JsonObject]]:
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


def _id_table(record: JsonObject) -> list[str]:
    """The recorded ids by SLOT, keeping the position of an entry `documents_of` would drop.

    `documents_of` skips a malformed entry, which is right for a reading over the documents and
    wrong for the id table -- a skipped entry there would shift every position after it and rename
    every document the maps refer to.
    """
    entries = record.get(DOCUMENTS_KEY)
    if not isinstance(entries, list):
        return []
    return [
        str(entry[DOC_ID_KEY]) if isinstance(entry, dict) and DOC_ID_KEY in entry else ""
        for entry in entries
    ]


def naming_of(record: JsonObject) -> DocumentNaming:
    """How this record names a document outside `documents`: by corpus position, or by id.

    The schema version is the only thing that separates the two forms, since a position and an id
    are both strings once they are JSON object keys. A record with no version at all is one of the
    earliest bundles, which predates the interning.
    """
    version = record.get(SCHEMA_KEY)
    if not isinstance(version, int) or version < INTERNED_IDS_SCHEMA_VERSION:
        return DocumentNaming.by_id()
    extras = record.get(EXTRA_IDS_KEY)
    return DocumentNaming.by_position(
        _id_table(record),
        [str(extra) for extra in extras] if isinstance(extras, list) else (),
    )


def recorded_inputs(record: JsonObject) -> RunInputs:
    """The three recorded parts, each None when the bundle carries no entry for it.

    Absence is read off the KEY rather than off the counts under it: a run below the semantic tier
    records no chunk accounting at all, while an empty accounting is a store that held nothing --
    which is the opposite claim.
    """
    parts: dict[str, JsonObject] = {
        key: value
        for key in (CHUNKS_KEY, EXCLUSIONS_KEY, CANDIDATES_KEY)
        if isinstance(value := record.get(key), dict)
    }
    naming = naming_of(record)
    return RunInputs(
        chunks=(
            DocumentChunks.from_payload(parts[CHUNKS_KEY], naming) if CHUNKS_KEY in parts else None
        ),
        exclusions=(
            DocumentExclusions.from_payload(parts[EXCLUSIONS_KEY], naming)
            if EXCLUSIONS_KEY in parts
            else None
        ),
        candidates=(
            CandidateRecord.from_payload(parts[CANDIDATES_KEY], naming)
            if CANDIDATES_KEY in parts
            else None
        ),
    )


def readable_record(summary: JsonObject) -> tuple[JsonObject | None, str]:
    """The bundle's record, or the short reason no reading can be taken from it."""
    record = summary.get(STAGE_INPUTS_FIELD)
    if not isinstance(record, dict) or not isinstance(record.get(DOCUMENTS_KEY), list):
        return None, NO_RECORD_REASON
    recorded_version = record.get(SCHEMA_KEY)
    version = recorded_version if isinstance(recorded_version, int) else 0
    if version > STAGE_INPUTS_SCHEMA_VERSION:
        return None, NEWER_SCHEMA_REASON.format(version=version, known=STAGE_INPUTS_SCHEMA_VERSION)
    if not isinstance(summary.get(COVERAGE_FIELD), dict):
        return None, NO_COVERAGE_REASON
    return record, ""
