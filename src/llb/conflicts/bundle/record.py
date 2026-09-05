"""What a finished audit writes down so its own readings survive the store it read.

An audit run's store moves on: the next `make build-index` collapses a different duplicate, chunks a
document differently, or is one ingest ahead of the corpus the run saw. Every question asked of the
finished run afterwards therefore has two answers -- the one that run would have given and the one
today's store gives -- and they look identical from the outside. So the run records what its
readings were read FROM, beside the coverage they explain:

- `documents`: every corpus document in corpus order, with the `effective_date` / `version` it was
  audited under -- and the id alone where it has neither, minus the head and tail every id shares
  (`document_affix.py`). Corpus order is data here, not presentation: it is what picks between two
  pairs lost at the same stage.
- `chunks`: what the store held per document, what the tier compared per document, and the copies
  the hash tier settled (`document_chunks.py`).
- `exclusions`: the exclusion reason per document, plus the floor each one would return at
  (`document_exclusions.py`).
- `candidates`: the ranked candidate list collapsed to document pairs, capped by the run's own
  budget or by a constant (`candidate_record.py`).

`documents` is also the record's ID TABLE: the last three name a document by its position in it
rather than by its id, so every id is written exactly once however many maps mention it
(`document_index.py`). That is the whole of schema 4, and a bundle at any earlier schema keys on the
id itself and reads exactly as it did. Schema 6 then takes the head and tail that one copy shares
with every other id out of the table and records them once (`document_affix.py`), so what a document
costs the record is its stem.

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

from llb.conflicts.bundle.candidate_record import CandidateRecord
from llb.conflicts.constants import COVERAGE_FIELD, STAGE_INPUTS_FIELD
from llb.conflicts.bundle.document_affix import IdAffix
from llb.conflicts.bundle.document_chunks import DocumentChunks
from llb.conflicts.bundle.document_exclusions import DocumentExclusions
from llb.conflicts.bundle.document_index import (
    EXTRA_IDS_KEY,
    DocumentInterner,
    DocumentNaming,
)
from llb.conflicts.governance.editions import ORDERING_FIELDS
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
#
# 5 drops the LABEL from a document that has nothing to label: an entry with no ordering field at
# all is the id itself instead of a one-key object. The label is kept wherever there is a value
# under it, so the only entry that changes shape is the one whose object carried no information
# beyond its id. Unlike 4, this form is self-describing -- a string is an id and an object is a
# labelled entry -- so the version marks the change for a consumer rather than being what tells the
# reader which form it holds.
#
# 6 folds the head and tail every id shares out of the table and records them once
# (`document_affix.py`), so what is left per entry is the stem. Self-describing like 5 and for the
# same reason: the two keys are present exactly when the entries are stems, and a corpus that shares
# nothing to fold writes neither key and is byte for byte a schema-5 table.
#
# 7 does the same one level down, to the COUNTS rather than the ids: a count map records the value
# most corpus documents share once, under `default`, and lists only the documents that differ
# (`document_index.py`). Self-describing again -- a map carrying a `default` key is at the new form
# -- and gated per map, so a map where no count dominates is byte for byte a schema-6 map.
STAGE_INPUTS_SCHEMA_VERSION = 7
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


def _document_entry(doc_id: str, governance: JsonObject, affix: IdAffix) -> str | JsonObject:
    """One document as the record carries it: its id, plus the ordering fields it actually has.

    The values are recorded RAW, exactly as `compare_editions` received them, so a replay orders
    the pair the same way the run did -- including the values it could not order.

    A document with NO ordering field is the id itself rather than a one-key object. Nothing is
    lost: the object it replaces carried a label and no value under it, and the bare id says the
    thing a reader wants said -- this document has nothing to order on, which is why its pairs are
    unorderable. The label stays wherever there is a value to label, so the field names are never
    positional and a later ordering field is still an additive change.

    The id itself is written minus whatever head and tail every id in the table shares, which is
    the whole id under the empty fold.
    """
    fields: JsonObject = {
        name: value
        for name in ORDERING_FIELDS
        if isinstance(value := governance.get(name), str) and value
    }
    stem = affix.stem(doc_id)
    return {DOC_ID_KEY: stem, **fields} if fields else stem


def _entry_id(entry: object, affix: IdAffix) -> str | None:
    """The document an entry names at either form, or None when it names none.

    None is not the empty stem: an entry recording an empty `doc_id` is inside the contract and
    stays a slot in the id table (under a fold it names the id the affixes spell out on their own),
    while an entry that is neither a stem nor a labelled object is not.
    """
    if isinstance(entry, str):
        return affix.expand(entry)
    if isinstance(entry, dict) and DOC_ID_KEY in entry:
        return affix.expand(str(entry[DOC_ID_KEY]))
    return None


def _entry_governance(entry: object) -> JsonObject:
    """The ordering fields an entry carries, which is none of them at the bare-id form."""
    if not isinstance(entry, dict):
        return {}
    return {name: entry[name] for name in ORDERING_FIELDS if isinstance(entry.get(name), str)}


def stage_attribution_inputs(
    documents: Sequence[tuple[str, JsonObject]], inputs: RunInputs
) -> JsonObject:
    """Everything the bundle's readings need that a finished run would otherwise lose.

    `documents` is written first because it IS the id table the rest of the record keys on; the
    extras it did not cover are only known once every other part has been written, so they are
    appended last. The fold the table is written under comes before it, so a reader meets the
    prefix and suffix before the stems they belong to.

    `extra_document_ids` is deliberately NOT folded: an extra id is one the audited corpus did not
    carry, so it need not share the corpus's head or tail, and folding the table around it would
    cost every document the saving to accommodate an id that is absent from a normal bundle.
    """
    doc_ids = [doc_id for doc_id, _ in documents]
    affix = IdAffix.over(doc_ids)
    interner = DocumentInterner(doc_ids)
    record: JsonObject = {
        SCHEMA_KEY: STAGE_INPUTS_SCHEMA_VERSION,
        **affix.payload(),
        DOCUMENTS_KEY: [
            _document_entry(doc_id, governance, affix) for doc_id, governance in documents
        ],
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
    affix = IdAffix.from_record(record)
    return [
        (doc_id, _entry_governance(entry))
        for entry in entries
        if (doc_id := _entry_id(entry, affix)) is not None
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
    affix = IdAffix.from_record(record)
    return [_entry_id(entry, affix) or "" for entry in entries]


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
    return _at_current_form(record)


def _at_current_form(record: JsonObject) -> tuple[JsonObject | None, str]:
    """Every reading is taken from the CURRENT form, whatever form the bundle was written at.

    An older record is re-encoded through the registered migration rather than read in place, so
    the forms are described once -- in the registry -- instead of once per reader. A record the
    registry cannot resolve is refused with its reason rather than read as an empty one.
    """
    from llb.artifacts.errors import ArtifactContractError
    from llb.conflicts.bundle.contract import is_current_form, stage_inputs_at_current

    if is_current_form(record):
        return record, ""
    try:
        return stage_inputs_at_current(record), ""
    except ArtifactContractError as exc:
        return None, str(exc)
