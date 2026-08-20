"""The SHAPE a finished bundle records its documents in, and the three changes it has made to it.

`test_bundle_record.py` covers what the record ANSWERS. This module covers the id table those
answers are keyed on -- `documents` -- and the only three changes the record has ever made to a
shape rather than adding to it, each of which removed something that carried no information:

1. schema 4 replaced every id OUTSIDE `documents` with the document's corpus position, so an id is
   written once however many maps mention it (`document_index.py`);
2. schema 5 dropped the LABEL from a document with no ordering field to label, leaving the bare id;
3. schema 6 folded the head and tail every id shares out of the table and recorded them once
   (`document_affix.py`), leaving each entry its stem.

The gate all three are held to is the same, and it is the point of the module: which form a bundle
is in must be INVISIBLE to every reading. Older forms are rewritten here from a fresh record
(`conflict_helpers.py`) rather than frozen as a blob, so a new field cannot quietly go untested on
an older side.

Everything goes through a JSON round-trip, because the record's whole purpose is to be read off
disk months later.
"""

import json

import pytest

from llb.conflicts.bundle.record import (
    CHUNKS_KEY,
    DOC_ID_KEY,
    DOCUMENTS_KEY,
    SCHEMA_KEY,
    STAGE_INPUTS_SCHEMA_VERSION,
    RunInputs,
    documents_of,
    recorded_inputs,
    stage_attribution_inputs,
)
from llb.conflicts.constants import STAGE_INPUTS_FIELD
from llb.conflicts.bundle.document_affix import PREFIX_KEY, SUFFIX_KEY, IdAffix
from llb.conflicts.bundle.document_chunks import COMPARABLE_KEY, STORED_KEY, DocumentChunks
from llb.conflicts.bundle.document_exclusions import FLOOR_KEY, DocumentExclusions
from llb.conflicts.bundle.document_index import (
    COUNT_DEFAULT_KEY,
    EXTRA_IDS_KEY,
    DocumentInterner,
    DocumentNaming,
    interned_counts,
    named_counts,
)
from llb.conflicts.bundle.stage_replay import replay_entry

from tests.llb.conflicts.conflict_helpers import (
    FOLD_BODIES,
    PATH_SHAPED,
    RANKED,
    RANKED_COS,
    RANKED_WITH_UNDATED,
    SCATTERED,
    UNDATED,
    audit_over,
    bundle_of,
    keyed_by_id,
    labelled_documents,
    unfolded_counts,
    unfolded_documents,
)


# --- schema 4: one copy of each id -------------------------------------------------------------


def test_every_document_id_is_written_exactly_once(tmp_path):
    """The interning itself: outside `documents` the record mentions no id, only positions.

    `documents` is the id table, so a corpus of thousands pays for its ids once instead of once per
    map plus twice per candidate row.
    """
    result = audit_over(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS)
    summary, _ = bundle_of(result)
    record = summary[STAGE_INPUTS_FIELD]
    outside = json.dumps(
        {key: value for key, value in record.items() if key != DOCUMENTS_KEY}, ensure_ascii=False
    )

    assert record[SCHEMA_KEY] == STAGE_INPUTS_SCHEMA_VERSION
    assert [entry["doc_id"] for entry in record[DOCUMENTS_KEY]] == sorted(RANKED)
    for doc_id in RANKED:
        assert doc_id not in outside, doc_id
    assert EXTRA_IDS_KEY not in record, "the store and the corpus agree, so there is nothing extra"


def test_the_interned_record_is_smaller_than_the_form_it_replaces(tmp_path):
    """The whole point of the change, asserted rather than left to the measured bundles.

    The saving is the difference between an id and its ordinal, so it grows with the corpus -- but
    it must never be negative, which is what a corpus of one-character ids would test for.
    """
    result = audit_over(tmp_path / "corpus", RANKED, cos_threshold=RANKED_COS)
    summary, _ = bundle_of(result)
    record = summary[STAGE_INPUTS_FIELD]

    interned = len(json.dumps(record, ensure_ascii=False))
    by_id = len(json.dumps(keyed_by_id(record, schema=3), ensure_ascii=False))

    assert interned < by_id, f"{by_id} -> {interned} bytes"


def test_a_document_the_corpus_never_carried_is_interned_beside_the_table():
    """A store one build ahead of the corpus holds chunks the `documents` table cannot name.

    Recording such an id as a bare key would make a key sometimes an ordinal and sometimes an id,
    which no reader can tell apart -- so it joins the table instead, after the corpus documents.
    """
    ahead = DocumentChunks(stored={"a-first.md": 1, "gone.md": 2}, comparable={"a-first.md": 1})
    documents = [("a-first.md", {"effective_date": "2021-01-01"})]

    record = json.loads(
        json.dumps(stage_attribution_inputs(documents, RunInputs(chunks=ahead)), ensure_ascii=False)
    )

    assert record[EXTRA_IDS_KEY] == ["gone.md"]
    assert set(record[CHUNKS_KEY]["stored"]) == {"0", "1"}
    assert recorded_inputs(record).chunks == ahead


# --- schema 5: no label on a document with nothing to label -------------------------------------


def test_a_document_with_nothing_to_order_on_is_recorded_as_the_id_alone(tmp_path):
    """The bare-id form, and the reason it loses nothing: it reads back as the same document.

    The object it replaces held a label and no value under it, so what a reader gets from the entry
    -- this document has no `effective_date` and no `version`, which is why its pairs cannot be
    ordered -- is what the bare id says outright.
    """
    result = audit_over(tmp_path / "corpus", UNDATED, cos_threshold=RANKED_COS)
    summary, _ = bundle_of(result)
    record = summary[STAGE_INPUTS_FIELD]

    assert record[DOCUMENTS_KEY] == sorted(UNDATED)
    assert documents_of(record) == [(doc_id, {}) for doc_id in sorted(UNDATED)]
    assert recorded_inputs(record) == recorded_inputs(labelled_documents(record))


def test_dropping_the_empty_label_is_a_saving_on_undated_documents_and_free_on_dated_ones(tmp_path):
    """Where the remaining per-document cost went, and where it deliberately did not.

    A corpus with governance dates keeps every label and pays exactly what it paid before: the
    change removes a label with nothing under it, never a label.
    """

    def bytes_of(name: str, documents: dict) -> tuple[int, int]:
        record = bundle_of(audit_over(tmp_path / name, documents, cos_threshold=RANKED_COS))[0][
            STAGE_INPUTS_FIELD
        ][DOCUMENTS_KEY]
        labelled = [{DOC_ID_KEY: entry} if isinstance(entry, str) else entry for entry in record]
        return (
            len(json.dumps(record, ensure_ascii=False)),
            len(json.dumps(labelled, ensure_ascii=False)),
        )

    undated, undated_labelled = bytes_of("undated", UNDATED)
    dated, dated_labelled = bytes_of("dated", RANKED)

    assert undated < undated_labelled, f"{undated_labelled} -> {undated} bytes"
    assert dated == dated_labelled, "a dated document keeps every label it had"


# --- schema 6: the head and tail every id shares ------------------------------------------------


def test_the_head_and_tail_a_path_shaped_corpus_shares_are_recorded_once(tmp_path):
    """The fold itself: `docs/uk/` and `.txt` leave the table and are written down one time each.

    What is left per entry is the stem, so the id table stops paying for the directory and the file
    type once per document -- the last thing in it that carried no information per row.
    """
    result = audit_over(tmp_path / "corpus", PATH_SHAPED, cos_threshold=RANKED_COS)
    summary, _ = bundle_of(result)
    record = summary[STAGE_INPUTS_FIELD]

    assert (record[PREFIX_KEY], record[SUFFIX_KEY]) == ("docs/uk/", ".txt")
    assert record[DOCUMENTS_KEY] == sorted(FOLD_BODIES)
    assert documents_of(record) == [(doc_id, {}) for doc_id in sorted(PATH_SHAPED)]

    folded = len(json.dumps(record, ensure_ascii=False))
    flat = len(json.dumps(unfolded_documents(record), ensure_ascii=False))

    assert folded < flat, f"{flat} -> {folded} bytes"


def test_every_folded_id_round_trips_to_the_exact_string_it_was_written_from(tmp_path):
    """The trap the fold has to avoid: an id is a JOIN KEY, so an approximate one is worse than bytes.

    `findings.jsonl` and the store both name a document by its id, so a stem that reassembles to
    anything but the original character for character silently unjoins the record from both.
    """
    result = audit_over(tmp_path / "corpus", PATH_SHAPED, cos_threshold=RANKED_COS)
    summary, rows = bundle_of(result)
    record = summary[STAGE_INPUTS_FIELD]
    from_rows = {row["a"]["doc_id"] for row in rows} | {row["b"]["doc_id"] for row in rows}

    assert [doc_id for doc_id, _ in documents_of(record)] == sorted(PATH_SHAPED)
    assert from_rows and from_rows <= set(PATH_SHAPED)
    assert set(recorded_inputs(record).chunks.stored) == set(PATH_SHAPED)


@pytest.mark.parametrize("name,documents", [("scattered", SCATTERED), ("too-small", RANKED)])
def test_a_corpus_the_fold_cannot_pay_on_records_the_table_it_always_did(tmp_path, name, documents):
    """The other half of the decision: where the fold buys nothing it must also cost nothing.

    Two ways a corpus reaches that: it shares no head or tail at all (documents in many directories
    under mixed extensions), or it shares one too rarely to earn the two keys recording it costs.
    Both write the schema-5 table -- every entry its WHOLE id, and neither key beside it.
    """
    result = audit_over(tmp_path / name, documents, cos_threshold=RANKED_COS)
    summary, _ = bundle_of(result)
    record = summary[STAGE_INPUTS_FIELD]
    written = [
        entry if isinstance(entry, str) else entry[DOC_ID_KEY] for entry in record[DOCUMENTS_KEY]
    ]

    assert PREFIX_KEY not in record and SUFFIX_KEY not in record
    assert written == sorted(documents), "the table holds whole ids, not stems"
    assert [doc_id for doc_id, _ in documents_of(record)] == written


def test_the_fold_never_takes_more_of_an_id_than_the_id_has():
    """The suffix is taken over what the PREFIX left, so the two can never overlap into a gap.

    Ids that are runs of one character share their head and, read INDEPENDENTLY, the same tail --
    which between them are longer than the shortest id, and a stem cannot be negative. Taking the
    tail from the remainder makes that impossible by construction rather than by a length check
    somebody has to remember. The shortest id here folds to the empty stem, which is the boundary
    case: it reassembles to the affixes themselves, and that is the id it came from.
    """
    runs = ["a" * length for length in range(10, 15)]

    folded = IdAffix.over(runs)

    assert folded == IdAffix(prefix="a" * 10)
    assert [folded.stem(doc_id) for doc_id in runs] == ["", "a", "aa", "aaa", "aaaa"]
    assert [folded.expand(folded.stem(doc_id)) for doc_id in runs] == runs
    assert IdAffix.over([]) == IdAffix() and IdAffix().stem("x") == "x"


def test_an_id_the_corpus_never_carried_is_recorded_whole_rather_than_folded():
    """An extra id comes from a corpus generation this run did not audit, so it shares nothing.

    Folding the table around it would charge every document the difference to accommodate an id
    that is absent from a normal bundle, so `extra_document_ids` keeps whole ids and the fold is
    computed over the audited documents alone.
    """
    documents = [(f"docs/uk/{index:02d}.txt", {}) for index in range(40)]
    ahead = DocumentChunks(stored={"elsewhere/gone.md": 2}, comparable={})

    record = json.loads(
        json.dumps(stage_attribution_inputs(documents, RunInputs(chunks=ahead)), ensure_ascii=False)
    )

    assert (record[PREFIX_KEY], record[SUFFIX_KEY]) == ("docs/uk/", ".txt")
    assert record[EXTRA_IDS_KEY] == ["elsewhere/gone.md"]
    assert recorded_inputs(record).chunks == ahead
    assert [doc_id for doc_id, _ in documents_of(record)] == [doc_id for doc_id, _ in documents]


# --- schema 7: the count most documents share ---------------------------------------------------


def test_the_count_most_documents_share_is_recorded_once(tmp_path):
    """The fold one level down from the ids: a map's dominant count leaves its per-document entries.

    On a corpus where nearly every document stores one chunk and excludes none, the maps were
    writing that same small number out once per document -- the last thing in the record repeating
    without carrying anything.
    """
    result = audit_over(tmp_path / "corpus", PATH_SHAPED, cos_threshold=RANKED_COS)
    summary, _ = bundle_of(result)
    record = summary[STAGE_INPUTS_FIELD]
    chunks = record[CHUNKS_KEY]

    assert chunks[STORED_KEY][COUNT_DEFAULT_KEY] == 1
    assert len(chunks[STORED_KEY]) < len(PATH_SHAPED), "most documents are not written at all"

    folded = len(json.dumps(record, ensure_ascii=False))
    plain = len(json.dumps(unfolded_counts(record), ensure_ascii=False))

    assert folded < plain, f"{plain} -> {folded} bytes"
    assert recorded_inputs(record) == recorded_inputs(unfolded_counts(record))


def test_a_map_where_no_count_dominates_records_the_entries_it_always_did():
    """The other half again: a map whose documents all differ is written plainly, not defaulted.

    A default there would name one document's count and then list every other document anyway, so
    the fold is refused and the map costs exactly what it cost before.
    """
    interner = DocumentInterner([f"d{index}.md" for index in range(4)])
    distinct = {f"d{index}.md": index + 1 for index in range(4)}
    dominated = {f"d{index}.md": 1 for index in range(4)}

    plain = interned_counts(distinct, interner)
    folded = interned_counts(dominated, interner)

    assert plain == {"0": 1, "1": 2, "2": 3, "3": 4}, "no default key at all"
    assert folded[COUNT_DEFAULT_KEY] == 1 and len(folded) == 1
    naming = DocumentNaming.by_position(list(distinct))
    assert named_counts(plain, naming) == distinct
    assert named_counts(folded, naming) == dominated


def test_a_document_the_default_does_not_speak_for_is_written_out(tmp_path):
    """Two documents a default must never cover, for two different reasons.

    A corpus document whose count DIFFERS is listed with its own count -- including a zero, which is
    how a document absent from `comparable` survives a non-zero default. And an id the audited
    corpus never carried is listed whatever its count, because the fold covers the documents the
    record can enumerate and an extra is not one of them.
    """
    documents = [(f"docs/uk/{index:02d}.txt", {}) for index in range(40)]
    ahead = DocumentChunks(
        stored={**{doc_id: 1 for doc_id, _ in documents}, "elsewhere/gone.md": 4},
        comparable={doc_id: 1 for doc_id, _ in documents[:39]},
    )

    record = json.loads(
        json.dumps(stage_attribution_inputs(documents, RunInputs(chunks=ahead)), ensure_ascii=False)
    )
    stored, comparable = record[CHUNKS_KEY][STORED_KEY], record[CHUNKS_KEY][COMPARABLE_KEY]

    assert stored[COUNT_DEFAULT_KEY] == 1 and stored["40"] == 4, "the extra keeps its own count"
    assert comparable[COUNT_DEFAULT_KEY] == 1 and comparable["39"] == 0, "the odd one out is a zero"
    assert recorded_inputs(record).chunks == ahead


def test_a_recovery_floor_declines_the_fold_because_absence_is_not_zero():
    """The map where the fold would trade a distinction for bytes, and so is not offered it.

    An absent `recovery_floor` says no `--min-claim-tokens` value returns this document; a floor of
    0 says one does. The fold writes a zero back as an absence, so a floor map that defaulted would
    turn the second claim into the first.
    """
    interner = DocumentInterner([f"d{index}.md" for index in range(4)])
    floors = {"d0.md": 7, "d1.md": 7, "d2.md": 7, "d3.md": 0}

    written = DocumentExclusions(recovery_floor=floors, min_claim_tokens=25).payload(interner)

    assert COUNT_DEFAULT_KEY not in written[FLOOR_KEY]
    naming = DocumentNaming.by_position(list(floors))
    assert DocumentExclusions.from_payload(written, naming).floor_for("d3.md") == 0


# --- the gate all four are held to --------------------------------------------------------------


def test_every_reading_replays_identically_through_all_five_forms(tmp_path):
    """The migration gate: which form a bundle is in must be invisible to every reading.

    The corpus is deliberately MIXED -- dated documents and one with no ordering field at all, under
    path-shaped ids -- so the record carries a labelled entry and a bare stem side by side under a
    live fold. The older forms are that record with the count defaults written back out (schema 6),
    the id fold undone too (schema 5), the label put back (schema 4), and the positions unwound
    (schema 3 and 1).
    """
    documents = {f"corpus/docs/uk/{name}": value for name, value in RANKED_WITH_UNDATED.items()}
    result = audit_over(tmp_path / "corpus", documents, cos_threshold=RANKED_COS)
    summary, rows = bundle_of(result)
    record = summary[STAGE_INPUTS_FIELD]
    forms = {
        6: unfolded_counts(record),
        5: unfolded_documents(record),
        4: labelled_documents(record),
        3: keyed_by_id(record, schema=3),
        1: keyed_by_id(record, schema=1),
    }

    assert record[PREFIX_KEY] == "corpus/docs/uk/", (
        "the fold has to be live for the forms to differ"
    )
    assert {type(entry) for entry in record[DOCUMENTS_KEY]} == {str, dict}, "both entry forms"
    for schema, older in forms.items():
        bundle = {**summary, STAGE_INPUTS_FIELD: older}
        assert documents_of(older) == documents_of(record), schema
        assert recorded_inputs(older) == recorded_inputs(record), schema
        assert replay_entry("b", "s.json", bundle, rows, budget=1) == replay_entry(
            "b", "s.json", summary, rows, budget=1
        ), schema
