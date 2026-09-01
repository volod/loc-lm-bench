# Corpus Hygiene: What a Finished Audit Bundle Can Answer Alone

An audit run writes `summary.json` and `findings.jsonl` and then the store it read moves on: the
next `make build-index` collapses a different duplicate, chunks a document differently, or is one
ingest ahead of the corpus the run saw. Every question asked of a finished run afterwards therefore
has two possible answers -- the one the run would have given, and the one today's store gives -- and
they look identical from the outside.

This page is the verdict per question: which ones the bundle answers from its own record, which ones
it refuses, and why the record stops where it does. The refusals are the point as much as the
answers: "not recomputable" is correct, and a reading recomputed against a rebuilt store is not.

## The questions and their verdicts

| question | answered from the bundle | what it costs to record |
| --- | --- | --- |
| Which STAGE lost an orderable document pair? | yes | one entry per document (its ordering fields) plus the per-document chunk accounting |
| WHY is a document not comparable, and at what `--min-claim-tokens` does it come back? | yes | three counters and one integer per document |
| What would a SMALLER CANDIDATE BUDGET have returned? | yes | the ranked candidate list collapsed to document pairs, capped by the run's own budget |
| Which CHUNK would a lost pair have matched on? | **no** | a chunk ordinal and a similarity per pair -- the store, re-recorded |
| What would a different `--min-claim-tokens` have excluded, CORPUS-WIDE? | **no** | a token count per chunk plus a re-run of the metadata grouping |

The rule the table is drawn on is a size bound, not a preference: a recorded input must be linear in
DOCUMENTS or bounded by a run parameter. The first three are; the last two are per CHUNK and grow
with the corpus, so recording them would be keeping a second copy of the store inside the bundle.
Both refusals have the same honest fix -- rebuild the store and re-run the audit.

The verdicts are computed per bundle by `bundle_readings` (`src/llb/conflicts/bundle/readings.py`)
and printed by `make recompute-conflict-stage` as a table, so an operator reads them instead of
discovering a gap when an answer turns out wrong. A question that is unavailable says WHY in the
operator's own terms: a bundle written before the record says it predates it, and a run below the
semantic tier says it read no store and names `--effort` rather than a re-run.

## The record

All of it rides under `stage_attribution_inputs` in `summary.json`
(`src/llb/conflicts/bundle/record.py` builds it, `AuditResult.stage_inputs` carries it,
`RunInputs` is what the semantic pass hands back). `schema_version` is the migration seam and is
**7**; versions 1-3 are additive, so a schema-1 bundle still replays its stage and answers the two
newer questions with a refusal, and a schema-2 bundle still answers a budget inside its prefix -- it
just cannot say what truncated that prefix. Versions 4 through 7 are the only ones that CHANGE a
shape rather than adding to it, and each removes one repetition: 4 is
[the id table itself](#the-id-table-every-document-named-once), 5 is
[the label it stopped carrying](#the-label-a-document-with-nothing-to-label-was-carrying), 6 is
[the head and tail every id shares](#the-head-and-tail-every-id-shares), and 7 is
[the count most documents share](#the-count-most-documents-share). All four obey one rule
(`src/llb/conflicts/bundle/fold.py`): a fold is taken only where it is actually smaller, so a corpus
it does not suit gets the previous form byte for byte instead of a penalty.

| key | what it carries | module |
| --- | --- | --- |
| `documents` | every corpus document in corpus order, with the `effective_date` / `version` it was audited under -- the id alone where it has neither, and the id minus whatever head and tail the whole table shares | `bundle/record.py` |
| `document_id_prefix` / `document_id_suffix` | the head and tail every id in the table shares, absent when the fold does not pay for itself | `document_affix.py` |
| `chunks` | `stored` / `comparable` / `copies` per document; the two count maps carry a `default` where one count dominates | `document_chunks.py` |
| `exclusions` | `front_matter` / `low_content` / `metadata_block` per document, plus `recovery_floor` and the run's `min_claim_tokens` | `document_exclusions.py` |
| `candidates` | the ranked candidate list collapsed to `[rank, left doc, right doc, cosine]`, with `total_pairs`, `covered_to_rank`, and the `cap` the prefix was written at | `candidate_record.py` |
| `extra_document_ids` | the ids the store held that the audited corpus did not, absent when they agree | `document_index.py` |

Why `documents` and `chunks` cannot be re-derived instead of recorded -- corpus order is data, and a
rebuilt store answers about itself -- is in
[recomputing the stage](conflict-decision-groups.md#recomputing-the-stage-from-a-finished-bundle).

`chunks`, `exclusions`, and `candidates` are ABSENT below the semantic tier, never empty: a run that
read no store built no accounting, no exclusion pass, and no ranking, and an empty one would say the
opposite (a store that held nothing).

### The id table: every document named once

The record is linear in DOCUMENTS as intended, but a document id used to be written down five times
over -- in `documents`, in `chunks.stored`, in `chunks.comparable`, in each of the three
`exclusions` maps, and twice per row of the ranked `candidates` list. That repetition carried no
information: `documents` is already the corpus-order index every other key could name a position in.
So it does. Every key outside `documents` is that POSITION -- a decimal string in the maps, an
integer in the candidate rows -- and `documents` is the table they resolve against
(`src/llb/conflicts/bundle/document_index.py`: `DocumentInterner` writes, `DocumentNaming` reads).

Two facts the change has to survive, and both are load-bearing:

- **A store can be one build ahead of the corpus**, so a chunk can carry a doc_id the audited
  `documents` list never had. Such an id is appended to `extra_document_ids` and referenced by
  position like any other. The alternative -- leaving it as a bare key -- would make a key sometimes
  an ordinal and sometimes an id, which no reader can tell apart.
- **Bundles on disk are at the older form.** `naming_of` picks the form off `schema_version`, and
  nothing else can: a position and an id are both strings once they are JSON object keys. Both forms
  resolve to the same document ids, so *every reading replays identically through either* -- checked
  over the whole archive below, and pinned in CI by
  `test_every_reading_replays_identically_through_all_five_forms`
  (`tests/llb/conflicts/test_bundle_id_table.py`), which rewrites a fresh record into the older form
  (`keyed_by_id` in `tests/llb/conflicts/conflict_helpers.py`) rather than freezing a blob, so a new
  field cannot quietly go untested on the older side.

**Measured over the whole bundle archive** (24 bundles under `.data/corpus-conflicts/`, spanning
no record at all, schema 1, and schema 2; CUDA host, no model call):

```bash
make recompute-conflict-stage STAGE_RUNS="<the 24 run dirs>" STAGE_BUDGET=2 \
  STAGE_OUT=.data/corpus-conflict-stage/20260815T-interned-ids-archive
```

The resulting `stage.json` is **byte-identical** to the sweep over the same 24 bundles taken
before the interning: same attribution, same agreement verdict, same budget answers, same refusals
and refusal reasons. That identity is not a one-off -- re-verified 2026-08-22 on this host, the
`stage.json` of every sweep in this section, across all five id-table forms, is the SAME file
(sha256 `6489f34d...`), which is the strongest form the claim can take: the record changed shape
four times and not one reading moved. Separately, every recorded bundle re-encoded through
`stage_attribution_inputs` returns the same `RunInputs` and the same `documents_of` as the bundle it
came from.

The saving is the difference between an id and its ordinal, so it grows with the id length and with
how many maps mention the document. On the largest bundle on this host -- the 250-document
SQuAD-derived quickstart corpus, 311 chunks, re-audited at cosine 0.6 (2026-08-15, RTX 4060 Ti
16 GB CUDA host, no model call):

| part | keyed by id | keyed by position | saving |
| --- | --- | --- | --- |
| `documents` (the table itself) | 9,500 | 9,500 | 0 |
| `chunks` | 14,455 | 4,794 | 67% |
| `exclusions` | 1,106 | 444 | 60% |
| `candidates` (36 pairs) | 2,437 | 896 | 63% |
| **whole record** | **27,578** | **15,714** | **43%** |

`summary.json` for that run falls from 57.7 KiB to 45.9 KiB. Below the semantic tier the record is
`documents` and nothing else, so it is unchanged at 637 bytes -- there is no repetition there to
remove. What remains is the floor the change was aiming at: **one id per document**, and on this
bundle the table is 9,500 of the 15,714 bytes left -- which is what the next section prices.

### The label a document with nothing to label was carrying

With every key outside `documents` naming a position, the table itself was the remaining cost, and
about a quarter of it was not id: every entry was a JSON object, so `"doc_id"` and each ordering
field name was written once per document. Four forms were priced -- the labelled object it started
as, a positional row `[doc_id, effective_date, version]`, a column-wise table of parallel arrays,
and the id ALONE for a document carrying no ordering field.

**What each form costs per document**, at the 250-document corpus's 22-character ids. Every one of
these forms removes a fixed number of bytes per row and none of them changes a growth term, so the
percentages hold at 250, 2,500, and 25,000 documents to within the table's own few bytes of
overhead -- the same shape as the interning above, and the reason a projection here is arithmetic
rather than a second measurement:

| form | undated document | fully dated document |
| --- | --- | --- |
| labelled object (schemas 1-4) | 38.00 B | 88.00 B |
| positional row, trailing `null`s trimmed | 28.00 B (-26.3%) | 49.00 B (-44.3%) |
| positional row, padded to fixed arity | 40.00 B (**+5.3%**) | 49.00 B (-44.3%) |
| column-wise table | 26.05 B (-31.5%) | 47.18 B (-46.4%) |
| **the id alone when undated (schema 5)** | **26.00 B (-31.6%)** | 88.00 B (0%) |

Projected onto whole tables, which is where the numbers a decision would rest on live:

| documents | labelled | positional row | column table | id alone when undated |
| --- | --- | --- | --- | --- |
| 250, undated | 9,500 | 7,000 | 6,512 | 6,500 |
| 2,500, undated | 95,000 | 70,000 | 65,012 | 65,000 |
| 25,000, undated | 950,000 | 700,000 | 650,012 | 650,000 |
| 25,000, 10% dated | 1,075,000 | 752,500 | 972,545 | 805,000 |
| 25,000, fully dated | 2,200,000 | 1,225,000 | 1,175,045 | 2,200,000 |

**Decision: take the saving where the label carries nothing, and decline the other two forms.** A
document with no ordering field is recorded as the id itself; a document with one keeps the labelled
object it always had. That is the whole of schema 5, and it takes the entire undated saving -- 26.00
bytes per document against the column table's 26.05 and the row form's 28.00 -- in the case that is
not hypothetical here: every corpus on this host except the planted 7-document fixture carries no
governance date at all, which is the same fact
[the zero-delta precondition](conflict-decision-groups.md#the-precondition-behind-a-zero-delta)
reports from the other side.

**What the other two forms cost a reader**, which is why the extra they buy on a dated corpus is
declined rather than unnoticed:

- a positional row is read by OFFSET, so `["archive-policy.md", null, "1.0"]` is legible only to a
  reader who knows `ORDERING_FIELDS` and its order -- and the record is read by hand exactly when a
  bundle disagrees with a run, which is the worst moment to need a second file to decode the first;
- pinning that order makes a later ordering field a schema BUMP rather than an additive change.
  Today a new field is a new key that older bundles simply lack, and `documents_of` reads it by
  name;
- the fixed-arity row is not even a saving on the corpora that exist here: `null`-padded it COSTS
  5%, and trimming the padding means a document with a `version` and no `effective_date` still needs
  an interior `null`;
- a column table takes one document's record out of one place: reading a single document means
  indexing three parallel arrays at the same offset, and a mis-length in one array silently
  re-labels every document after it.

The bare id charges none of that, and it says something the object did not: this document has
nothing to order on, which is precisely why its pairs are unorderable. The form is also
self-describing -- a string is an id, an object is a labelled entry -- so unlike the position/id
seam of schema 4, no reader needs `schema_version` to tell the two apart; the version is bumped for
a consumer that assumed every entry was an object, not because the reader could not cope. What was
left after it is the id STRING, which is what [the fold](#the-head-and-tail-every-id-shares) prices.

**What the decision declines is measured, not assumed**: on a fully dated corpus of 25,000
documents the positional row would save a further ~975 KB (1,225,000 against 2,200,000 bytes). If
such a corpus appears, the same schema seam that carried the interning carries that too.

**Measured, on real bundles** (2026-08-15, RTX 4060 Ti 16 GB CUDA host, no model call). The
250-document quickstart audit and the committed 7-document fixture audit were each re-taken on the
schema-5 build, so each row is the same audit as its schema-4 predecessor with only the table form
changed:

| bundle | documents | dated | `documents` | whole record | `summary.json` |
| --- | --- | --- | --- | --- | --- |
| `20260815T-bare-id-squad-cos060` | 250 | 0 | 9,500 -> 6,500 (-32%) | 15,714 -> 12,714 (-19%) | 45,885 -> 42,885 (-6.5%) |
| `20260815T-bundle-record-squad-semantic-cos080` | 250 | 0 | 9,500 -> 6,500 (-32%) | 14,909 -> 11,909 (-20%) | 42,097 -> 39,097 (-7.1%) |
| `20260815T-bare-id-fixture-semantic` | 7 | 7 | 601 -> 601 (0%) | 1,209 -> 1,209 | unchanged |
| `20260815T-bundle-record-fixture-hash` | 7 | 7 | 601 -> 601 (0%) | 637 -> 637 | unchanged |

Both squad runs are byte-for-byte the same audit as their schema-4 predecessors apart from the
table: `documents_of` and `recorded_inputs` return equal values, and every other key of the record
is identical. The dated fixture is the control -- it keeps every label and pays exactly what it
paid.

**Every reading replays identically through all three forms.** The 24-bundle archive sweep
(no record at all, schema 1, schema 2, schema 4) re-taken on the schema-5 build produces a
`stage.json` and a `stage.md` BYTE-IDENTICAL to the sweep taken on the schema-4 build before it,
and the two re-taken bundles above replay to the same attribution, the same agreement verdict, and
the same budget answer as the schema-4 and schema-2 bundles of the same runs. The same gate now
covers four forms and is [re-stated with the fold](#the-head-and-tail-every-id-shares).

### The head and tail every id shares

With the table down to one bare id per undated document, every remaining lever was on the id STRING
rather than on the shape around it. A corpus-relative id is a PATH, and a corpus is usually one
directory of one file type, so the ids share a head and a tail that carry nothing per document: the
250-document quickstart corpus writes `squad/` and `.txt` 250 times each, 2.5 KiB of the table's
6.5. So the record writes them ONCE (`document_id_prefix` / `document_id_suffix`,
`src/llb/conflicts/bundle/document_affix.py`) and each entry keeps only its stem.

**The fold is applied only where it pays for itself**, which is the whole of the "buys nothing,
costs nothing" side. `IdAffix.over` computes the shared head and tail and then compares what they
save across the table against what the two keys cost; below break-even it returns the EMPTY fold and
the table is written exactly as schema 5 wrote it, byte for byte. Break-even is small and depends
only on how much is shared:

| shared affix | bytes per document | pays from |
| --- | --- | --- |
| `squad/` + `.txt` | 10 | 6 documents |
| `.txt` alone | 4 | 8 documents |
| `.md` alone | 3 | 10 documents |
| `corpus/docs/uk/` + `.txt` | 19 | 4 documents |

The reassembly is EXACT, which matters more than the bytes: an id is the join key for
`findings.jsonl` and for the store, so a stem that reassembles to anything but the original would
silently unjoin the record from both. Two things make it exact rather than nearly so:

- the suffix is taken over what the PREFIX left behind, so the two can never overlap into a negative
  stem. Ids that are runs of one character share their head and, read independently, the same tail;
  taking the tail from the remainder makes the overlap impossible by construction rather than by a
  length check somebody has to remember;
- `extra_document_ids` is deliberately NOT folded. An extra id comes from a store one build ahead of
  the audited corpus, so it need not share that corpus's head or tail, and folding the table around
  it would charge every document the difference to accommodate an id that is absent from a normal
  bundle.

Like schema 5 and unlike schema 4, the form is self-describing: the two keys are present exactly
when the entries are stems, so no reader needs `schema_version` to tell a stem from an id. The
version is bumped for a consumer that assumed `documents` held whole ids.

**Measured, on real bundles** (CUDA host, no adjudication call -- the encoder runs only to build the
stores; compact `json.dumps` bytes, the unit the rest of this page quotes). The two `scattered`
corpora are the SAME 250 documents as the squad run,
re-laid under a different id shape (`.data/corpus-conflicts/_id_shape_corpora/`) and re-indexed at
the same e5-base settings into 311 chunks (`.data/id-fold-stores/`), so the id is the only thing
that differs -- and they return the same 38 rows over the same 36 document pairs:

| bundle | id shape | prefix | suffix | `documents` | whole record | `summary.json` |
| --- | --- | --- | --- | --- | --- | --- |
| `20260815T-id-fold-squad-cos060` | `squad/<hex>.txt` | `squad/` | `.txt` | 6,500 -> 4,000 (-38.5%) | 12,714 -> 10,276 (-19.2%) | 42,915 -> 40,477 (-5.7%) |
| `20260815T-id-fold-squad-cos080` | `squad/<hex>.txt` | `squad/` | `.txt` | 6,500 -> 4,000 (-38.5%) | 11,911 -> 9,473 (-20.5%) | 39,165 -> 36,727 (-6.2%) |
| `20260815T-id-fold-scattered-txt` | `<dir>/<hex>.txt`, 159 directories | -- | `.txt` | 5,750 -> 4,750 (-17.4%) | 11,964 -> 10,994 (-8.1%) | 46,826 -> 45,856 (-2.1%) |
| `20260815T-id-fold-scattered-mixed` | `<dir>/<hex>.txt`\|`.md` | -- | -- | 5,625 (0%) | 11,839 (0%) | 46,578 (0%) |
| `20260815T-id-fold-fixture-semantic` | 7 flat `.md` | -- | -- | 601 (0%) | 1,209 (0%) | 13,223 (0%) |
| `20260815T-id-fold-fixture-hash` | 7 flat `.md` | -- | -- | 601 (0%) | 637 (0%) | 3,276 (0%) |

The last three rows are the two ways a corpus reaches "costs nothing", and both are byte-for-byte
identical to the schema-5 table rather than nearly so: `scattered-mixed` shares no head and no tail
at all, and the 7-document fixture shares `.md` but is three documents short of the ten that earn
the key recording it. The middle row is the case the fold half-pays: the prefix is not shared, the
suffix is, and the table takes the 4 bytes per document that are actually there.

**Decision: take the fold.** It pays on the path-shaped corpus (-38.5% of the table) and costs
exactly zero where it buys nothing, which is what the keep-or-change verdict was conditioned on.
The saving is a fixed number of bytes per document -- `len(prefix) + len(suffix)`, minus the two
keys once -- so it does not change a growth term and the percentage only firms up with the corpus:

| documents | flat table | folded table | saving |
| --- | --- | --- | --- |
| 250 | 6,500 | 4,062 | -37.5% |
| 2,500 | 65,000 | 40,062 | -38.4% |
| 25,000 | 650,000 | 400,062 | -38.5% |

**What it costs a reader** is that `documents` no longer spells an id out, so a grep of
`summary.json` for `squad/0004bc24d22e.txt` finds it in `findings.jsonl`'s rows and not in the
table. That is the one objection, and it is smaller than the ones that sank the positional row and
the column table
([the label section](#the-label-a-document-with-nothing-to-label-was-carrying)): the join is one
concatenation of two values named in the same object, not an OFFSET into a field order kept in
another file, and there is no array whose mis-length silently re-labels every document after it.

**Every reading replays identically through every form.** The 24-bundle archive sweep
(no record at all, schema 1, schema 2, schema 4/5) re-taken on this build produces a `stage.json`
and a `stage.md` that are BYTE-IDENTICAL to the pre-change sweep
against the sweep taken on the schema-5 build before it, and the two bundles above that have a
schema-5 predecessor -- the 250-document audit at cosine 0.6 and the 7-document fixture audit --
replay to the same attribution, the same agreement verdict, and the same budget answer as it. In
CI,
`test_every_reading_replays_identically_through_all_five_forms` runs a path-shaped MIXED corpus --
dated documents and one undated, so the table carries a labelled entry and a bare stem under a live
fold -- and asserts every reading equal across the unfolded schema 5 (`unfolded_documents`), the
labelled schema 4, and the id-keyed schema 3 and schema 1, with schema 6 joining them when
[the count default](#the-count-most-documents-share) landed. Each is rewritten from a fresh record
rather than frozen as a blob.

### The count most documents share

With the id table folded, `documents` fell to 4.0 KiB of the 250-document record's 10.3 and `chunks`
became the largest part at 4.8 -- so the next repetition was one level down, in the COUNTS rather
than in the keys. A count map is one small integer per document, and on a real corpus nearly every
document has the same one: 190 of the 250 documents store exactly one chunk, and 201 of them have
exactly one comparable chunk. That number was being written out 250 times.

So a count map records the value most CORPUS documents share once, under `default`, and lists only
the documents that differ (`interned_counts` / `named_counts` in
`src/llb/conflicts/bundle/document_index.py`). Every count map in the record is offered the fold --
the two under `chunks` and the three exclusion reasons -- under the same per-map gate as the id
fold.

**Three things a default must not swallow**, because in a count map absence already carries meaning:

- a corpus document whose count DIFFERS is listed with its own count, including an explicit `0`.
  That is how a document absent from `comparable` survives a non-zero default: the record says zero
  rather than letting the default speak for it;
- an id the audited corpus never carried is listed whatever its count. The default covers the
  documents the record can enumerate, and an extra (`extra_document_ids`) is not one of them, so a
  store one build ahead of the corpus cannot acquire a count it never had;
- `recovery_floor` declines the fold outright (`absent_is_zero=False`). It is the one map here where
  a missing document is not a zero one -- absence says *no* `--min-claim-tokens` value returns this
  document, a floor of `0` says one does -- and the fold trades those two being the same for bytes.

Reading back, the default is expanded over the corpus documents FIRST so an explicit entry always
wins, and the zeros are then dropped -- because under a fold a zero is the absence the plain map
expresses. That is what makes a folded map read back as exactly the mapping the plain one gives,
rather than as the same answers in a denser dict.

**Measured, on real bundles** (CUDA host, no adjudication call; compact `json.dumps` bytes). The
schema-6 column is the same content with the defaults written back out:

| bundle | documents | `stored` | `comparable` | `chunks` | whole record | `summary.json` |
| --- | --- | --- | --- | --- | --- | --- |
| `20260815T-count-default-squad-cos060` | 250 | 2,390 -> 591 (-75%) | 2,362 -> 485 (-79%) | 4,794 -> 1,118 (-77%) | 10,276 -> 6,600 (-36%) | 40,477 -> 36,801 (-9%) |
| `20260815T-count-default-squad-cos080` | 250 | 2,390 -> 591 (-75%) | 2,362 -> 485 (-79%) | 4,794 -> 1,118 (-77%) | 9,473 -> 5,797 (-39%) | 36,728 -> 33,052 (-10%) |
| `20260815T-count-default-fixture-semantic` | 7 | 48 -> 46 (-4%) | 48 -> 46 (-4%) | 175 -> 171 (-2%) | 1,209 -> 1,179 (-2%) | 13,223 -> 13,193 (-0.2%) |
| `20260815T-count-default-fixture-floor` | 7 | 48 -> 46 (-4%) | 2 -> 2 (declined) | 129 -> 127 (-2%) | 1,136 -> 1,106 (-3%) | 13,152 -> 13,122 (-0.2%) |
| `20260815T-count-default-fixture-hash` | 7 | absent | absent | absent | 637 (0%) | 3,276 (0%) |

**Where the gate declines is as much of the result as where it takes.** On the same 250-document
bundle the four `exclusions` maps are offered the fold and refuse it -- 444 bytes before and 444
after -- because they are SPARSE: their dominant value is zero, so the default would name it and
then list every document anyway, for 13 bytes more per map. `comparable` on the floor run refuses
for the same reason from the other end (nothing is comparable, so the map is empty). The fixture
takes it in three places and gains 30 bytes; the hash-tier bundle records no counts at all and is
untouched. No bundle on this host pays anything for the option.

**Decision: take the default.** It removes three quarters of the record's largest part on the corpus
size where the record actually costs something, costs nothing on the four maps and two bundles where
it buys nothing, and needs no new concept to read -- `default` is one key naming a number, beside
the entries that were already there.

**What was declined**, and measured rather than assumed: folding the three maps into ONE entry per
document (`{"9": [2, 1]}` instead of two maps) reaches 965 bytes against the default's 1,118 on the
same bundle -- 153 bytes better. It is refused for the reason
[the positional document row was](#the-label-a-document-with-nothing-to-label-was-carrying): a pair
read by OFFSET is legible only to a reader who knows the field order, and it makes a fourth count a
schema bump rather than an additive key. Inverting each map to `{count: [position, ...]}` was priced
too and reaches only 2,371 (-51%), less than half of what the default takes.

**Every reading replays identically.** The 24-bundle archive sweep re-taken on this build produces a
`stage.json` and a `stage.md` BYTE-IDENTICAL to the pre-change sweep
against the sweep taken on the schema-6 build before it, and each re-taken bundle replays to the
same attribution, agreement verdict, and budget answer as its predecessor -- including the
count-defaulted floor fixture against its **schema-2** original, which is the bundle whose reading
quotes exclusion counts and a recovery floor BY VALUE, so a default that swallowed either would
show up as a changed sentence. Directly on disk, every re-taken
bundle's `chunks` and `exclusions` unfold byte-for-byte to the schema-6 bundle of the same run. In
CI, `test_every_reading_replays_identically_through_all_five_forms` adds the count-unfolded schema 6
to the four id-table forms, and `test_a_document_the_default_does_not_speak_for_is_written_out`,
`test_a_map_where_no_count_dominates_records_the_entries_it_always_did`, and
`test_a_recovery_floor_declines_the_fold_because_absence_is_not_zero` pin the three things a default
must not swallow.

### Why a document is not comparable, and the floor that returns it

The stage attribution could already say a pair stopped at the CLAIM-TOKEN FLOOR, but the sentence it
printed there was a disjunction -- "front matter, below `--min-claim-tokens`, or a repeated metadata
block" -- because the run kept one total and threw the three reasons away. Two of those three are
not moved by the knob the stage names: front matter is excluded BEFORE the floor is consulted, and a
repeated metadata block is excluded for being ABOVE it. So "lower `--min-claim-tokens`" was advice
that could not work on two thirds of the cases it was printed for.

`ContentSelection` (`semantic_filter.py`) now keeps the exclusion ORDINALS rather than three counts,
and `DocumentExclusions.of` folds them per document in the same pass. The reading gains two things:

- the reason names the counts: `` `archive-policy.md`: 1 front matter, 2 below `--min-claim-tokens` ``;
- the knob names a VALUE: `recovery_floor` is the largest claim-token count among a document's
  low-content chunks, which is the highest floor at which that document has a candidate chunk again.
  A pair needs every filtered side back, so the printed floor is the LOWEST of its sides' floors.

A document with no low-content chunk has no `recovery_floor` entry, and that absence is the reading:
the knob then says so outright rather than naming a value -- "no `--min-claim-tokens` value returns
this pair, because nothing the floor governs is what excluded it".

**Measured, on the committed 7-document fixture** (CUDA host, real e5-base 19-chunk heading store,
no model call). Run at a floor nothing clears, then at the floor the record names, then one above it:

```text
make build-index CORPUS=samples/corpora/conflicts_uk_v1/corpus CHUNK_STRATEGY=heading CHUNK_SIZE=600
make audit-corpus-conflicts CORPUS=samples/corpora/conflicts_uk_v1/corpus EFFORT=semantic \
  STORE=<that-store> COS_THRESHOLD=0.9 MIN_CLAIM_TOKENS=200
[conflicts] ... Earliest stage an orderable document pair was lost at: the CLAIM-TOKEN FLOOR, shown
  by `archive-policy.md` + `deadline-note.md` (every chunk of `archive-policy.md` and
  `deadline-note.md` in the store is excluded from comparison (`archive-policy.md`: 1 front matter,
  2 below `--min-claim-tokens`; `deadline-note.md`: 1 front matter, 2 below `--min-claim-tokens`)).
  One knob: lower `--min-claim-tokens` to 30 or below (this run used 200), or re-chunk so the claim
  lands in a longer chunk.
```

| run | `MIN_CLAIM_TOKENS` | stage named | `deadline-note.md` comparable chunks |
| --- | --- | --- | --- |
| `20260815T-bundle-record-fixture-floor` | 200 | `claim_token_floor` | 0 |
| `20260815T-bundle-record-fixture-floor-31` | 31 | `claim_token_floor` | 0 |
| `20260815T-bundle-record-fixture-floor-30` | 30 | `candidates` | 1 |

The recorded floor is EXACT, not a hint: 30 returns the pair and 31 does not, which is what the
per-document floors say (`archive-policy.md` 51, `deadline-note.md` 30, and the pair takes the
lower). `tests/llb/conflicts/test_bundle_record.py` pins that acceptance the same way -- it re-runs
the audit at the floor the record named and asserts the document became comparable, rather than
comparing the record with itself.

**What the floor is NOT.** It is a per-document necessary condition, never a corpus-wide sweep.
Repeated-metadata detection runs OVER the candidate set the floor produces, so a lower floor can add
chunks that then re-group other chunks out of comparison; an exact "what would floor N have
excluded" needs a token count per chunk and a re-run of the grouping, which is the refused
`floor_sweep` reading above.

### What a smaller candidate budget would have returned

The claim tier's budget is a RANK CUTOFF into the store's own cosine ordering:
`detect_semantic_pairs` returns its pairs sorted by descending similarity, `--max-claim-pairs` keeps
a prefix, and `--max-candidate-pairs` picks the cosine threshold that makes the list about that
long. A smaller budget therefore returns a PREFIX of the same ranking, and a prefix is answerable
with no store, no vectors, and no re-adjudication.

What it needs is the ranking, and the ranking is the one thing `findings.jsonl` loses. Every
candidate pair does reach that file -- pairs past `--max-claim-pairs` land there as provisional
`duplicate` rows -- but the rows are written in report order (actionable first, then by the
ADJUDICATOR's score, which for a claim-tier row is not the cosine that ranked it). The bundle holds
the candidate SET and not the candidate ORDER.

`CandidateRecord` records the order, collapsed to document pairs with the rank each one first
appears at. Collapsed, because every reading built on it is a document-pair reading and the first
rank a pair appears at is exactly what decides whether a budget returns it. Capped, because the
uncollapsed list is quadratic in CHUNKS and the collapsed one is quadratic in DOCUMENTS: the cap is
`--max-candidate-record-pairs` when the run set one, the run's own `--max-candidate-pairs` when it
set that instead, and `DEFAULT_CANDIDATE_RECORD_PAIRS` (200) otherwise. `covered_to_rank` says how
deep the record reaches and `cap` says what stopped it there. A budget past the prefix is REFUSED,
not answered from a truncated list, and the refusal names the knob:

```text
[stage] <run> at budget 223: not recomputable -- budget 223 is past rank 222, the deepest the
  capped candidate record reaches (capped at 200 document pairs -- re-run with a larger
  `--max-candidate-record-pairs` to record more)
```

```bash
make recompute-conflict-stage STAGE_RUNS="<audit-run-dir> ..." STAGE_BUDGET=2
# -> $DATA_DIR/corpus-conflict-stage/<run>/{stage.md,stage.json}
```

The reading reports THREE facts, because the attribution alone under-reports the budget: the named
lost pair and stage; how many distinct DOCUMENT pairs return; and how many of the run's ORDERABLE
returned pairs belong to those document pairs. The last count deliberately preserves
`findings.jsonl` row multiplicity. Governance coverage counts returned passage-pair rows, while the
candidate record collapses them to document pairs; treating both as sets can otherwise put a
deduplicated numerator beside the run's row-count denominator. The recorded prefix decides which
document pairs remain, hash and lexical document pairs remain budget-independent, and
`compare_editions` over the bundle's recorded document governance counts the run's evidence
attached to the retained pairs. It does not claim that every original chunk-pair row survives a
smaller rank cutoff; the bundle deliberately records first appearance at document-pair granularity.

`budget_entry` in `src/llb/conflicts/bundle/stage_replay.py` records both counts and the run's own
denominators. `pairs_phrase` in `src/llb/conflicts/report/stage_replay.py` states the orderable-pair
cost in the CLI line and report table. The focused fixtures in
`tests/llb/conflicts/bundle/test_bundle_record.py` pin row multiplicity, equality with recorded
`orderable_pairs` at the run's own budget, and a cheaper budget that drops a document pair but
costs zero orderable returned pairs.

**Measured on 2026-09-01 on the RTX 4060 Ti 16 GB CUDA host, with no model, store, corpus, or CUDA
call.** One `make recompute-conflict-stage` replay at `STAGE_BUDGET=2` read four preserved audits:
the committed 7-document, 19-chunk dated fixture at semantic and hash effort, and two semantic
audits of the 250-document, 311-chunk SQuAD quickstart corpus at cosine thresholds 0.60 and 0.80.

| audit input | document pairs, run -> budget 2 | orderable run rows on retained pairs | attribution moves |
| --- | --- | --- | --- |
| dated fixture, semantic | 8 -> 5 | 16 -> 9 (cost 7) | no |
| SQuAD, semantic, cosine 0.60 | 36 -> 2 | 0 -> 0 (cost 0) | no |
| SQuAD, semantic, cosine 0.80 | 1 -> 1 | 0 -> 0 (cost 0) | no |
| dated fixture, hash effort | refused | refused: no ranked candidate list | -- |

Separate own-budget replays at the semantic runs' recorded candidate totals -- ranks 14, 38, and 1
in table order -- return 16 of 16, 0 of 0, and 0 of 0 orderable pairs. Each therefore equals its
bundle's recorded `orderable_pairs`, which is the compatibility gate. The dated fixture shows the
distinction the new count exists to make: three of eight document pairs disappear at budget 2, but
the run attaches seven of its sixteen orderable returned rows to those excluded pairs, so the
cheaper budget costs governance evidence rather than only noise. Both undated SQuAD audits exclude
no orderable evidence even where the total falls from 36 document pairs to 2. The named attribution
moves on none of the four inputs: the fixture's corpus-first lost orderable pair,
`archive-policy.md` +
`deadline-note.md`, is absent at every cutoff, so its sentence cannot reveal either cost. The prior
2026-08-15 no-model archive sweep on the same host found the same fixed-attribution result across
all 24 bundles; the replay above narrows to the four budget-answer shapes and adds their evidence
cost.

This reading would be overturned if the candidate ranking ceased to be prefix-stable, or if a
replay at the run's own budget differed from its recorded `orderable_pairs`; both conditions are
pinned in CI. It does not re-adjudicate a row or claim that an undated SQuAD pair is useful conflict
evidence -- zero here means only that `compare_editions` cannot order it.

### How deep the prefix reaches, and what the depth costs

The cap used to be a round number nobody had priced -- 200 document pairs, pinned only by a fixture,
because no corpus on this host formed a candidate list long enough to reach it. It is now measured,
and it is a run parameter (`--max-candidate-record-pairs` / `MAX_CANDIDATE_RECORD_PAIRS`) recorded
beside the prefix it produced.

**Where the cap starts to bite.** On the 250-document quickstart corpus (CUDA host, the committed
e5-base 311-chunk store, `--effort semantic`, no model call) the crossing is between cos 0.51 and
cos 0.50: at 0.51 the ranking is 188 chunk pairs collapsing to 170 document pairs and the cap never
bites; at 0.50 it is 224 collapsing to 202, and the record keeps 200 of them, reaching rank 222 of
224. That is with `DEFAULT_COSINE_THRESHOLD` at **0.9** and the runs behind this page at 0.6 and
0.8, where the same corpus forms 38 and 1 candidate rows. So on this host the constant is
unreachable at any threshold an operator would audit at, and the sweep below has to manufacture the
density to price it. The other way a run gets a deep list -- `--max-candidate-pairs`, which resolves
its own cosine (0.36 on the goods corpus) -- caps the record at that budget and never reaches the
constant either.

**The depth/cost curve** (same corpus and store at cos 0.25, where the ranking is 3,127 chunk pairs
collapsing to 2,560 document pairs; one `make audit-corpus-conflicts` run per cap, 2026-08-15,
RTX 4060 Ti 16 GB CUDA host). Those runs predate
[the id table](#the-id-table-every-document-named-once), so both columns are given -- the bytes as
those bundles hold them, and the bytes the same content occupies re-encoded at schema 4:

| cap | pairs recorded | answers budgets to rank | `candidates` bytes | whole record | `summary.json` |
| --- | --- | --- | --- | --- | --- |
| 25 | 25 | 26 | 1,713 -> 643 | 26,854 -> 15,461 | 81,489 -> 51,334 |
| 50 | 50 | 53 | 3,362 -> 1,220 | 28,503 -> 16,038 | 84,589 -> 51,912 |
| 100 | 100 | 107 | 6,665 -> 2,380 | 31,806 -> 17,198 | 90,792 -> 53,072 |
| 200 (default) | 200 | 222 | 13,353 -> 4,783 | 38,494 -> 19,601 | 103,280 -> 55,475 |
| 400 | 400 | 458 | 26,736 -> 9,596 | 51,877 -> 24,414 | 128,264 -> 60,289 |
| 800 | 800 | 920 | 53,488 -> 19,182 | 78,629 -> 34,000 | 178,215 -> 69,874 |
| 2,600 (whole list) | 2,560 | 3,127 | 172,918 -> 63,199 | 198,059 -> 78,017 | 399,727 -> 113,893 |

**The cost side cannot pick the value.** The curve is a straight line with no knee anywhere on it --
**66.7 to 68.5 bytes per recorded document pair** before the id table, **23.8 to 25.7 bytes** after
it -- so every cap is affordable and every cap is worse than the one below it by the same amount per
pair. Interning moved the whole line down without bending it, which is the point: it removes a
constant per row, not a growth term. What the curve does establish is the price of the extremes:
recording the whole list still turns a 15 KiB record into a 78 KiB one and takes more than half of
`summary.json`, which is the outcome the cap exists to prevent.

**What picks it is the depth a re-read can ask at**, and two facts bound that:

- a cap of N document pairs answers every budget up to at least rank N -- each recorded pair
  consumes one rank or more, so `covered_to_rank >= N` by construction, and the cap is therefore
  readable in the budget's own units;
- the question is downward. `--budget` asks what a SMALLER candidate budget would have returned, so
  the run's own budget is the ceiling on it, not a starting point. The deepest budget any measured
  claim-tier run on this host used is 100 (`MAX_CANDIDATE_PAIRS=100` on the goods corpus) and
  `SUGGESTED_MAX_CANDIDATE_PAIRS` is 50.

**Decision: the constant stays 200**, now for a stated reason rather than by default -- it is 2x the
deepest measured operating budget and 4x the suggested one, it guarantees an answer to rank 200 or
better, and it costs 4.8 KiB. The id table cut that price from 13.3 KiB without touching the depth
argument, which is what the decision actually rests on, so the value does not move. The constant is
no longer the only lever either: a corpus that is genuinely re-read deeper sets
`--max-candidate-record-pairs` and pays ~24 bytes per pair, and the refusal past the prefix names
that flag instead of leaving an operator to find it. On the cos 0.50 run above, raising the cap from
200 to 250 recovers the last 2 document pairs and the last 2 ranks for **48 bytes**.

**The cap is recorded because a truncated prefix and a short ranking look identical.** `cap` sits
beside `covered_to_rank`, so a reader can tell "the corpus ranked no more" from "this run declined
to write more down" -- only the second has a knob. A schema-2 bundle carries no `cap` and says so
(`cap not recorded: this bundle predates it`) rather than reporting today's constant as if that run
had used it.

## The size the record actually costs

Bytes at schema 7, with the same content at each form the record has had before it: the counts
written per document (schema 6), the ids unfolded too (5), the labels back (4), and the pre-interning
form keyed by id:

| bundle | documents | store chunks | record bytes | schema 6 | schema 5 | schema 4 | keyed by id | of which `candidates` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20260815T-count-default-fixture-hash` | 7 | -- (no store) | 637 | 637 | 637 | 637 | 637 | absent |
| `20260815T-count-default-fixture-semantic` | 7 | 19 | 1,179 | 1,209 | 1,209 | 1,209 | 2,056 | 186 |
| `20260815T-count-default-squad-cos080` | 250 | 311 | 5,797 | 9,473 | 11,911 | 14,911 | 25,278 | 93 |
| `20260815T-count-default-squad-cos060` | 250 | 311 | 6,600 | 10,276 | 12,714 | 15,714 | 27,578 | 896 |

The largest bundle on this host now records 6.6 KiB over 250 documents -- about 26 bytes per
document -- against a 13.1 KiB `summary.json`, from 27.6 KiB before any of the four folds. What it
costs is back to the id table and nothing else: `documents` is 4.0 KiB of the 6.6, against
`candidates` 0.9 KiB, `chunks` 1.1 KiB, and `exclusions` 0.4 KiB. That is the floor the size bound
predicts, reached from both ends -- one stem per document, one shared affix, one default per count
map, and a handful of integers for the documents that differ from it.

The record is now the LARGEST thing in `summary.json` (6,600 of 13,409 bytes) rather than a share of
it. It was not, until the four folds finished: at that point the file's biggest key was the `tree`
block, and almost all of that was a second copy of the store's own manifest, which is what
[the store the bundle does not copy](#the-store-the-bundle-does-not-copy) removed.

At the thresholds those runs used, no corpus here has a candidate list long enough for the cap to
bite -- it takes a deliberately loosened cosine to reach it, which is what
[the depth/cost curve](#how-deep-the-prefix-reaches-and-what-the-depth-costs) does. The cap, the
refusal, and the knob the refusal names are therefore pinned in CI over fixtures
(`test_the_candidate_record_stops_at_its_cap_and_says_how_far_it_reaches`,
`test_the_cap_is_a_run_parameter_recorded_beside_the_prefix_it_produced`), alongside a growth test
that doubles a corpus past the cap and asserts the record less than doubles, and
`test_the_interned_record_is_smaller_than_the_form_it_replaces`, which keeps the saving from ever
going negative on a corpus of very short ids. The document table's own form is pinned the same way:
`test_a_document_with_nothing_to_order_on_is_recorded_as_the_id_alone` and
`test_dropping_the_empty_label_is_a_saving_on_undated_documents_and_free_on_dated_ones`, the second
of which asserts BOTH halves of the decision -- smaller on an undated corpus, and byte-for-byte
unchanged on a dated one. The id fold carries both halves too:
`test_the_head_and_tail_a_path_shaped_corpus_shares_are_recorded_once`,
`test_a_corpus_the_fold_cannot_pay_on_records_the_table_it_always_did` (parameterized over the
scattered corpus and the too-small one),
`test_every_folded_id_round_trips_to_the_exact_string_it_was_written_from`,
`test_the_fold_never_takes_more_of_an_id_than_the_id_has`, and
`test_an_id_the_corpus_never_carried_is_recorded_whole_rather_than_folded`.

### The decision-range block records the reading, not every row behind it

Once the store manifest left `tree`, `group_granularity` was the next avoidable cost in
`summary.json`. Its schema-1 form repeated the unit prose and two rule descriptions in every run,
then recorded BOTH the expanded group-size list and the histogram the report rendered. It also
carried `quoted_group_split` for every transitive group although the report names only the three
longest chains. None of those extra copies changed the decision range, partition/cover verdict, or
operator prose.

**Decision: schema 2 records rule names as the keys under `rules`, not as data plus prose.** It drops
`unit`, each `description`, and the repeated inner `rule`; a consumer joins the rule key to
`schema_version`. Each distribution keeps either `sizes` or `size_counts`, whichever is smaller in
compact JSON, through the same `smaller_form` gate as the bundle record's other folds. Repeated
sizes therefore collapse to a histogram while a short irregular list pays no histogram overhead.

The per-group list becomes a bounded report record. Schema 2 keeps the three longest chains as
`quoted_group_chains`, or the complete `quoted_group_split` when that smaller form already has no
more than three entries. An empty chain record is omitted. The full split remains exactly
recomputable from `findings.jsonl` with `make compare-conflict-granularity`; it is not copied into
every summary for a consumer that does not exist. `reported_chains` and
`distribution_size_counts` in `src/llb/conflicts/grouping/granularity.py` read schema 1 and both
schema-2 folds, while `src/llb/conflicts/report/granularity.py` renders only those readers.

**Measured 2026-09-01 on the RTX PRO 3000 Blackwell 12 GiB CUDA host.**
`make compare-conflict-granularity` re-read four existing bundles from `findings.jsonl`; it made no
model, encoder, store, or CUDA call. Bytes use compact `json.dumps`, the record's own fold unit.

| audited corpus and setting | documents | rows | `group_granularity` schema 1 -> 2 | whole `summary.json` with schema 2 |
| --- | --- | --- | --- | --- |
| 250-document SQuAD corpus, cosine 0.060 | 250 | 38 | 2,537 -> 704 (-72.3%) | 13,409 -> 11,576 (-13.7%) |
| committed conflict fixture, semantic tier | 7 | 17 | 1,398 -> 562 (-59.8%) | 12,656 -> 11,820 (-6.6%) |
| 250-document SQuAD corpus, cosine 0.050 | 250 | 224 | 2,882 -> 816 (-71.7%) | 71,268 -> 69,202 (-2.9%) |
| 250-document SQuAD corpus, cosine 0.025 | 250 | 3,127 | 2,777 -> 1,038 (-62.6%) | 233,935 -> 232,196 (-0.7%) |

On the 2,537-byte target block, dropping constant prose and duplicate rule names saves 583 bytes;
choosing one size form saves another 175; bounding the per-group chain record saves the remaining
1,075. More rows therefore do not imply proportionally more summary bytes: the 224-row and
3,127-row SQuAD blocks differ by only 222 bytes at schema 2. The adversarial fixture pins the same
property without relying on that corpus shape: ten times as many disconnected singleton rows (100
to 1,000) move the block from 479 to 490 bytes, not tenfold.

Compatibility was replayed over all 44 conflict bundles readable on this host: the schema-1 block
and a schema-2 recomputation produced identical **How many decisions the row count is** sections in
every case. `tests/llb/conflicts/grouping/test_group_granularity_record.py` pins the schema fields,
the per-form fold, the 100-to-1,000 growth bound, and exact old/new rendering; the existing grouping
suite still pins the rules themselves. Run the focused artifact check or the full acceptance gate
with `make ci`. A future report that needs more than three named chains, or a real consumer that
needs every per-group chain length without reading `findings.jsonl`, would overturn the bounded
record decision and require a newly priced schema. Until then, restoring a row-linear list adds no
reading.

## The store the bundle does not copy

Once the four folds finished, the per-document record was no longer what `summary.json` cost. On the
250-document bundle the `tree` block was **23,897 of the 36,801 bytes** (65%) against the record's
6,600, and 23,500 of that was ONE key: `tree.doc_fingerprints`, the store's whole `{doc_id: sha256}`
manifest copied verbatim out of `store_meta.json`. It repeated in full both things the record had
just learned not to -- every document id, unfolded and un-interned, and a 64-hex digest per document
that nothing compares for anything but equality.

**What the map was read FOR is the question that picks the form, and the answer is: nothing, per
document.** The per-document question -- WHICH documents changed -- has a consumer, and it is not a
bundle:

| who asks | of what | why not the bundle |
| --- | --- | --- |
| `llb refresh-index` / `refresh_store` (`src/llb/rag/refresh/diff.py`) | `store_meta.json`, the authoritative map | the store's own manifest is current; a bundle's copy is a snapshot from run time and can only be a worse answer to the same question |
| the projection/tree reuse gate (`prepare_projected_index`) | `source_fingerprint`, which already hashes the map | it needs one equality test, and hashes the chunk table with it |
| anything reading a finished bundle | -- | nothing read it at all: the copy was write-only in `summary.json` and in the persisted `tree_meta.json` sidecar alike |

What a BUNDLE is asked is one question with a yes-or-no answer: **is the store on disk still the
store this run read?** That is an equality test over the whole map, so it is answered by one digest
over the sorted pairs -- 64 hex characters, independent of the corpus size. The pairs are sorted
before hashing, so the digest is a property of the mapping rather than of the order a store happened
to write it in: a rebuild that visits the corpus in a different order over identical content is the
same store and reads as one.

**Decision: record the identity, not the manifest.** `doc_fingerprints_digest` plus
`doc_fingerprints_documents` replace `doc_fingerprints`
(`identity_payload`, `src/llb/conflicts/bundle/store_identity.py`). The document count rides along because
it is 3 bytes and it is what makes a changed-store sentence readable ("250 documents recorded, 251
on disk now"); the verdict itself never rests on it, which is why an EDITED document under an
unchanged count is still detected below. A store that records no fingerprints at all -- a store
built before them, or a run below the semantic tier -- records no identity rather than the digest of
an empty map: "identical to every other fingerprintless store" would be a claim where the absence is
a silence.

**What the other option costs.** Keying the map on the record's own document table and truncating
each digest to a stated collision bound would preserve a per-document answer for roughly
`len(table) + n * bound` bytes -- still linear in DOCUMENTS, still a second copy of a manifest that
is authoritative elsewhere, and still answering a question no reader of a bundle asks. It is
declined for the reason the positional document row was
([the label section](#the-label-a-document-with-nothing-to-label-was-carrying)): the extra it buys
is real only for a consumer that does not exist, and it charges every bundle a growth term for it.

**Measured, on real bundles** (CUDA host, no adjudication call -- the encoder runs only to build the
stores; compact `json.dumps` bytes, the unit the rest of this page quotes). Each run is the same
audit as its `count-default` predecessor, re-taken on this build:

| bundle | documents | `tree` | `summary.json` | record |
| --- | --- | --- | --- | --- |
| `20260815T-store-identity-squad-cos060` | 250 | 23,897 -> 505 (-97.9%) | 36,801 -> 13,409 (-63.6%) | 6,600 (0%) |
| `20260815T-store-identity-squad-cos080` | 250 | 23,897 -> 505 (-97.9%) | 33,052 -> 9,660 (-70.8%) | 5,797 (0%) |
| `20260815T-store-identity-fixture-semantic` | 7 | 1,037 -> 500 (-51.8%) | 13,193 -> 12,656 (-4.1%) | 1,179 (0%) |
| `20260815T-store-identity-fixture-floor` | 7 | 1,037 -> 500 (-51.8%) | 13,122 -> 12,585 (-4.1%) | 1,106 (0%) |
| `20260815T-store-identity-fixture-hash` | 7 | absent | 3,276 (0%) | 637 (0%) |

The saving is the whole map minus 130 bytes, so it grows with the corpus exactly as the map did:
94 bytes per document at the quickstart corpus's 22-character ids, which is 23.5 KB at 250 documents
and would be 2.35 MB at 25,000. The hash-tier bundle is the control -- it built no tree, has no
`tree` block, and is byte-for-byte unchanged. Every other key of every re-taken bundle is identical
to its predecessor's apart from the wall-clock `seconds` each tier reports.

**The form is self-describing, so no version moves.** `doc_fingerprints_digest` is present exactly
when `doc_fingerprints` is not, and `StoreIdentity.of` reads either -- computing the digest from a
recorded map when that is all a bundle has. So `TREE_VERSION` stays where it is (bumping it would
force every persisted tree on the host to rebuild, for a change that touches no geometry) and
`stage_attribution_inputs.schema_version` stays at 7, because the `tree` block is not part of that
record.

**The bundle also records where to ask the question.** `tree.store_data_dir_relative` is the exact
resolved `StoreView.index_dir` the semantic pass read, relative to the configured `DATA_DIR`. A
generation-backed store therefore records its immutable generation directory, not the live base
whose newest generation can later advance. A store outside `DATA_DIR` records no location and
continues to need an explicit flag; an absolute host path never reaches `summary.json`.

The re-read has three deliberate cases:

1. An explicit `STAGE_STORE` / `--store` wins for every bundle. This preserves the existing
   operator-directed comparison and supplies the location for an older bundle that has none.
2. Without that flag, each current bundle resolves its own recorded location under the current
   host's `DATA_DIR` and reads that exact directory's `store_meta.json`. A sweep can therefore span
   stores and generations without being told which bundle belongs to which one.
3. A recorded directory whose metadata is gone is `not comparable`: the detail names its portable
   `$DATA_DIR/...` reference and says that no identity comparison was made. The reader neither
   searches the host nor compares a convenient different store and calls that a mismatch. Invalid
   absolute and parent-traversal references are refused on the same terms.

The ordinary multi-store path needs no store flag:

```bash
make recompute-conflict-stage STAGE_RUNS="<audit-run-dir> <audit-run-dir>" \
  STAGE_OUT=<report-dir>
# [stage] <run>: the recorded store at `$DATA_DIR/<store-a>` is the one this run read
# [stage] <run>: the recorded store at `$DATA_DIR/<store-b>` is the one this run read

# Deliberate override, and the fallback for bundles without a recorded location:
make recompute-conflict-stage STAGE_RUNS="<audit-run-dir> ..." \
  STAGE_STORE=<index-dir> STAGE_OUT=<report-dir>
```

It reads `store_meta.json` and nothing else -- no chunks, vector index, encoder, corpus, or model.
The command caches each resolved manifest within the sweep, so many bundles over one store pay for
one small JSON read while bundles over different stores keep their own placement.

**Acceptance run on 2026-09-01, RTX 4060 Ti 16 GB CUDA host, no model call.** Two semantic audits
ran at explicit cosine 0.9: the committed seven-document Ukrainian conflict fixture and its
one-sentence edited variant, each over its own 19-chunk FAISS store built with
`intfloat/multilingual-e5-base` at heading/600 chunking. Each audit returned 17 findings over eight
document pairs. Replaying both bundles together with no store flag resolved two of two distinct
bundle-recorded locations, matched both identities, needed zero explicit fallbacks, produced zero
unavailable placements, and preserved both recorded stage readings. Pointing the same sweep
explicitly at the fixture store made the flag win: the fixture bundle matched and the edited bundle
reported an identity mismatch even though both stores contain seven documents. The first reading
shows that a mixed-store archive is self-placing; the second shows that the explicit precedence and
content digest remain live rather than being inferred from the path or document count. Deleting or
moving a recorded store overturns only its availability -- it must become `not comparable`, never
`changed` -- while editing, adding, or removing a fingerprint at the same location must produce a
real mismatch.

**A store that genuinely changed is still detected as changed.** Measured on this host: the fixture
corpus copied to `.data/store-identity-stores/edited-corpus/` with ONE document extended by one
sentence, re-indexed at the same e5-base heading settings into
`.data/store-identity-stores/edited/`, still holds 7 documents and 19 chunks -- and both fixture
bundles are placed against it as `NOT the one this run read: 7 documents recorded, 7 on disk now`.
A count comparison would have missed it -- the count is identical on both sides -- and the digest
does not.

**Every reading replays identically through both forms.** The 24-bundle archive sweep re-taken on
this build produces a `stage.json` and a `stage.md` that are BYTE-IDENTICAL to the pre-change sweep
against the sweep taken on the build before the identity change, and each re-taken bundle replays
to the same attribution, agreement verdict, and budget answer as its count-defaulted predecessor.
The identity verdict itself is the same through either form on real data: pointed at the fixture
store, the schema-7 bundle and its old-form predecessor both read as "the one this run read". In
CI,
`test_a_bundle_at_either_form_returns_the_identical_verdict` pins that equality over both a matching
and a changed store, `test_a_store_that_genuinely_changed_is_detected_as_changed` covers all three
ways a manifest changes (a document edited, added, and removed),
`test_the_digest_is_a_property_of_the_mapping_and_not_of_the_order_it_was_written_in` pins the
sorting, and `test_the_identity_costs_the_same_whatever_the_corpus_size_is` asserts the recorded
size is constant in the corpus where the map it replaces was linear
(`tests/llb/conflicts/semantic_tree/test_store_identity.py`).

## Where it lives

| what | where |
| --- | --- |
| the record a run writes | `src/llb/conflicts/bundle/record.py` (`RunInputs`, `stage_attribution_inputs`, `recorded_inputs`, `readable_record`, `naming_of`) |
| the id table both forms resolve against | `src/llb/conflicts/bundle/document_index.py` (`DocumentInterner`, `DocumentNaming`) |
| the head and tail the table is folded on | `src/llb/conflicts/bundle/document_affix.py` (`IdAffix.over`, `pays_for_itself`, `stem`, `expand`) |
| the rule every fold obeys | `src/llb/conflicts/bundle/fold.py` (`json_bytes`, `smaller_form`) |
| the count default a map is folded on | `src/llb/conflicts/bundle/document_index.py` (`interned_counts`, `named_counts`, `absent_is_zero`) |
| the store identity the `tree` block records | `src/llb/conflicts/bundle/store_identity.py` (`fingerprint_digest`, `identity_payload`, `StoreIdentity.of`, `compare_store`), written by `tree_meta` (`semantic_tree/refresh.py`) |
| the portable store location and placement precedence | `src/llb/conflicts/bundle/store_location.py` (`store_location_payload`, `recorded_store_location`, `resolve_store_placement`) |
| the store manifest it is compared against | `src/llb/conflicts/store_access.py` (`store_doc_fingerprints` for the live explicit store, `store_doc_fingerprints_at` for an exact recorded directory; meta only -- no chunks or vectors) |
| re-reading a bundle with it | `src/llb/conflicts/bundle/stage_replay.py` (`replay_attribution`, `replay_entry`, the budget prefix) |
| per-document exclusions | `src/llb/conflicts/bundle/document_exclusions.py`, folded in `semantic_run.py` from `ContentSelection` |
| the ranked candidate list | `src/llb/conflicts/bundle/candidate_record.py` |
| the per-question verdicts | `src/llb/conflicts/bundle/readings.py` |
| rendering | `src/llb/conflicts/report/stage_replay.py`, `src/llb/cli/prep/conflict_stage.py` |
| tests | `tests/llb/conflicts/bundle/test_bundle_record.py` (what the record answers), `tests/llb/conflicts/bundle/test_bundle_id_table.py` (the shape it answers from), `tests/llb/conflicts/semantic_tree/test_store_identity.py` (identity, location, exact resolution, override, and gone-store refusal), `tests/llb/conflicts/bundle/test_stage_replay.py` (CLI auto-resolution), `tests/llb/conflicts/test_store_access_and_cli.py` (real FAISS-store recording) |
