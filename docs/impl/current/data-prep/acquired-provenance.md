# Acquired-Corpus Provenance Fields

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

A staged corpus records who may READ a document and when THIS project ingested it. Neither answers
where the text came from: `ingestion_time` is a local event, and `source_sha256` covers the staged
file rather than the bytes a publisher served. This page covers the fields that close that gap --
what is read, from where, and what each downstream artifact does with them.

Reading is all that lands here. Nothing yet ACTS on a value: binding a bundle to an acquisition run,
revision semantics, and the redistribution gate are separate open work under the same capability.
The contract these fields are read against is
[the acquired-corpus projection](../../../design/acquired-corpus-projection.md).

## The seven fields and where they come from

An upstream acquisition service renders one `<source>.metadata.json` sidecar per document -- the
same sidecar convention the operator lane already used. `src/llb/prep/corpus/governance_fields.py`
holds the field names in one place, split by which lane authors them:

| Field | What it answers |
| --- | --- |
| `source_uri` | which URI the text came from |
| `capture_time` | when the source was observed, as distinct from `ingestion_time` |
| `capture_id` | resolvable in the producer's capture store |
| `payload_digest` | digest of the CAPTURED bytes, not of the local staged file |
| `licence` | whether the material may leave the host |
| `acquisition_run_id` | which run produced this corpus version |
| `revision_of` | which document this one supersedes |

`OPERATOR_GOVERNANCE_FIELDS` are the six this project authors (`language`, `version`,
`effective_date`, `ingestion_time`, `source_system`, `acl_label`); `ACQUIRED_GOVERNANCE_FIELDS` are
the seven above; `GOVERNANCE_FIELDS` is their concatenation, and it names both the manifest row's
keys and the `CorpusItem` dataclass fields, so the two cannot be enumerated separately and drift.

**Sidecar only.** Front matter still supplies operator fields and is deliberately not widened:
a projected corpus writes a sidecar, and front matter stays the operator-authored lane, so a
projected field name never has to be told apart from prose a document happens to open with.
`_front_matter_metadata` keys on `OPERATOR_GOVERNANCE_FIELDS` rather than on the full set.

## Where the values travel

One row is built per document (`source_governance` for the text lane,
`converted_governance` for the PDF lane) and splatted into the `CorpusItem`, so every consumer
reads the same row:

- **`corpus_manifest.json` item** -- all thirteen fields on every item, including skipped ones.
- **The fingerprinted item row** (`_manifest_item_row` in `src/llb/prep/corpus/fingerprints.py`) --
  twelve of them: `ingestion_time` is excluded because it is a local event that must not move a
  corpus fingerprint. `FINGERPRINTED_GOVERNANCE_FIELDS` is that subset, derived rather than
  re-listed.
- **`ChunkRecord.metadata`** -- via `manifest_governance_by_doc` in
  `src/llb/rag/chunking/corpus.py`, so retrieval hits carry them.
- **`store_meta.json`** -- `governance_fields` publishes the full list a store was built against.
- **The gold-set provenance record** -- `document_rows` in
  `src/llb/prep/ontology/pipeline/bundle_provenance.py` puts the seven acquired fields beside each
  document's `doc_id`, `sha256` and `n_chars` in `provenance.json`.

**Absence is recorded, not omitted.** Every field is present on every row, `None` where the corpus
supplies nothing, so a reader never has to tell a missing field apart from an unasked question.
There is no conditional-omission machinery: there is no released corpus and no published store to
stay fingerprint-compatible with, so the fields enter the row unconditionally.

**The provenance is additive.** No new field alters document text, a `doc_id`, or a character
offset, so no gold-set label can move because one was added. What DOES move is the corpus
fingerprint -- once, by design, when the row widened. Every fingerprint assertion in the suite is
relative and no committed fixture pins a literal, so a store built before the change reports stale
through the existing `stale_store_message` path and is rebuilt with `llb refresh-index` or
`llb build-index`.

**A re-capture without a text change still updates provenance.** `source_sha256` covers the
document, so a rewritten sidecar leaves the item `reused: true` -- but its governance row is
re-read every ingest, so the new `acquisition_run_id` lands in the manifest and in the next store
build. `preserve_ingestion_time` keeps the prior `ingestion_time` only while every other governance
field is unchanged, so an acquisition change also re-stamps the ingest time.

## How to run it

```bash
make ingest-corpus CORPUS_ROOT=<projected-corpus-dir> CORPUS_OUT_DIR=<out-dir>
make build-index CORPUS=<out-dir>
```

Nothing new is passed on the command line: the sidecar is the only input, and a corpus carrying no
sidecar takes exactly the path it did before.

## Tests

`tests/llb/prep/test_corpus_acquired_provenance.py` pins the whole path: the seven fields read into
the manifest item and into chunk metadata; a corpus carrying none of them ingesting to the same
text, the same `doc_id`s and the same offsets with every field recorded absent; front matter
refusing to supply an acquired field while still supplying an operator one; each of the seven
moving the corpus fingerprint (parametrized, so a field silently dropped from the row fails);
`ingestion_time` staying out of it; a rewritten sidecar updating a reused document's row; and the
gold-set provenance record carrying the fields, or recording their absence. The
`GOVERNANCE_FIELDS`-versus-`CorpusItem` agreement the splat depends on is asserted directly.
`tests/llb/prep/ontology/drafting/test_ontology_draft.py` asserts the same absence in the full
draft-flow bundle.

## Evidence

On 2026-09-02 on this CUDA host, all CPU-only:

- `make ingest-corpus` over a two-document projected fixture (`reg-0001.md` at `version` 1 with
  `licence: redistributable`, `reg-0002.md` at `version` 2 with `licence: local-only` and
  `revision_of: reg-0001.md`) ingested 2 of 2 documents with all seven fields present and correct
  on both manifest items, and reported 1 of 1 document pair orderable.
- `make ingest-corpus CORPUS_ROOT=samples/corpus CORPUS_MIN_CHARS=1` staged both committed fixture
  documents byte-identically to their sources (`diff -r` clean) with all seven fields present and
  `None`, and kept the existing 0-of-2 dated, 0-of-1 orderable governance coverage. The reading:
  widening the row changed the fingerprint and nothing else a label depends on.
- A `sentence`/`flat` store built over the projected fixture with the default embedder carried all
  seven fields on its chunk metadata and published the thirteen-name `governance_fields` list in
  its meta. Feeding the pre-change fingerprint of `samples/corpus` to `stale_store_message` against
  the post-change corpus returned the rebuild message, which is the path an existing store takes.

What would overturn this: a producer rendering a field name the projection does not list (it would
be dropped silently, which is what the round-trip fixture under the same capability is for), or a
consumer reading `ChunkRecord.metadata` positionally rather than by key.
