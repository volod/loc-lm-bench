# Acquired-Corpus Provenance Fields

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

The bundle record that carries these fields is the registered `llb.ontology-provenance` contract; a
bundle drafted before the corpus binding existed migrates to it by stating each absence. See
[data-prep contracts](../artifact-contracts/data-prep-contracts.md).

A staged corpus records who may READ a document and when THIS project ingested it. Neither answers
where the text came from: `ingestion_time` is a local event, and `source_sha256` covers the staged
file rather than the bytes a publisher served. This page covers the fields that close that gap --
what is read, from where, and what each downstream artifact does with them.

Ingestion also acts on document identity and `revision_of`: the acquired lane is append-only, so a
revision retains the text and manifest row its predecessor's offsets name. A gold-set bundle binds
its corpus fingerprint to the acquisition runs represented in that version, and every export and
frontier egress path refuses material whose recorded `licence` forbids redistribution -- see
[the redistribution gate](#redistribution-gate). The contract these fields are read against is
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
  document's `doc_id`, `sha256` and `n_chars` in `provenance.json`; its aggregate
  `corpus_version` record binds the corpus fingerprint to the contributing acquisition runs.

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

## Corpus-version binding

`corpus_version_binding` in `src/llb/prep/corpus/fingerprints.py` returns one portable record:

```json
{
  "corpus_fingerprint": "<sha256>",
  "acquisition_run_ids": ["<acquisition-run-id>"]
}
```

`provenance_payload` writes that record as `corpus_version` in every completed ontology-assisted
gold-set bundle's `provenance.json`. The fingerprint is the existing corpus identity, unchanged by
this feature. Acquisition IDs come from the same successful manifest rows the fingerprint reads;
they are deduplicated and sorted so document order cannot move the bundle record. If successful
documents came from two acquisition runs, both IDs remain in the list rather than one being picked.
Per-document provenance remains beside it, so a reader can still resolve each span to its specific
capture.

A local corpus, or a corpus manifest whose successful rows carry no acquisition ID, records
`"acquisition_run_ids": []`. The empty list is the explicit no-acquisition state; an omitted key,
`null`, or an empty string is never emitted. The binding is computed without contacting the
producer and does not resolve or rerun an acquisition.

## Append-only document revisions

The projection reserves `source_system: local` for operator directories. A non-empty, non-local
value selects the acquired lane. In that lane, changing the bytes at an already manifested source
path raises a `ValueError` naming the document and instructing the producer to emit a new document
identity with `revision_of`; neither the staged bytes nor the manifest is rewritten. The local-file
lane keeps its existing changed-source replacement behavior.

`src/llb/prep/corpus/revisions.py` follows each current acquired item's `revision_of` through the
previous manifest. A staged ancestor is carried into the next manifest with `reused: true`, and its
file is protected from stale-output cleanup. The walk is transitive, so revision three retains
revision two and revision one even when only revision three remains in the input projection. An
ancestor named in the prior manifest but missing from the staged corpus fails with that document
named; a reference older than this corpus, for which no prior row exists, remains valid provenance
and does not invent a document.

Retention also applies under `--refresh`. Refresh still disables ordinary source reuse and keeps
the local lane's existing behavior, while a separate read of the prior manifest supplies only the
acquired immutability check and revision ancestry. Because retained rows remain ordinary `ok`
manifest items, `corpus_fingerprint` and `corpus_doc_fingerprints` include both versions without a
second identity mechanism. Gold spans continue to resolve as `(doc_id, char_start, char_end)`
against the preserved predecessor text; choosing which version a new gold item should cite remains
a review decision.

## Redistribution gate

`acl_label` says who may READ a document inside a run; `licence` says whether its text may be copied
off the host. `src/llb/prep/corpus/redistribution.py` is the one check that enforces the second, and
every path where corpus-derived text leaves calls it. It makes no determination of its own: the
producing service decides, and this side enforces the recorded value.

| Recorded `licence` | Egress |
| --- | --- |
| absent (`None`) | not gated -- an operator's own corpus, or one carrying no provenance |
| `redistributable` | permitted |
| `local-only` | refused |
| any other value | refused -- a class this side cannot read is not a "yes" |

The comparison is exact (`Redistributable` and the empty string refuse). The vocabulary is
`REDISTRIBUTION_CLASSES`; a producer that needs a third permitting class widens it here first.

**A refusal, not a filter.** `refuse_non_redistributable` raises `RedistributionRefused` (a
`ValueError`) naming EVERY offending document in sorted order with its licence and the egress path,
before the path writes a file, records consent, or makes a call. Dropping the offending rows instead
would leave a partial export that reads as complete. At the CLI boundary `run_typer` prints that one
`[error] [redistribution] ...` line and exits 2 rather than reporting a crash.

**Where the recorded terms are read from.** Three readers feed the check, and `merge_licences` keeps
the strictest value when two sources disagree, so a store and a corpus that differ cannot turn a
`local-only` passage into a permitted one:

- `corpus_licences` -- the staged `corpus_manifest.json` (`ok` items);
- `chunk_licences` -- the `licence` each chunk carried into the store's chunk metadata;
- `bundle_licences` -- a drafted bundle's `provenance.json` document rows, because the bundle's
  corpus copy carries no manifest.

**The paths it guards.** Each checks only the documents it would actually carry, so a corpus that
holds a `local-only` document can still feed a path that never touches it:

| Path | Wiring | Documents checked | Terms read from |
| --- | --- | --- | --- |
| `export-finetune-set` (and the self-improvement loop and campaign that call it) | `src/llb/finetune/export_gate.py` | each selected item's source and gold-span documents, and every document its persisted retrieved spans stand for, collapsed duplicate occurrences included | the corpus `--corpus` names, else the run manifest's `config.corpus_root` |
| Contrastive-pair export (`finetune-embedder`) | `exported_chunks` in `src/llb/finetune/embedder/pairs.py` | every positive and every kept negative chunk | chunk metadata |
| Ontology drafting with a frontier endpoint on either stage | `src/llb/prep/ontology/endpoints/egress.py`, from `draft_goldset` and from the CLI before the consent prompt | every inventoried document after `--doc-limit` | corpus manifest |
| `run-eval` with `scorer_policy=frontier`, including worksheet ratings | `src/llb/executor/runner_egress.py` | retrieved-chunk and gold-item documents of the judged batch | chunk metadata merged with the `corpus_root` manifest |
| Auto-RAG verification with the frontier policy | `src/llb/auto_rag/verification_egress.py` | drafted items' source and span documents | bundle `provenance.json` |
| `frontier-judge` agreement | `_refuse_unredistributable` in `src/llb/scoring/frontier_agreement/run.py` | gold-span documents of the judged rows | corpus manifest merged with the bundle provenance beside the gold set |
| `compare-embeddings --api-model` | `api_lane_enabled` in `src/llb/rag/embedding_bakeoff/run.py`, after the `open`-classification check and before consent | the whole corpus (every chunk is embedded) | corpus manifest |
| `prepare-goldset` (frontier drafting) | `prepare_goldset` in `src/llb/prep/frontier/client.py` | every corpus document | corpus manifest |
| `cross-check-goldset` | `cross_check_corpus_goldset` in `src/llb/prep/goldset/cross_check.py` | items' documents, a bare stem resolved to the staged path | corpus manifest |

Consent and the gate are separate questions. Every frontier path is refused BEFORE it records or
asks for consent, so no consent entitles material whose terms refuse it.

**The fine-tune export's corpus.** That export reads a run bundle rather than a corpus. A run that
names no corpus has no recorded terms and exports as before. A run that NAMES a corpus this host no
longer holds is refused with "is not on this host" rather than exported unchecked, because a
missing corpus cannot be told apart from a permitted one; `--corpus` (`LICENCE_CORPUS=` on the make
target) checks a relocated copy instead.

**Where the gate does not reach, and why:**

- `subset_dataset` re-slices a dataset that already passed the gate, so it carries no new material.
- The external-draft import records an upload the operator made OUTSIDE this project; its own
  `data_classification: open` refusal stays the control for that lane.
- `prepare-synthetic-corpus` reads no corpus, and the local graph-extraction and conflict-claim
  lanes send nothing off the host.
- The `run-eval` frontier lane learns its retrieved documents only by retrieving, so it refuses at
  the judging stage, after local generation and before any frontier call.
- A `frontier-judge` run without a gold set grounds on the reference answer alone and names no
  document; with a gold set but neither a staged manifest nor a bundle provenance record, there are
  no recorded terms to read.

## How to run it

```bash
make ingest-corpus CORPUS_ROOT=<projected-corpus-dir> CORPUS_OUT_DIR=<out-dir>
make build-index CORPUS=<out-dir>
```

Nothing new is passed on the command line: the sidecar is the only input, and a corpus carrying no
sidecar takes exactly the path it did before.

The gate needs no flag either: every guarded path reads the recorded terms itself. The fine-tune
export alone can be pointed at a relocated corpus:

```bash
make export-finetune-set RUN_DIR=<tuning-run> GOLDSET=<goldset> OUT_DIR=<dataset-dir> \
  LICENCE_CORPUS=<corpus-dir>
```

## Projection conformance fixture

`samples/corpora/acquired_projection_v1/` is the committed consuming-side fixture for the
[projection contract](../../../design/acquired-corpus-projection.md). It contains twenty synthetic
documents and one complete sidecar per document. The plant includes one revision pair
(`fixture-doc-02.md` revises `fixture-doc-01.md`) and one document whose `licence` is `local-only`
(`fixture-doc-07.md`), which the revision and redistribution tests use rather than inventing
another corpus.
Source URIs, capture ids, and payload digests are unique per document.

`tests/llb/prep/test_acquired_projection_roundtrip.py` enforces both halves of the consuming
contract inside `make ci`:

- fixture shape -- exactly twenty text/sidecar pairs, every sidecar supplying all twelve projected
  fields (the five upstream-supplied operator fields plus the seven acquired fields), with no
  orphan sidecar;
- ingestion -- all twenty documents report `ok`, their text lands byte-for-byte, and every supplied
  value agrees with the corresponding manifest field;
- negative control -- each of the twelve field names is renamed in turn, and the check must fail
  with the missing original field named. This first checks the sidecar keys because ordinary
  ingestion deliberately records an absent governance value as `None` and ignores additive unknown
  keys; comparing only the resulting manifest would therefore let a rename pass silently.

The fixture is repo-authored and the check has no network, model, or GPU dependency. It validates
the projection shape the consumer accepts, not how an acquisition service derived a URI, digest,
licence determination, or document identity.

## Tests

`tests/llb/prep/test_corpus_acquired_provenance.py` pins the field path: the seven fields read into
the manifest item and into chunk metadata; a corpus carrying none of them ingesting to the same
text, the same `doc_id`s and the same offsets with every field recorded absent; front matter
refusing to supply an acquired field while still supplying an operator one; each of the seven
moving the corpus fingerprint (parametrized, so a field silently dropped from the row fails);
`ingestion_time` staying out of it; a rewritten sidecar updating a reused document's row; and the
gold-set provenance record carrying the fields, or recording their absence. It also checks that a
local corpus binds to an empty acquisition-run list and three successful documents spanning two
runs retain both IDs once each, while a skipped fourth document's run does not enter the version.
The `GOVERNANCE_FIELDS`-versus-`CorpusItem` agreement the splat depends on is asserted directly.
`tests/llb/prep/ontology/drafting/test_ontology_draft.py` asserts the same absence in the full
draft-flow bundle and runs the full fake-endpoint flow over a two-run acquired corpus to assert the
written `corpus_version` record. The committed projection's positive and drifted-name checks live in
`tests/llb/prep/test_acquired_projection_roundtrip.py` as described above. That file also runs the
fixture revision pair as a two-ingest lifecycle, checks the predecessor span after replacement,
checks both per-document fingerprints and manifest rows, and proves that an in-place acquired edit
fails with the document named. The existing mixed-ingest changed-text test remains the local-lane
control.

`tests/llb/prep/test_redistribution_gate.py` pins the shared check: only an absent licence and the
exact `redistributable` class permit (`local-only`, an unrecognised class, the empty string, and a
case variant refuse); a refusal names every offending document in sorted order; the check is scoped
to the documents a path carries; a merge keeps the strictest terms; each of the three readers; the
staged projection records exactly one refused document while a plain corpus records none; and the
CLI boundary turns a refusal into one `[error]` line with exit 2 and no traceback.
`tests/llb/prep/test_redistribution_egress_paths.py` drives each guarded path over the projection
staged through real ingestion (`tests/llb/prep/redistribution_fixtures.py`) with material that
includes `fixture-doc-07.md`. Each path must refuse naming exactly that document, with no output
file, no consent record, and an injected provider that fails the test if called; its control
stages the same fixture with that one sidecar set to `redistributable` and must complete with the
document's text in the payload. The fine-tune export also has tests for the explicit `--corpus`
override, the refusal when a named corpus is gone, and an unchanged export for a run that names no
corpus; ontology drafting has tests for a local plan passing and `--doc-limit` excluding the
document; the draft CLI refuses before its consent prompt.

## Evidence

Field path, revisions, and bundle binding, on 2026-09-02 on an RTX 4060 Ti 16 GB CUDA host, all
CPU-only and model-free:

- `make ci` completed with 4,798 tests passed and 50 deselected. Within it, the
  acquired-projection conformance check ingested all 20 committed documents with status `ok` and
  matched all 12 supplied sidecar fields on every manifest item. Its 12 of 12 deliberate field
  renames failed with the missing original field named. The reading: the committed seam now detects
  a producer/consumer field-name drift instead of accepting the renamed value as an ignored key.
- The same gate exercised the committed revision pair in sequence: the first ingest staged the 19
  documents other than `fixture-doc-02.md`; the input then removed `fixture-doc-01.md` and supplied
  its revision. The second ingest reported 20 `ok` documents, no removed source, both manifest and
  per-document fingerprint rows, and the original `three years` span still resolving in the
  predecessor while the revision said `five years`. An in-place edit of the acquired predecessor
  was refused with its name, while the local changed-text control continued to replace its staged
  file. The reading: a producer following `revision_of` preserves existing labels, and one reusing
  an acquired identity cannot silently move them.
- `make ingest-corpus CORPUS_ROOT=samples/corpus CORPUS_MIN_CHARS=1` staged both committed fixture
  documents byte-identically to their sources (`diff -r` clean) with all seven fields present and
  `None`, and kept the existing 0-of-2 dated, 0-of-1 orderable governance coverage. The reading:
  widening the row changed the fingerprint and nothing else a label depends on.
- A `sentence`/`flat` store built over the projected fixture with the default embedder carried all
  seven fields on its chunk metadata and published the thirteen-name `governance_fields` list in
  its meta. Feeding the pre-change fingerprint of `samples/corpus` to `stale_store_message` against
  the post-change corpus returned the rebuild message, which is the path an existing store takes.
- The same `make ci` run built completed gold-set bundle records through injected local endpoint
  responses. The local-corpus record carried its fingerprint and an empty acquisition-run list; a
  two-document acquired corpus whose rows named two distinct runs carried the identical computed
  fingerprint and both run IDs in sorted order. A separate three-document binding check collapsed
  a repeated run ID while retaining the other one and excluded a run carried only by a skipped
  source. The reading: bundle provenance distinguishes local absence from an unanswered field,
  preserves every producing run for the fingerprinted mixed corpus, and does not claim a producer
  whose document is outside that version, all without a model, network, or producer-store
  dependency.

What would overturn this: a real producer no longer rendering the committed fixture shape, a new
required projected field being added without widening the shared field tuples and fixture, a
revision run dropping either manifested ancestor, a bundle omitting or choosing only one of several
manifested acquisition runs, or a consumer reading `ChunkRecord.metadata` positionally rather than
by key. The fixture deliberately does not establish that an upstream producer derived any projected
value correctly, and the finite two-version sequence does not prove producer-side immutability.

Redistribution gate, on 2026-11-19 on an RTX PRO 3000 Blackwell laptop host, all CPU-only,
model-free, and network-free:

- The 36 gate tests passed. Every guarded path (the two exports, ontology drafting through a
  frontier endpoint, the `run-eval` frontier scorer, auto-RAG frontier verification, `frontier-judge`
  agreement, the `compare-embeddings` API row, `prepare-goldset`, `cross-check-goldset`, and the
  draft CLI's pre-consent check) refused the staged projection naming exactly `fixture-doc-07.md`,
  wrote no output, and never reached its injected provider; each control over the same fixture with
  that sidecar set to `redistributable` completed with the document's text in the payload. The
  reading: each refusal comes from the recorded licence alone and stands between real material and
  a real export, not in front of a path that would have carried nothing.
- Removing the gate makes the tests fail. Each of the ten call sites was disabled in turn: every
  mutation failed that path's own refusal test and no other path's (the fine-tune site also failed
  its relocated-corpus and missing-corpus tests), and making the shared predicate permit everything
  failed 21 of the 35 tests then in the two files. The reading: no path is guarded by accident of
  another's wiring.
- `make export-finetune-set` over a run bundle whose `corpus_root` named the staged projection
  exited 2 with the single `[error] [redistribution] fine-tune dataset export refused` line naming
  `fixture-doc-07.md (licence=local-only)`, and no dataset directory existed afterwards. The
  identical bundle over the redistributable control exported `sft=2 dpo=0`.
- `make ci-github` (the base-environment gate) passed 4,887 tests with 6 skipped and 161
  deselected, and `make ci-checks` reported zero lint, type, complexity, spec-plan, and contract
  findings. The 229 existing tests of the touched lanes (fine-tune, auto-RAG, frontier agreement,
  embedding bake-off, cross-check, frontier drafting, scorer policy, and the projection round-trip)
  passed unchanged, which is the "a corpus with no `licence` field exports as it does today" reading:
  none of their fixtures carries the field, and none needed editing. Full `make ci` on this host's
  base install reported 4,898 passed with 49 failed and 4 errors; the same 53 tests fail
  identically with this change stashed, all on optional extras this venv lacks (`faiss`,
  `langgraph`, `rdflib`, `splink`).

What would overturn the gate: a new path that sends corpus text off the host without calling the
shared check; a producer rendering a permitting class other than `redistributable` (it refuses
until the vocabulary is widened); a store whose chunks lost their `licence` metadata over a corpus
whose manifest is also gone; or a run-eval frontier refusal that arrives after a call because the
judge stage stopped being the first frontier contact.
