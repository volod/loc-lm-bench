# Mixed-Corpus Ingestion And Review Slices

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

The corpus and PDF manifests and the citation sidecars are registered artifact contracts, and
`make check-bundle BUNDLE=<corpus-dir> BUNDLE_KIND=corpus` validates a staged corpus member by
member. See [data-prep contracts](../artifact-contracts/data-prep-contracts.md).

## Mixed txt/md/pdf ingestion

`make ingest-corpus` / `llb ingest-corpus` turns ONE mixed `txt`/`md`/`pdf` directory into the
canonical corpus in a single command (`src/llb/prep/corpus/ingest.py`). PDFs route through the
`ingest_pdf_corpus` converter above (same `pdf-<digest>.md` ids and citation sidecars); `.md`/`.txt`
files pass through verbatim under their relative path so offsets stay exact. Both lanes share the
PDF manifest contract: a per-source `source_sha256`, incremental reuse when the source is unchanged
(`reused: true`), and skip diagnostics for short/failed documents. A unified `corpus_manifest.json`
records every source with its `kind` (`pdf`|`text`), status, and reuse flag, so a rerun over an
unchanged mixed corpus reports `reused: true` for every document. The staged corpus walk excludes
the output subtree, so the default `<root>/_md` output is never re-ingested as new input.

Governance metadata is part of the same manifest contract. Every manifest item records
`language`, `ingestion_time`, `source_system`, optional `version`, optional `effective_date`, and
optional `acl_label`, plus the seven acquisition fields an upstream service renders into the
projection sidecar ([acquired-corpus provenance](acquired-provenance.md)). Text sources can provide
per-document values in `<source>.metadata.json` or markdown front matter; otherwise
`--default-language` is used, then a cheap deterministic detector.
`--source-system` and `--acl-label` set defaults for sources that do not provide their own values.
PDF rows inherit any conversion-manifest governance fields when present and otherwise use the same
operator defaults. Re-ingesting an unchanged source keeps the previous `ingestion_time` when its
non-time governance fields are unchanged.

### Governance coverage at ingestion

Ingestion reports whether those fields can support a later dated supersession, before an operator
builds a store or runs the conflict audit. `src/llb/prep/corpus/governance_report.py` reuses the
audit's `document_coverage` and `document_pair_orderability` functions, so both stages count the
same admitted documents with the same `compare_editions` ordering rule. The command's second
summary line reports:

- documents carrying either ordering field, plus separate `effective_date` and `version` counts;
- corpus document pairs with distinct comparable editions; and
- the consequence when that pair count is zero: no supersession can ever be derived on this
  corpus, including when every document carries one shared edition.

The same values are persisted under `governance_coverage` in `corpus_manifest.json` (schema version
1), together with the consequence sentence. This remains a report: zero coverage does not fail the
command, fill a date from document text or file time, or exclude an otherwise valid document. A
positive pair count is only a precondition; the conflict audit must still find a contradiction
before it can derive `superseded_by`.

`tests/llb/prep/test_corpus_ingest_governance_report.py` pins the undated, per-field dated, and
single-shared-edition cases, manifest round-tripping, successful zero-coverage ingestion, and exact
agreement with the audit-side coverage over the same staged corpus. On 2026-08-31, the CPU-only
`make ingest-corpus CORPUS_ROOT=samples/corpus CORPUS_OUT_DIR=<out-dir> CORPUS_MIN_CHARS=1` path on
this CUDA host ingested both committed fixture documents and reported 0 of 2 documents carrying an
ordering field and 0 of 1 orderable document pairs in both the CLI and manifest, with the
no-supersession consequence. This checks the real Make/CLI/persistence path, not external-corpus
prevalence; adding governance metadata to that fixture would intentionally overturn those fixture
counts. Acquisition provenance is reported nowhere here: `source_uri`, `capture_time`,
`capture_id`, `payload_digest`, `licence`, `acquisition_run_id` and `revision_of` are read into the
same manifest item and carried onward, but this coverage report counts only the two ordering fields
supersession needs. What reads the other seven is in
[acquired-corpus provenance](acquired-provenance.md).

### Refresh and downstream workflows

Deletion propagation is explicit for the local lane: a source removed from the input root is
removed from the next `corpus_manifest.json`, its staged output file is deleted from the canonical
corpus, and the manifest records `removed_sources` plus `n_removed_sources`. Changed PDF ids also
clean up stale old staged outputs. An acquired document is different: non-local `source_system`
marks it append-only, an in-place text change is refused, and a new document whose `revision_of`
names a previously staged version retains that version's file and manifest row instead of reporting
it removed. `src/llb/prep/corpus/revisions.py` follows the full known ancestry, including during a
forced refresh; references that predate the local manifest remain provenance-only. Both old and new
rows therefore enter the normal corpus and per-document fingerprints, so stores see an added
revision rather than a modified document whose offsets moved. Details and the fixture result are in
[acquired-corpus provenance](acquired-provenance.md#append-only-document-revisions). The rollback
unit is the immutable store directory built from a manifest
fingerprint (`llb refresh-index` publishes each refresh as a new
`$DATA_DIR/llb/rag/generations/<utc-ts>/` generation; deleting the newest one rolls back).

Manifest-diff contract (dynamic-corpus-refresh): `corpus_doc_fingerprints` in
`src/llb/prep/corpus/governance.py` maps `doc_id -> fingerprint` from the same two sources as
`corpus_fingerprint` -- the canonical per-item row (content sha256 plus governance fields) when
`corpus_manifest.json` exists, else the sha256 of each committed `.md`/`.txt` file keyed by its
corpus-relative path. In both modes a document's PDF citation sidecar
(`pdf-<digest>.citations.json`) hash is folded into its fingerprint when one exists, so a
sidecar-only regeneration (page spans rebuilt, text unchanged) reads as a modified document;
sidecar-less docs keep the plain hash, so older stores stay refresh-compatible. `build-index`
records the map in `store_meta.json` as `doc_fingerprints`; `llb refresh-index` diffs it against
the current corpus to re-chunk/re-embed only added or modified documents and to drop deleted
ones (details in [RAG core](../rag-core/retrieval-store.md#store-lifecycle-dynamic-corpus-refresh)).

```bash
make ingest-corpus CORPUS_ROOT=<mixed-dir> CORPUS_OUT_DIR=<out-dir> CORPUS_MIN_CHARS=500
make ingest-corpus CORPUS_ROOT=<mixed-dir> CORPUS_DEFAULT_LANGUAGE=uk CORPUS_ACL_LABEL=<tag>
make ingest-corpus CORPUS_ROOT=<mixed-dir> CORPUS_REFRESH=1
llb ingest-corpus --root <mixed-dir> --out-dir <out-dir> --min-chars 500 --parser auto \
  --default-language uk --acl-label <tag>
```

`make quickstart-corpus CORPUS_SRC=<dir>` (script target `corpus`) generalizes the PDF quickstart
stages to a mixed corpus: `ingest-corpus` -> full-corpus index -> ontology draft -> graph ->
validate, logging each stage under `$DATA_DIR/llb/logs/quickstart/`. It reuses the PDF quickstart's
model selection, workload estimate, and confirmation gate, and drafts directly over the converted
corpus (passthrough text has no citation sidecar, so no per-doc staging step is needed). The mixed
fixture `samples/corpus/` (`.md` + `.txt`) backs the ingestion unit tests.

Ontology draft bundles preserve that PDF evidence. When a source document has a matching
`*.citations.json` sidecar, `prepare-goldset-draft` copies it into the bundle `corpus/` directory
and writes these review artifacts beside `goldset.jsonl`:

- `pdf_ontology_report.json`: parse rate, elapsed seconds, grounded entity/event/claim/fact counts,
  page-span citation coverage, citation-valid needle count, dictionary-term yield, needle-retrieval
  metrics when enabled, and quality gates with a `passed` roll-up (grounded extractions of any kind
  + a non-empty gold set, plus a citation-valid needle for PDF corpora).
- `prompt_dictionary_candidates.jsonl`: source-backed entity and relation terms with supporting
  spans and PDF page references when sidecars exist. This artifact also seeds the query-side
  glossary (see Query Glossary below).
- `needle_items.jsonl`: drafted gold items whose source spans map back to PDF page sidecars. Each
  row carries its `question_type` (closed taxonomy: factoid, definition, procedural, numeric,
  comparative, multi-hop) and `difficulty` label. When
  `prepare-goldset-draft --retrieval-index-dir <full-rag-index>` is set, each row also carries
  `retrieval_rank` and `retrieval_k`; `retrieval_rank: null` marks a citation-valid needle whose
  gold span was not retrieved from the full corpus within top-k, and the report adds
  `retrieval_unique_needle_fraction_by_question_type`.

The artifacts are diagnostics for review and construction. Drafted rows still remain
`verified=false` until the human verification gate emits an accepted ledger.

The retrieval-uniqueness check is opt-in for generic drafts and enabled by the PDF quickstart after
the full-corpus RAG store exists. Use `DRAFT_RETRIEVAL_INDEX_DIR=<data>/llb/rag` and
`DRAFT_RETRIEVAL_K=<k>` with `make prepare-goldset-draft`; add
`DRAFT_DROP_NONRETRIEVABLE_NEEDLES=1` only when the review artifact should omit misses instead of
flagging them. The report records `needle_retrieval`, `retrieval_unique_needle_items`,
`retrieval_unique_needle_fraction`, and `needle_items_written`. `has_retrieval_unique_needles` is
informational in `gates`; the existing `passed` roll-up still gates on citation-valid needles so
operators can inspect broad-but-grounded misses.

The ontology-assisted seed sampler uses entities, subject-relation-object facts, grounded claims,
and grounded events as draft targets. Seeds carry document, section, difficulty, and semantic-type
coverage strata, so a full-corpus draft can spread questions across manuals, dictionaries, and
after-action-style documents even when a document has few SRO facts.

Three opt-in yield-max knobs raise the meaningful-question yield of a draft:
`DRAFT_COVERAGE_TARGET=N` drafts up to N seeds per stratum bucket (with a `coverage_matrix`
exhaustion report) instead of a flat `DRAFT_MAX_ITEMS` cap; `DRAFT_MULTI_HOP=1` adds multi-span
chain questions walked from 2-hop knowledge-graph paths (each carrying >= 2 grounded spans); and
`DRAFT_DEDUP_AGAINST=<bundle[,bundle]>` drops questions that are pinned-E5 near-duplicates of prior
bundles (add `DRAFT_DEDUP_LINKAGE_SHADOW=1` to score the record-linkage model beside that constant
and get each rejection's match probability and level agreements -- see [the gold-item
lane](../entity-resolution.md#the-gold-item-lane)). Every drafted item is tagged with a
`question_type` and `difficulty` label reviewers and the miss analyzer can filter on. See [robust
backends and ontology drafting](../robustness-ontology-backends.md) for the module map, report
fields, and command reference.

`DRAFT_MULTI_HOP=1` alone walks strict directed `A -r1-> B -r2-> C` chains, which extracted
Ukrainian PDF graphs rarely supply: the 625-node, 213-edge public-literature graph yields exactly
one such path, so a multi-hop question slice cannot be measured from it.
`DRAFT_MULTI_HOP_BRIDGE_FILL=1` (`--multi-hop-bridge-fill`) keeps directed paths first and then
fills the path budget with the same shared-bridge fact pairs the chain lane uses -- two distinct
facts incident on one entity, cited from two distinct spans. The drafted item still has to name the
bridge or end entity in its reference answer and still has to re-ground >= 2 distinct exact spans,
so the multi-span contract is unchanged; only the supply of candidate paths widens. The strict walk
remains the default, because a directed chain is the stronger multi-hop claim.

## Widening a multi-hop review slice

`make widen-multihop-draft` turns a prior ontology-draft bundle into one larger, reviewable
multi-hop ledger. Set `MULTIHOP_DRAFT_PRIOR_BUNDLE=<bundle>`,
`MULTIHOP_DRAFT_OUT=<new-bundle>`, and `MULTIHOP_DRAFT_MODEL=<model>`. For an incremental pass,
set `MULTIHOP_DRAFT_DEDUP_AGAINST=<original-bundle>,<latest-bundle>` so flat questions in the
original draft and multi-hop additions in the latest ledger both remain inside the semantic
duplicate boundary.

The target composes four reusable `prepare-goldset-draft` controls:

- `DRAFT_REUSE_EXTRACTION_BUNDLE=<bundle>` verifies and reuses its `extraction.jsonl`, avoiding
  repeated extraction calls over unchanged corpus bytes;
- `DRAFT_MULTI_HOP_ONLY=1` skips flat-question drafting;
- `DRAFT_MULTI_HOP_PATH_STRATIFIED=1` allocates the bounded call budget across observed ordered
  relation pairs, same-document versus cross-document paths, and source documents before
  drafting;
- prior multi-hop evidence-span pairs are excluded before the path cap is applied, so the model
  call budget counts unseen graph paths rather than redraws;
- `DRAFT_CARRY_FORWARD_MULTI_HOP=1` prepends the prior labeled multi-hop rows, collapses inherited
  exact-question duplicates, and emits one complete `verify_sample.csv`.

The three independent per-stratum targets are
`MULTIHOP_DRAFT_RELATION_PAIR_TARGET`,
`MULTIHOP_DRAFT_DOCUMENT_MODE_TARGET`, and
`MULTIHOP_DRAFT_SOURCE_DOCUMENT_TARGET`. The deterministic allocator writes
`multihop_path_strata.json` and the same payload in provenance. Every category reports its target,
available candidates, selected paths, and one of `covered`, `exhausted`, or `budget-exhausted`.
Only the first two states satisfy readiness: `budget-exhausted` means known candidates remain and
the operator must raise the path budget or narrow the requested strata. Documents with no
candidate path and an unavailable same/cross-document mode are recorded as exhausted rather than
silently omitted.

Prior questions are also supplied to the multi-hop prompt as bounded novelty guidance. This is an
efficiency hint, not an acceptance shortcut: the pinned-E5 `DRAFT_DEDUP_AGAINST` filter still
decides which new drafts survive. Flat rows retain the question-only 0.90 cosine rule. Multi-hop
rows always drop an exact normalized-question repeat; a non-exact row is dropped only when its
question cosine is at least 0.90 and its reference-answer cosine against the same prior row is at
least 0.95. The nearest prior question, answer, and both similarities are recorded for every
rejection. This keeps common domain wording from erasing a distinct two-fact answer while
preserving an inspectable duplicate boundary. The final `audit-multihop-draft` gate writes
`multihop_expansion_report.json` and fails unless the combined ledger meets a declared relative
headroom requirement, records prior-question dedup, contains only Ukrainian-gated multi-hop rows,
and every row re-grounds at least two distinct exact spans. The required combined size is derived
from the carried ledger and `MULTIHOP_DRAFT_MIN_HEADROOM_FRACTION`; no corpus-specific accepted-row
floor is embedded in the command. Exact normalized questions are collapsed within the new batch
before the prior-bundle semantic comparison, and both rejection kinds remain in dedup provenance.
For text-only bundles, where the PDF citation-needle sidecar is intentionally empty, a
`multi_hop_only` provenance setting provides the lossless label fallback needed for carry-forward
and prior-span exclusion.

CUDA acceptance was run 2026-07-28 on the RTX PRO 3000 Blackwell 12 GiB CUDA host. The final bounded
lane used the committed seven-document Ukrainian conflict corpus and `qwen3:14b` over Ollama. Its
initial bundle supplied one carried multi-hop row. The widening pass reused all 48 extracted facts
with zero extraction calls, selected all 30 unseen candidates, and spent 30 drafting calls. Of 22
grounded model rows, three exact intra-batch repeats and one prior-bundle near-duplicate were
rejected, leaving 18 new rows plus the carried row in one worksheet. Five selected paths were
same-document and 25 were cross-document; six source documents were covered and the document with no
available path was explicitly exhausted. The final audit reports exact spans and Ukrainian output
for every row, `path_strata_ready: true`, relative review headroom of 18.0 against the declared 0.15
minimum, and `ready_for_human_review: true`. Reading: the widening pass is CHEAP where it matters --
zero extraction calls because all 48 facts were reused, 30 drafting calls for 18 surviving new rows
-- and the composed gates catch what they are meant to, since 4 of 22 grounded rows were rejected as
repeats rather than silently kept. This small carried baseline validates the workflow and gate
composition; it is not a substitute for the human-reviewed goods ledger, and one prior-bundle
near-duplicate rejection exercises that rule far more lightly than a corpus with a wide prior bundle
would. Lookup key: `graph-vector-fusion-multihop` run `relation-strata-cuda`.

## Yield-max empirical acceptance

The 2026-07-12 local acceptance preparation compares coverage-target sampling with the flat
180-seed cap on the public one-document PDF quickstart corpus. Both lanes use `gemma4:e4b`, seed 13,
the same 55-window extraction journal, a 16,384-token context, and the pinned-E5 store at
`$DATA_DIR/llb/rag`; this holds extraction and retrieval constant while changing seed selection.
Measured 2026-07-12.

| Lane | Raw seeds | Grounded needles | Retrieval-unique needles | Unique fraction |
| --- | ---: | ---: | ---: | ---: |
| Coverage target 6, 240 safety ceiling | 240 | 215 | 194 | 0.9023 |
| Flat cap | 180 | 165 | 149 | 0.9030 |

Both bundles have parse rate 1.0, pass the PDF calibration gates, and pass `validate-goldset`.
Coverage-target sampling therefore prepared 50 more citation-valid needles and 45 more
retrieval-unique needles. Both deterministic 40-row human samples accepted 40/40 items, for equal
1.0 accept rates and 0.0 reject rates at tolerance 0.05. The coverage-target lane therefore passes
the "more citation-valid needles at an equal-or-better accept rate" gate. Both accepted ledgers
pass `validate-goldset` and live under each bundle's `accepted/goldset.jsonl`.

| Question type | Coverage-target unique fraction | Flat-cap unique fraction |
| --- | ---: | ---: |
| Comparative | 0.8889 | 0.8571 |
| Definition | 1.0000 | 1.0000 |
| Factoid | 0.9000 | 0.9123 |
| Numeric | 0.8696 | 0.8500 |
| Procedural | 0.9231 | 0.8889 |
| All types | 0.9023 | 0.9030 |

The drafting contract is Ukrainian-only for user-facing text, including bilingual source corpora:
`question` and `reference_answer` must be Ukrainian, while `answer_span` remains an exact quote in
the source language so evidence offsets stay verifiable. `prep.ontology.draft` and
`prep.ontology.multi_hop` state that foreign evidence must be translated rather than copied into
the reference answer. `src/llb/prep/ontology/language.py` applies a deterministic
Ukrainian-marked, Cyrillic-dominant gate in both flat and multi-hop refinement. The current bundles,
worksheets, and accepted ledgers have zero question/answer violations under that gate; the flat and
coverage runs rejected one and two model outputs respectively for failing it.

That run emitted one deterministic 40-row review worksheet per lane
(`coverage-target` and `flat-cap-180`), and their acceptance commands emitted a 40-item verified
ledger each.
