# Prepare a gold set from scratch

The benchmark separates stable public development data from user-specific private evaluation
data. `make demo-eval` uses the committed public fixture and never regenerates it. Arbitrary
runtime imports and drafts remain unverified until explicitly reviewed.

## Choose a workflow

### Stable public development fixture

No gold-set generation or dataset download is required:

    make validate-goldset
    make build-index
    make run-eval MODEL=llama3.2:3b LIMIT=20

`build-index` may download the pinned embedding on its first run, and evaluation still requires
the selected model/backend. `validate-goldset` itself is fully offline.

The default paths point to `samples/goldsets/ua_squad_postedited_v1/`. Its README records the
upstream revision, selection rule, verification basis, attribution, and data license.

### Reproduce the reviewed development fixture

Use this to exercise ingestion against the exact pinned source behind the fixture:

    make ingest-uk-squad GOLDSET_MODE=development GOLDSET_N=250

The command uses the code-owned `--pinned-development-source` profile: exact dataset revision,
validation split, and first-grounded-QA-per-context selection. Its 250 output items are adopted
from the reviewed ledger and are all verified. Output is written under
`$DATA_DIR/llb/goldset/goldset_uk_development.jsonl`, with matching documents under
`$DATA_DIR/llb/corpus/`. For a different HF source, invoke `llb.prep.ingest_squad` directly;
nonmatching IDs remain unverified.

### Manual skeleton

Create a timestamped, editable SQuAD file plus concise instructions:

    make ingest-uk-squad GOLDSET_MODE=skeleton

The command writes `$DATA_DIR/goldset-skeleton/<timestamp>/squad_goldset.json`. Replace the
example with your own contexts and QA pairs, then import it:

    make ingest-squad SQUAD_JSON=<path-to-edited-squad-json>

The imported canonical JSONL remains unverified. Review it before setting accepted items to
`verified: true` and, for locally reviewed items, `provenance: human-verified`.

## Authoring rules

For every item:

1. Use a stable unique id and one clear Ukrainian question.
2. Ensure the answer is supported by the supplied context.
3. Copy the answer verbatim and record its zero-based character offset.
4. Avoid ambiguous questions, duplicate facts, and clues that expose the answer trivially.
5. Preserve calibration, tuning, and final split isolation after canonical import.
6. Record reviewer, decision, timestamp, and notes in a sidecar review log.

Validate the result against its corpus:

    make validate-goldset GOLDSET=<canonical.jsonl> CORPUS=<corpus-dir>

Structural validation checks ids, files, offsets, exact span text, and split counts. Human
review remains responsible for factual correctness, question quality, and sufficient evidence.

## Assisted corpus drafting (ontology-assisted, M4.4)

`GOLDSET_MODE=draft` runs the ontology-assisted draft pipeline over a supplied corpus: it
inventories the docs, extracts entities and evidence-backed relations, induces an ontology
candidate, samples for coverage, and drafts unverified QA items -- all through one configured
inference endpoint. It is intentionally not a thin synonym for the one-prompt frontier utility.

    make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<corpus-dir> DRAFT_MODEL=<tag>

By default the endpoint is LOCAL (an OpenAI-compatible server such as Ollama; no corpus leaves
the box). Opt into a frontier endpoint with `DRAFT_ENDPOINT=frontier` (egress; needs a provider
key). The CLI form is `llb prepare-goldset-draft --corpus-root <dir> --model <id>
[--endpoint local|frontier] [--base-url <url>] [--max-items N]`.

The run writes a self-contained bundle under `$DATA_DIR/prepare-goldset/<timestamp>/`:
`goldset.jsonl` (every item `verified: false`, `provenance: ontology-drafted`, answer spans
exact), a verbatim `corpus/` copy, the induced `ontology.json`, per-document `extraction.jsonl`,
and `provenance.json` (endpoint, prompt fingerprints, per-doc hashes, stage counts, cost).
Validate and review before promoting any item:

    make validate-goldset GOLDSET=$DATA_DIR/prepare-goldset/<timestamp>/goldset.jsonl \
      CORPUS=$DATA_DIR/prepare-goldset/<timestamp>/corpus

Drafts never score a model until a frontier cross-check and a human stratified sample-verify
accept them (set accepted items to `verified: true`, and `provenance: human-verified` for
locally reviewed items).
