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

## Assisted corpus drafting

`GOLDSET_MODE=draft` is reserved for Milestone M4.4. It will scan a supplied corpus, extract
entities and evidence-backed relations, induce an ontology candidate, and use a configured
internal or external inference endpoint to draft unverified QA items. It is intentionally not
implemented as a thin synonym for the existing one-prompt frontier utility.
