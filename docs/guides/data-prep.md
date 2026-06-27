# Data prep (data prep)

Runtime output lands under `$DATA_DIR` (default `.data/`, gitignored). The stable public
development fixture is committed under `samples/goldsets/`. Module detail is in
[implementation/current.md](../impl/current.md).

## Gold set

    make validate-goldset     # committed 250-item fixture: spans + splits
    make gen-rag-items        # tiny generated format fixture under DATA_DIR
    make ingest-uk-squad GOLDSET_MODE=development
    # reproduce reviewed fixture from pinned HF source
    *
    make ingest-uk-squad GOLDSET_MODE=skeleton # editable from-scratch template
    make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<dir>
    # ontology-assisted draft (verified=false)

`*` may need `HF_TOKEN` in `.env` (the `goldset` extra is installed by `make venv`). The
committed default requires no token. For a local SQuAD export:
`make ingest-squad SQUAD_JSON=path.json`.

A drafted set is `verified=false` and cannot score a model until it clears the data gates --
`make cross-check-goldset` (second-frontier re-confirm) then the human verification gate sample-verify
(`make verify-sample` / `verify-review` / `verify-accept`) and the ledger flip. The full ordered
flow, with authoring + review rules, is the
[create-a-gold-set workflow](goldset-from-scratch.md).

Schema (one JSON object per line): `id, lang, question, reference_answer, source_doc_id,
source_spans[{doc_id, char_start, char_end, text}], provenance, verified, split`. Labels are
SOURCE-SPAN (char offsets, not chunk ids). Only `verified: true` items score models. Fresh
imports start false, then the ingester replaces matching IDs with canonical items and corpus
files from the committed human-reviewed fixture. Unmatched IDs remain false. Use repeatable
`--verified-goldset <reviewed.jsonl>` options for reviewed frontier draft/planted-label bundles, or
`--no-verification-ledger` for a raw import. Each custom JSONL must have a sibling `corpus/`.

The development target pins the exact `FIdo-AI/ua-squad` revision and validation split recorded
by the fixture, normalizes its nested SQuAD rows, and applies the same context-diverse selection.
The resulting 250 items are all adopted from the reviewed ledger and exactly match the committed
items and corpus. Use the committed fixture for normal initial model tests; regenerate only to
exercise ingestion or verify reproducibility.

## RAG store (chunking)

    make build-rag-store               # chunk samples/corpus, all strategies
    make build-rag-store CORPUS_DIR=…  # your own documents

Strategies: `fixed`, `sentence` (never cuts mid-sentence), `recursive` (paragraph ->
sentence -> char). Each chunk carries doc id + char offsets. Add `--embed` (with the `[rag]`
extra) to also build a FAISS index per strategy.

## Judge calibration

    make calibration-run    # candidate answers + ungated judge ratings -> worksheet (CAL_WS)
    make calibration-rate   # interactive: fill human ratings/answers (judge column hidden)
    make calibration-score  # rho + bootstrap CI + trust decision; gate at 0.6

`make calibration-worksheet` emits a BLANK worksheet (rows only) when you want the structure
without a run. Full walkthrough, the rater command reference, and the new-goldset /
text-corpus-draft cases: the [calibration-tooling manual](calibration-tooling.md).
