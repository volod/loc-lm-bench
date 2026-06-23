# Data prep (Milestone 0)

Runtime output lands under `$DATA_DIR` (default `.data/`, gitignored). The stable public
development fixture is committed under `samples/goldsets/`. Module detail is in
[implementation/current.md](../implementation/current.md).

## Gold set

    make validate-goldset     # committed 250-item fixture: spans + splits
    make gen-rag-items        # tiny generated format fixture under DATA_DIR
    make ingest-uk-squad GOLDSET_MODE=development
    # reproduce reviewed fixture from pinned HF source
    *
    make ingest-uk-squad GOLDSET_MODE=skeleton # editable from-scratch template
    make ingest-uk-squad GOLDSET_MODE=draft CORPUS=<dir>
    # M4.4 ontology-assisted draft (verified=false)

`*` may need `HF_TOKEN` in `.env` (the `goldset` extra is installed by `make venv`). The
committed default requires no token. For a local SQuAD export:
`make ingest-squad SQUAD_JSON=path.json`. See the
[from-scratch guide](goldset-from-scratch.md) for authoring and review rules.

Schema (one JSON object per line): `id, lang, question, reference_answer, source_doc_id,
source_spans[{doc_id, char_start, char_end, text}], provenance, verified, split`. Labels are
SOURCE-SPAN (char offsets, not chunk ids). Only `verified: true` items score models. Fresh
imports start false, then the ingester replaces matching IDs with canonical items and corpus
files from the committed human-reviewed fixture. Unmatched IDs remain false. Use repeatable
`--verified-goldset <reviewed.jsonl>` options for reviewed M3.5 draft/planted-label bundles, or
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

    make calibration-worksheet # blank worksheet from the calibration split
    # after Milestone 1 produces model answers, fill the ratings, then:
    python -m llb.judge.calibration score --ratings <file> # rho + CI; gate at
    0.6
