# Data prep (Milestone 0)

All output lands under `.data/llb/` (gitignored). Module detail is in
[implementation/current.md](../implementation/current.md).

## Gold set

    make gen-rag-items        # tiny sample gold set + corpus (format demo)
    make ingest-uk-squad      # real 250-item UA gold set from HPLT/ua-squad *
    make validate-goldset     # spans resolve + splits disjoint (acceptance)

`*` needs `HF_TOKEN` in `.env` (the `goldset` extra is installed by `make venv`). For a
local SQuAD export instead: `make ingest-squad SQUAD_JSON=path.json`.

Schema (one JSON object per line): `id, lang, question, reference_answer, source_doc_id,
source_spans[{doc_id, char_start, char_end, text}], provenance, verified, split`. Labels are
SOURCE-SPAN (char offsets, not chunk ids). Only `verified: true` items score models;
public-reused items start `false` pending human review.

## RAG store (chunking)

    make build-rag-store               # chunk samples/corpus, all strategies
    make build-rag-store CORPUS_DIR=…  # your own documents

Strategies: `fixed`, `sentence` (never cuts mid-sentence), `recursive` (paragraph ->
sentence -> char). Each chunk carries doc id + char offsets. Add `--embed` (with the `[rag]`
extra) to also build a FAISS index per strategy.

## Judge calibration

    make calibration-worksheet         # blank worksheet from the calibration split
    # after Milestone 1 produces model answers, fill the ratings, then:
    python -m llb.judge.calibration score --ratings <file>   # rho + CI; gate at 0.6
