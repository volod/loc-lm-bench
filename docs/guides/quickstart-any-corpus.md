# Quickstart Any Corpus (mixed txt/md/pdf)

This guide runs the corpus-prep track against a **mixed** source directory containing any
combination of `.txt`, `.md`, and `.pdf` documents. It generalizes
[Quickstart PDF Corpus](quickstart-pdf-corpus.md): PDFs are converted with PyMuPDF4LLM / Docling
OCR, while `.md`/`.txt` files pass through verbatim, and the whole set is drafted as one corpus.

Like every draft flow, the output is source material only: drafted rows are `verified=false` and
cannot score a model. Continue to scoring only after `verify-review` and `verify-accept` emit an
accepted ledger (the PDF-corpus review/accept/score wrappers work on this bundle too).

## One command

```sh
make quickstart-corpus CORPUS_SRC=<your-mixed-dir>
```

`CORPUS_SRC` is set via `QUICKSTART_CORPUS_SRC` (default `$DATA_DIR/quickstart-corpus`). The wrapper
writes timestamped logs under `$DATA_DIR/llb/logs/quickstart/` and runs the stages:
ingest -> full-corpus RAG index -> ontology/goldset draft -> graph -> validation.

## The same flow in groups

```sh
make quickstart-corpus-convert    # llb ingest-corpus: mixed dir -> one .md/.txt corpus
make quickstart-corpus-index      # build the full-corpus FAISS index
make quickstart-corpus-draft      # select drafter + draft the unverified goldset/ontology
make quickstart-corpus-graph      # build graph artifacts from the draft bundle
make quickstart-corpus-validate   # validate draft structure + retrieval
```

Model selection, the character-based workload estimate, and the confirmation gate are shared with
the PDF quickstart (`QUICKSTART_DRAFT_MODEL`, `QUICKSTART_DRAFT_MAX_ITEMS`, `QUICKSTART_ASSUME_YES`,
etc.). See [Quickstart PDF Corpus](quickstart-pdf-corpus.md) for those knobs.

## Environment

```sh
export QUICKSTART_CORPUS_SRC=.data/quickstart-corpus          # mixed txt/md/pdf input
export QUICKSTART_CORPUS_MD=.data/quickstart-corpus-md        # converted corpus (RAG/ontology input)
export QUICKSTART_CORPUS_RAG_DATA=.data/quickstart-corpus-rag
export QUICKSTART_CORPUS_DRAFT=.data/quickstart-corpus-draft  # the review bundle
export QUICKSTART_CORPUS_GRAPH_DATA=.data/quickstart-corpus-graph
export QUICKSTART_CORPUS_MIN_CHARS=500                        # skip shorter documents
export QUICKSTART_CORPUS_PARSER=auto                          # PDF parser selection
```

## Ingest on its own

```sh
make ingest-corpus CORPUS_ROOT=<mixed-dir> CORPUS_OUT_DIR=<out-dir> CORPUS_MIN_CHARS=500
llb ingest-corpus --root <mixed-dir> --out-dir <out-dir> --min-chars 500 --parser auto
```

Ingestion is incremental: it writes a unified `corpus_manifest.json` with a `source_sha256` per
document, and a rerun over an unchanged corpus reports `reused: true` for every document. PDFs keep
their `pdf-<digest>.md` ids and citation sidecars; `.md`/`.txt` keep their relative path as the doc
id. `CORPUS_REFRESH=1` forces a full reconversion. The default output `<root>/_md` is excluded from
the source walk, so it is never re-ingested as new input.

## Resuming an interrupted draft

A full-corpus draft can run for hours. If it is interrupted, resume it instead of restarting:

```sh
make quickstart-corpus-draft QUICKSTART_CORPUS_RESUME=$QUICKSTART_CORPUS_DRAFT
# or directly:
make prepare-goldset-draft DRAFT_RESUME=<bundle>
llb prepare-goldset-draft --resume <bundle>
```

Resume reuses the completed extraction windows recorded in the bundle's `extraction_journal.jsonl`,
re-extracts only the missing windows, and replays the deterministic seed/draft/emit stages, so the
finished bundle matches an uninterrupted run. See
[data prep](../impl/current/data-prep.md#resumable-extraction-interrupt-safe-drafting) for the
mechanics.

## See also

- [Quickstart PDF Corpus](quickstart-pdf-corpus.md) -- the PDF-only track and shared draft knobs.
- [Create a gold set (end-to-end)](goldset-from-scratch.md) -- the full create -> verify -> score spine.
- [Data prep](../impl/current/data-prep.md) -- ingestion, drafting, and resume behavior in brief.
