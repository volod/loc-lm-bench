# Quickstart PDF Corpus

This guide runs the README corpus-prep track against the local test PDF corpus in
`.data/quickstart-pdf-corpus` instead of the committed UA-SQuAD fixture. It uses hyphenated
quickstart artifact roots so the default fixture artifacts stay untouched and path names remain
consistent.

The PDF corpus is source material only. Model scoring commands such as `demo-eval`, `sweep`, and
`pipeline` require a `verified=true` gold set. The draft commands below produce review artifacts;
continue to scoring only after `verify-review` and `verify-accept` emit an accepted ledger.

## Wrapper Commands

Use the wrapper targets for the normal quickstart. They write timestamped logs under
`$DATA_DIR/llb/logs/quickstart/` with step headings, the exact commands, important metrics, and
`[result]` artifact summaries.

```sh
# PDF corpus prep up to the human verification gate.
make quickstart-pdf-corpus

# The same flow split into groups for experiments and debugging.
make quickstart-pdf-corpus-convert
make quickstart-pdf-corpus-index
make quickstart-pdf-corpus-draft
make quickstart-pdf-corpus-graph
make quickstart-pdf-corpus-validate

# Human gate and post-acceptance scoring.
make quickstart-pdf-corpus-review
make quickstart-pdf-corpus-accept
make quickstart-pdf-corpus-score
```

## Environment

```sh
export PDF_SOURCE=.data/quickstart-pdf-corpus
export PDF_MD=.data/quickstart-pdf-corpus-md
export PDF_RAG_DATA=.data/quickstart-pdf-corpus-rag
export PDF_DRAFT_MD=.data/quickstart-pdf-corpus-draft-md
export PDF_DRAFT=.data/quickstart-pdf-corpus-draft
export PDF_GRAPH_DATA=.data/quickstart-pdf-corpus-graph
```

Use `SKIP_APT=1` only on hosts that already have the OCR system packages installed. On a fresh
host, omit it so `make venv` can install the apt packages declared by the project.

```sh
make venv SKIP_APT=1
make venv SKIP_APT=1 EXTRAS=pdf-quality
```

The first command keeps the normal local workflow extras ready. The second command adds Docling,
RapidOCR, MarkItDown, and Unstructured for scanned-PDF recovery. The first OCR run may need network
access once so Docling can cache its layout model snapshots. The first RAG index build may also need
network access once to cache `intfloat/multilingual-e5-base`.

## Convert PDFs

```sh
make pdf-to-markdown \
  PDF_DIR=$PDF_SOURCE \
  PDF_OUT_DIR=$PDF_MD \
  PDF_MIN_CHARS=500 \
  PDF_PARSER=auto
```

Expected artifacts:

- `$PDF_MD/pdf-<digest>.md`
- `$PDF_MD/pdf-<digest>.citations.json`
- `$PDF_MD/pdf_corpus_manifest.json`
- `$PDF_MD/pdf_corpus_quality.json`

Validated result for the local test corpus:

- 19 of 19 PDFs extracted.
- 16 PDFs used PyMuPDF4LLM.
- 3 image-only PDFs used Docling OCR.
- 0 PDFs were skipped.

## Build The Full Index

```sh
env DATA_DIR=$PDF_RAG_DATA make build-index CORPUS=$PDF_MD
```

Validated result:

- RAG store: `$PDF_RAG_DATA/llb/rag/`
- Chunks: 12,745
- Vector store: FAISS
- Embedding: `intfloat/multilingual-e5-base`
- Dimensions: 768

After the embedder is cached, use offline mode for later retrieval checks:

```sh
export HF_HUB_OFFLINE=1
```

## Draft A Reviewable Gold Set

The wrapper draft target is the normal path. It stages all converted markdown documents, selects a
drafter, estimates the full draft duration, asks for confirmation, and writes the review bundle.
When `QUICKSTART_DRAFT_MODEL=auto`, the model-selection step uses existing benchmark evidence when
available; otherwise it prompts to run the local committed-goldset benchmark, select a local model
manually, or opt into a frontier `litellm` route.

```sh
make quickstart-pdf-corpus-draft
```

To force a benchmark before drafting:

```sh
QUICKSTART_MODEL_SELECTION=benchmark make quickstart-pdf-corpus-draft
```

To pin a local model and skip the model-selection prompt:

```sh
QUICKSTART_DRAFT_MODEL=hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M \
  make quickstart-pdf-corpus-draft
```

To opt into an external provider, set the provider API key expected by `litellm`, then run:

```sh
QUICKSTART_DRAFT_ENDPOINT=frontier QUICKSTART_DRAFT_MODEL=<litellm-model-id> \
  make quickstart-pdf-corpus-draft
```

Expected artifacts:

- `$PDF_DRAFT/goldset.jsonl`
- `$PDF_DRAFT/corpus/`
- `$PDF_DRAFT/extraction.jsonl`
- `$PDF_DRAFT/ontology.json`
- `$PDF_DRAFT/provenance.json`
- `$PDF_DRAFT/verify_sample.csv`
- `$PDF_DRAFT/pdf_ontology_report.json`
- `$PDF_DRAFT/prompt_dictionary_candidates.jsonl`
- `$PDF_DRAFT/needle_items.jsonl`

Default full-draft knobs:

- `QUICKSTART_PDF_DRAFT_DOCS=all`
- `QUICKSTART_DRAFT_MODEL=auto`
- `QUICKSTART_DRAFT_MAX_ITEMS=180`
- `QUICKSTART_DRAFT_VERIFY_N=40`
- `QUICKSTART_DRAFT_TIMEOUT=900`

Optional bounded probe for debugging only:

```sh
QUICKSTART_PDF_DRAFT_DOCS="pdf-d2e2499d3d06 pdf-b117ebb25eb7" \
  QUICKSTART_DRAFT_MODEL=hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M \
  QUICKSTART_DRAFT_MAX_ITEMS=8 QUICKSTART_DRAFT_VERIFY_N=4 \
  make quickstart-pdf-corpus-draft
```

All drafted items remain `verified=false` until the human gate accepts them.

## Build The Knowledge Graph

Build a GraphRAG store from the draft bundle's ontology extraction:

```sh
env DATA_DIR=$PDF_GRAPH_DATA make build-graph BUNDLE=$PDF_DRAFT
```

Expected result:

- Graph store: `$PDF_GRAPH_DATA/llb/graph/`
- Nodes, edges, communities, and metadata derived from the full draft extraction.

## Validate Retrieval

Validate the draft structure against its copied corpus:

```sh
make validate-goldset \
  GOLDSET=$PDF_DRAFT/goldset.jsonl \
  CORPUS=$PDF_DRAFT/corpus
```

Then validate retrieval against the full 19-PDF FAISS index:

```sh
env DATA_DIR=$PDF_RAG_DATA HF_HUB_OFFLINE=1 make validate-retrieval \
  GOLDSET=$PDF_DRAFT/goldset.jsonl \
  RAG_K=10
```

The validation output prints the drafted item count, recall@10, and MRR for the current full
draft bundle.

## Human Verification Gate

Do not run model scoring on this draft until a human accepts the sample and emits an accepted
ledger.

```sh
make verify-review VERIFY_WS=$PDF_DRAFT/verify_sample.csv
make verify-accept \
  BUNDLE=$PDF_DRAFT \
  VERIFY_WS=$PDF_DRAFT/verify_sample.csv \
  VERIFY_TOLERANCE=0.05
```

If accepted, the verified bundle is written to:

```text
$PDF_DRAFT/accepted/
  goldset.jsonl
  corpus/
```

## Score After Acceptance

Only after the accepted ledger exists:

```sh
export PDF_ACCEPTED=$PDF_DRAFT/accepted

make quickstart-goldset \
  QUICKSTART_A_GOLDSET=$PDF_ACCEPTED/goldset.jsonl \
  QUICKSTART_A_CORPUS=$PDF_ACCEPTED/corpus \
  QUICKSTART_A_DATA_DIR=.data/quickstart-pdf-corpus-leaderboard \
  QUICKSTART_A_SWEEP_ID=quickstart-pdf-corpus \
  QUICKSTART_RECOMMEND_MIN_CASES=1

make demo-eval \
  DATA_DIR=.data/quickstart-pdf-corpus-leaderboard \
  ALL_GOLDSET=$PDF_ACCEPTED/goldset.jsonl \
  ALL_CORPUS=$PDF_ACCEPTED/corpus \
  MODEL=<selected-local-model> \
  BACKEND=ollama \
  LIMIT=2
```

For a direct single-model run:

```sh
env DATA_DIR=.data/quickstart-pdf-corpus-leaderboard make build-index \
  CORPUS=$PDF_ACCEPTED/corpus
env DATA_DIR=.data/quickstart-pdf-corpus-leaderboard HF_HUB_OFFLINE=1 make run-eval \
  MODEL=<selected-local-model> \
  BACKEND=ollama \
  GOLDSET=$PDF_ACCEPTED/goldset.jsonl \
  LIMIT=2
```

For a sweep, use the committed candidate manifest:

```sh
env DATA_DIR=.data/quickstart-pdf-corpus-leaderboard HF_HUB_OFFLINE=1 make sweep \
  SWEEP_ID=quickstart-pdf-corpus \
  MODELS_MANIFEST=samples/models_uk.yaml \
  GOLDSET=$PDF_ACCEPTED/goldset.jsonl \
  SPLIT=final
```

`make pipeline` additionally requires existing public-screen reports under `$DATA_DIR/screen/`:

```sh
env DATA_DIR=.data/quickstart-pdf-corpus-leaderboard make pipeline \
  MODELS_MANIFEST=samples/models_uk.yaml \
  GOLDSET=$PDF_ACCEPTED/goldset.jsonl \
  PIPELINE_TOP_N=2 \
  PIPELINE_TRIALS=20
```

Inspect scoring artifacts after `run-eval`, `demo-eval`, `sweep`, or `pipeline` writes run bundles:

```sh
env DATA_DIR=.data/quickstart-pdf-corpus-leaderboard make board
env DATA_DIR=.data/quickstart-pdf-corpus-leaderboard make mlflow
```

`make board` serves `http://127.0.0.1:8501`; `make mlflow` serves
`http://127.0.0.1:5000`.
