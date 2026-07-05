# PDF Corpus To Goldset And Graph Quickstart Commands

This is the command-chain layer of the PDF corpus track: the workflow at a glance, the wrapper
targets, and the human-verification-gate context are in
[Quickstart PDF corpus](quickstart-pdf-corpus.md) -- read its at-a-glance section first if this
is your first run, because scoring is gated on a human-accepted ledger.

The granular commands below are the same operations without the wrapper orchestration:

```sh
# Purpose: name source and generated artifact roots with a consistent hyphenated prefix.
# Default input: source PDFs in .data/quickstart-pdf-corpus.
# Output/result: markdown, RAG, graph, and draft artifacts land in sibling quickstart roots.
export PDF_SOURCE=.data/quickstart-pdf-corpus
export PDF_MD=.data/quickstart-pdf-corpus-md
export PDF_RAG_DATA=.data/quickstart-pdf-corpus-rag
export PDF_DRAFT_MD=.data/quickstart-pdf-corpus-draft-md
export PDF_DRAFT=.data/quickstart-pdf-corpus-draft
export PDF_GRAPH_DATA=.data/quickstart-pdf-corpus-graph

# Purpose: install OCR/layout extras when PDFs include scans.
# Default input: pyproject.toml pdf-quality extra and system OCR packages.
# Output/result: Docling/RapidOCR path is available for image-only PDFs.
make venv EXTRAS=pdf-quality

# Purpose: convert PDFs into markdown files with citation sidecars.
# Default input: PDF_DIR=$PDF_SOURCE, PDF_MIN_CHARS=500, PDF_PARSER=auto.
# Output/result: markdown corpus and quality reports under $PDF_MD.
make pdf-to-markdown PDF_DIR=$PDF_SOURCE PDF_OUT_DIR=$PDF_MD PDF_PARSER=auto

# Purpose: build the full vector index for the converted corpus.
# Default input: CORPUS=$PDF_MD.
# Output/result: FAISS RAG store under $PDF_RAG_DATA/llb/rag/.
env DATA_DIR=$PDF_RAG_DATA make build-index CORPUS=$PDF_MD

# Purpose: stage the full converted corpus for drafting.
# Default input: every converted markdown/citation sidecar under $PDF_MD.
# Output/result: draft input corpus under $PDF_DRAFT_MD.
rm -rf $PDF_DRAFT_MD
mkdir -p $PDF_DRAFT_MD
cp -R $PDF_MD/*.md $PDF_MD/*.citations.json $PDF_DRAFT_MD/

# Purpose: draft unverified gold items, needle items, and ontology from the full corpus.
# Default input: selected local or frontier drafter; this example pins the recommended local model.
# Output/result: goldset.jsonl, ontology.json, extraction.jsonl, provenance, and verify_sample.csv.
make prepare-goldset-draft DRAFT_CORPUS=$PDF_DRAFT_MD \
  DRAFT_MODEL=hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M \
  DRAFT_ENDPOINT=local DRAFT_MAX_ITEMS=180 DRAFT_VERIFY_N=40 DRAFT_NO_THINK=1 \
  DRAFT_OUT_DIR=$PDF_DRAFT DRAFT_TIMEOUT=900

# Optional bounded probe for debugging only.
QUICKSTART_PDF_DRAFT_DOCS="pdf-d2e2499d3d06 pdf-b117ebb25eb7" \
  QUICKSTART_DRAFT_MODEL=hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M \
  QUICKSTART_DRAFT_MAX_ITEMS=8 QUICKSTART_DRAFT_VERIFY_N=4 \
  make quickstart-pdf-corpus-draft

# Purpose: build a knowledge graph from the draft bundle's ontology extraction.
# Default input: BUNDLE=$PDF_DRAFT.
# Output/result: nodes, edges, communities, and graph metadata under $PDF_GRAPH_DATA/llb/graph/.
env DATA_DIR=$PDF_GRAPH_DATA make build-graph BUNDLE=$PDF_DRAFT

# Purpose: validate draft structure and retrieval before human review.
# Default input: draft goldset/corpus and full-corpus RAG index.
# Output/result: structural PASS and recall/MRR gate output.
make validate-goldset GOLDSET=$PDF_DRAFT/goldset.jsonl CORPUS=$PDF_DRAFT/corpus
env DATA_DIR=$PDF_RAG_DATA HF_HUB_OFFLINE=1 make validate-retrieval \
  GOLDSET=$PDF_DRAFT/goldset.jsonl RAG_K=10

# Purpose: human verification gate; only accepted ledgers may be scored.
# Default input: verify_sample.csv created by the draft command.
# Output/result: $PDF_DRAFT/accepted/goldset.jsonl and corpus/ when accepted.
make verify-review VERIFY_WS=$PDF_DRAFT/verify_sample.csv
make verify-accept BUNDLE=$PDF_DRAFT VERIFY_WS=$PDF_DRAFT/verify_sample.csv
```

After `verify-accept` emits `$PDF_DRAFT/accepted/`, run `make quickstart-pdf-corpus-score` or
rerun the granular goldset scoring commands with
`GOLDSET=$PDF_DRAFT/accepted/goldset.jsonl` and `CORPUS=$PDF_DRAFT/accepted/corpus`.
To run the full goldset leaderboard quickstart against the accepted PDF-derived set, point the
Track A wrapper at that accepted ledger:

```sh
make quickstart-goldset \
  QUICKSTART_A_GOLDSET=$PDF_DRAFT/accepted/goldset.jsonl \
  QUICKSTART_A_CORPUS=$PDF_DRAFT/accepted/corpus \
  QUICKSTART_A_DATA_DIR=.data/quickstart-pdf-corpus-leaderboard \
  QUICKSTART_A_SWEEP_ID=quickstart-pdf-corpus \
  QUICKSTART_RECOMMEND_MIN_CASES=1
```

For an existing text or markdown corpus, set `PDF_MD=<existing-corpus-dir>` and start at
`make build-index`; keep the same draft, graph, verification, and post-acceptance steps.

Run `make` with no target to list commands. `.env.example` documents runtime settings.
