# Quick Start

Requires [`uv`](https://docs.astral.sh/uv/) and a running local backend such as Ollama.
CUDA and `HF_TOKEN` are only needed for GPU serving or gated model weights.

## Goldset Leaderboard Quickstart

Use the committed, already verified UA-SQuAD fixture to build a leaderboard for the current CUDA
host. This is the fastest path for model and inference-backend comparison because it skips corpus
creation and human verification.

All quickstart wrapper targets write timestamped logs under
`$DATA_DIR/llb/logs/quickstart/`. Each log contains step headings, the exact `make` commands,
important metrics from the underlying tools, and `[result]` lines with artifact paths.

```sh
# Purpose: run the committed-goldset leaderboard flow end to end.
# Default input: committed UA-SQuAD goldset/corpus, samples/configs/models_uk.yaml, detected CUDA tier.
# Output/result: RAG metrics, serving configs, model-prep output, sweep runs, platform-matrix
# runs, security ASR/defense metrics, prompt candidates, and a debug log under
# $DATA_DIR/llb/logs/quickstart/.
# Note: goldset quickstart skips apt provisioning by default;
#       set QUICKSTART_SKIP_APT=0 to include it.
# Note: QUICKSTART_SETUP_VENV=auto reuses .venv when present;
#       set QUICKSTART_SETUP_VENV=1 to sync.
# Note: wrapper dependency cache defaults to $DATA_DIR/uv-cache for self-contained artifacts.
make quickstart-goldset

# Purpose: run the same flow in reviewable groups for experiments and debugging.
# Default input/output/result: same as quickstart-goldset, split by pipeline stage.
make quickstart-goldset-setup
make quickstart-goldset-rag
make quickstart-goldset-models
make quickstart-goldset-eval
make quickstart-goldset-security

# Purpose: prepare prompt candidates, then pin and score one reviewed prompt id.
# Default input: committed fixture corpus;
#                reviewer selects QUICKSTART_PROMPT_ID from the summary.
# Output/result: prompt package and prompt-system comparison runs under
#                quickstart leaderboard data.
make quickstart-goldset-prompt
make quickstart-goldset-prompt QUICKSTART_PROMPT_ID=<prompt-id>
```

[See granular commands without the wrapper orchestration](quickstart-goldset-commands.md)

The default candidate-family intent is to compare the largest runnable MamayLM, Lapa, Gemma 4,
Qwen 3.6, and Mistral variants for the detected 12/16/24/32 GiB CUDA tier. The candidate and
serving manifests cover all five families: the Mistral default is Mistral Small 3.1 24B
(Apache-2.0, ungated), served via vLLM FP8 on the 32 GiB tier, vLLM w4a16 on the 24 GiB tier, and
the curated `mistral-small3.1:24b` GGUF on Ollama (CPU offload) on the 12/16 GiB tiers.

## PDF Corpus To Goldset And Graph Quickstart

Use this track when you start from PDFs, text, or markdown. It prepares corpus artifacts, drafts a
reviewable gold set and ontology, builds a knowledge graph, and then hands the accepted ledger to
the goldset scoring flow.

The all-in-one PDF corpus target intentionally stops before model scoring because drafted rows are
`verified=false`. Continue with the review, acceptance, and score targets only after a human review.

```sh
# Purpose: run PDF corpus prep end to end up to the verification gate.
# Default input: .data/quickstart-pdf-corpus PDFs and all converted markdown documents.
# Model selection: QUICKSTART_DRAFT_MODEL=auto uses the host-fit CUDA Gemma 4 tier target;
# override with benchmark, a pinned local model, or a frontier litellm route when needed.
# Output/result: converted markdown, full RAG index, draft goldset, ontology, graph,
# validation metrics, and a debug log under $DATA_DIR/llb/logs/quickstart/.
# approve the multi hour draft gate with QUICKSTART_ASSUME_YES=1
QUICKSTART_ASSUME_YES=1 make quickstart-pdf-corpus

# Purpose: run the same corpus flow in reviewable groups for experiments and debugging.
# Default input/output/result: same as quickstart-pdf-corpus, split by pipeline stage.
make quickstart-pdf-corpus-convert
make quickstart-pdf-corpus-index
make quickstart-pdf-corpus-draft
make quickstart-pdf-corpus-graph
make quickstart-pdf-corpus-validate

# Purpose: complete the human gate, then score only the accepted ledger.
# Default input: .data/quickstart-pdf-corpus-draft/verify_sample.csv and accepted ledger.
# Output/result: accepted verified goldset/corpus, then sweep artifacts under
# .data/quickstart-pdf-corpus-leaderboard/.
make quickstart-pdf-corpus-review
make quickstart-pdf-corpus-accept
make quickstart-pdf-corpus-score
```

Common model-selection overrides:

```sh
# Default: resolve the context-capable Gemma 4 target for this CUDA tier.
QUICKSTART_MODEL_SELECTION=auto make quickstart-pdf-corpus

# On the 16 GiB CUDA tier this selects Gemma 4 12B vLLM with a 16k context,
# 16 GiB CPU weight offload, and a 32 GiB CPU KV offload buffer.

# Approve the full-draft confirmation gate in the logged all-in-one wrapper.
QUICKSTART_ASSUME_YES=1 make quickstart-pdf-corpus

# Use benchmark evidence from the committed-goldset quickstart,
# then draft the full PDF corpus.
QUICKSTART_MODEL_SELECTION=benchmark make quickstart-pdf-corpus

# Pin a known local model and skip the model-selection prompt.
QUICKSTART_DRAFT_MODEL=hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M \
  make quickstart-pdf-corpus

# Opt into an external provider through litellm. The workflow names the corpus and
# destination in its consent prompt and applies the quickstart 1000-call guard.
QUICKSTART_DRAFT_ENDPOINT=frontier QUICKSTART_DRAFT_MODEL=<litellm-model-id> \
  make quickstart-pdf-corpus
```

The exact tier buckets, candidate ranking, context filter, selected models, and propagated vLLM
offload settings are documented in
[Automatic CUDA-host draft model selection](../../inference/config-example.md#automatic-cuda-host-draft-model-selection).

[See granular commands without the wrapper orchestration](quickstart-pdf-corpus-commands.md)
