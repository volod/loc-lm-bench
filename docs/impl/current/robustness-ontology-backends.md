# Robust Backends And Ontology Drafting

This page covers the implementation pieces that make local serving reliable on constrained GPUs and
make ontology-assisted data drafting reusable by GraphRAG.

## Memory Planner

`src/llb/backends/planner.py` estimates whether a model can run on the host. The estimate is
architecture-aware because simple `params * bpw` math is wrong for partial quantization.

Key logic:

- `weights_mib_detailed` prices high-precision embedding/norm mass separately from quantized
  linear weights;
- `hi_precision_params` derives that mass from vocab and hidden size when possible;
- `arch_from_config` reads cached Hugging Face `config.json` files without downloading;
- `enrich_arch` fills or, when explicitly requested, overrides curated architecture fields;
- sliding-window KV helpers keep Gemma-style long context estimates from assuming full attention in
  every layer.

The rationale is practical fit prediction. A quantized checkpoint can still carry a large
high-precision embedding table, and vLLM startup failures are expensive compared with a conservative
pre-flight estimate.

## Model Preparation Contracts

`src/llb/backends/prepare.py` expands `ModelSpec.sources` into backend-specific `PreparedModel`
rows using the `SourceRecord` metadata from `src/llb/contracts.py`, matching resolver source
normalization. `prep-models` and `prep-serving-targets` progress callbacks receive those typed
rows before backend dispatch. `make ci` covers formatting, ruff, mypy, and non-slow pytest for this
path.

`src/llb/backends/hardware.py` detects CUDA hosts through `nvidia-smi`. Detection first tries the
resolved executable from `PATH`, then falls back to common absolute locations such as
`/usr/bin/nvidia-smi`, so planner commands still see the GPU when an execution environment has a
minimal `PATH`. On the local RTX 4060 Ti host, `detect-gpu-vram`, `list-models`, and
`resolve-models --offline` now report a 16,380 MiB GPU tier.

## VRAM Contention Guard

`src/llb/executor/contention.py` protects vLLM launches from resident GPU users.

Before launching a VRAM-owning backend, `run-eval` can:

- read free and total VRAM;
- lower `gpu_memory_utilization` to a safe value;
- abort before startup when the remaining headroom cannot hold weights, overhead, and KV;
- unload resident Ollama models with `--evict`;
- wait for free VRAM with `--wait`.

The default behavior is non-destructive derating. The guard records a `ContentionReport` in the
manifest so a surprising context cap or launch abort is explainable after the run.

## llama.cpp Backend

`src/llb/backends/llamacpp.py` implements `LlamaCppLauncher` behind the same `BackendLauncher`
protocol as Ollama and vLLM.

It builds `llama-server` argv from either a local GGUF path or an HF GGUF source, controls context
size, and sets `-ngl` for GPU/CPU layer split. Readiness is checked through `/health`; served
context is parsed from `/props` across known llama.cpp response shapes.

```bash
make build-llamacpp
llb run-eval --backend llamacpp --model <gguf-source> --gpu-layers -1
```

The resolver can derive a partial offload split for oversized GGUF candidates. Direct `run-eval`
still honors an explicit `--gpu-layers` override.

## vLLM Serving Preflight

`src/llb/backends/preflight.py` probes the flashinfer sampler and stores a host verdict under
`$DATA_DIR/llb/preflight/vllm_sampler.json`. The verdict includes the driver, sampler choice, and
flashinfer version when known.

```bash
llb preflight-vllm
llb preflight-vllm --force
llb preflight-vllm --auto-pin
```

`--auto-pin` is opt-in because it changes the Python environment. The launcher records the sampler
actually used in telemetry.

## Ontology-Assisted Drafting

`src/llb/prep/ontology/` drafts unverified gold-set bundles from a corpus. It is data preparation,
not a runtime retrieval backend.

Pipeline stages:

1. `inventory.py`: read `.md` and `.txt` files, keep corpus-relative ids, hashes, and section
   boundaries.
2. `extract.py`: extract entities, aliases, claims, events, and subject-relation-object facts with
   grounded evidence spans.
3. `entity_types.py`: normalize to the closed 13-type vocabulary used by the graph schema.
4. `induce.py`: aggregate extracted types and relations into an ontology candidate.
5. `coverage.py`: select coverage-first seeds across entity, relation, section, and difficulty
   strata.
6. `draft.py`: ask for Ukrainian questions and answers around bounded evidence windows.
7. `refine.py` and `pipeline.py`: re-ground, reject circular items, deduplicate, split, and emit
   the bundle.

```bash
make prepare-goldset-draft DRAFT_CORPUS=<dir> DRAFT_MODEL=<local-model> DRAFT_NO_THINK=1
make prepare-goldset-draft DRAFT_CORPUS=<dir> DRAFT_MODEL=<local-model> \
  DRAFT_DOC_LIMIT=1 DRAFT_EXTRACT_MAX_CHARS=12000 DRAFT_VERIFY_N=30
llb prepare-goldset-draft --corpus-root <dir> --model <local-model> \
  --max-tokens 2048 --temperature 0 --timeout 300 --no-think \
  --doc-limit 1 --extract-max-chars 12000 --verification-sample-size 30
llb prepare-goldset-draft --corpus-root <dir> --model <model> --extractor spacy
```

Outputs land under `$DATA_DIR/prepare-goldset/<timestamp>/` unless `--out-dir` is supplied:

```text
goldset.jsonl
corpus/
ontology.json
extraction.jsonl
provenance.json
pdf_ontology_report.json
prompt_dictionary_candidates.jsonl
needle_items.jsonl
```

For PDF-derived corpora, `pipeline.py` copies matching PDF citation sidecars into the bundle and
`artifacts.py` writes the calibration report, source-backed prompt dictionary candidates, and
citation-valid needle items. The report records the bounded-probe settings (`doc_limit`,
`extract_max_chars`, `max_items`, seed), elapsed time, parse rate, page-span coverage, grounded
entity/event/claim/fact counts, dictionary-term yield, and quality gates. The gates broadened past
"nonzero SRO facts": grounding counts if the extraction produced evidence of ANY kind
(`nonzero_grounded_extractions`), the gold set must be non-empty (`nonzero_draft_items`), and for
PDF corpora at least one citation-valid needle must exist (`has_citation_valid_needles`, marked
applicable by `pdf_citation_gate_applicable`). A single `passed` roll-up ANDs the required gates
(the needle gate only when page sidecars exist); `nonzero_grounded_facts` stays informational since
SRO relations power the GraphRAG store but no longer solely block a fact-sparse corpus. The
pipeline logs the roll-up (WARNING when it fails), and a failing gate is never fatal -- the bundle
is always written for inspection and the human verification gate remains the real block on scoring.

Every emitted gold item remains `verified=false`. The bundle must pass cross-check and human
verification before it can score real models.

## spaCy Adapter And Long Documents

`src/llb/prep/ontology/spacy_adapter.py` implements the Python-native NER adapter over spaCy
`uk_core_news` models. It is opt-in and lazy-imported. The adapter maps labels through the same
closed vocabulary as LLM extraction.

`LLMExtractionAdapter` chunks over-long documents, extracts per window, and merges entities and
facts while grounding offsets against the full original document. This keeps later sections from
being silently truncated by endpoint context limits.

For long local PDF corpora, extraction logs document and window progress. Ollama reasoning models
should use `--no-think`; the command routes through Ollama native `/api/chat` so `think=false` is
honored and JSON extraction is not spent on hidden reasoning.

`--num-ctx` (make: `DRAFT_NUM_CTX`) right-sizes the Ollama context window for drafting through the
same native endpoint. Without it, Ollama loads the model with its modelfile context (often 128k+),
which forces CPU offload on VRAM-bound hosts even though drafting prompts are bounded by
`extract_max_chars`. Measured on the 16 GB RTX 4060 Ti host with `batiai/qwen3.6-35b:iq3`:
the default context loaded 19 percent CPU / 81 percent GPU at about 120 s per extraction window;
`--num-ctx 16384` loaded 4 percent CPU / 96 percent GPU at about 43 s per window. Keep headroom
over `extract_max_chars` plus the completion budget -- Ollama silently truncates prompts longer
than `num_ctx`. The quickstart PDF flow passes `QUICKSTART_DRAFT_NUM_CTX` (default 16384). The
chosen value lands in bundle provenance (`endpoint.num_ctx`).

The one-document probe path is the default safety valve before a multi-hour full PDF run:
`DRAFT_DOC_LIMIT=1` bounds documents and `DRAFT_EXTRACT_MAX_CHARS=<n>` bounds each extraction
window. Increase those only after `pdf_ontology_report.json` shows `gates.passed` (grounded
extractions, a non-empty gold set, and citation-valid needles) and usable citation coverage.
