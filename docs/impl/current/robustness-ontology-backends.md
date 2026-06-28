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
llb prepare-goldset-draft --corpus-root <dir> --model <local-model>
llb prepare-goldset-draft --corpus-root <dir> --model <model> --extractor spacy
```

Outputs land under `$DATA_DIR/prepare-goldset/<timestamp>/` unless `--out-dir` is supplied:

```text
goldset.jsonl
corpus/
ontology.json
extraction.jsonl
provenance.json
```

Every emitted gold item remains `verified=false`. The bundle must pass cross-check and human
verification before it can score real models.

## spaCy Adapter And Long Documents

`src/llb/prep/ontology/spacy_adapter.py` implements the Python-native NER adapter over spaCy
`uk_core_news` models. It is opt-in and lazy-imported. The adapter maps labels through the same
closed vocabulary as LLM extraction.

`LLMExtractionAdapter` chunks over-long documents, extracts per window, and merges entities and
facts while grounding offsets against the full original document. This keeps later sections from
being silently truncated by endpoint context limits.
