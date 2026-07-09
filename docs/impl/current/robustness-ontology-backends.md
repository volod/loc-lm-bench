# Robust Backends And Ontology Drafting

This page covers the implementation pieces that make local serving reliable on constrained GPUs and
make ontology-assisted data drafting reusable by GraphRAG.

## CLI Exit Codes

`src/llb/core/runtime.py` `run_typer` drives the Typer app with `standalone_mode=False` so Ctrl-C is
not swallowed by click. In that mode click **returns** a `typer.Exit(code)` as an int rather than
raising it (`typer.Exit` IS `click.exceptions.Exit`), so `run_typer` reads the returned code and
re-raises it as `SystemExit`. Without that, every `raise typer.Exit(code=N)` in `src/llb/cli/`
exited 0 and a failing command looked successful to `make` and CI.

`--help` returns 0 through the same path and stays exit 0. `tests/test_runtime.py` pins both with a
`_ReturningApp` fake that models click's real non-standalone contract; the older `_FakeApp` (which
raises `Exit`) cannot express it.

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
5. `coverage.py`: select seeds across entity, relation, section, and difficulty strata --
   coverage-first up to the flat `max_items` cap, or (yield-max) up to a per-stratum
   `coverage_target`; `coverage_report` emits the "seeds remaining vs drafted" exhaustion matrix.
6. `draft.py`: ask for Ukrainian questions and answers around bounded evidence windows;
   `graph_paths.py` + `multi_hop.py` (yield-max) walk 2-hop graph chains and draft multi-span
   multi-hop questions.
7. `refine.py` and `pipeline.py`: re-ground, reject circular items, deduplicate, tag each item with
   a `question_type`/`difficulty` label (`question_types.py`), optionally drop near-duplicates of
   prior bundles (`dedup.py`), split, and emit the bundle.

```bash
make prepare-goldset-draft DRAFT_CORPUS=<dir> DRAFT_MODEL=<local-model> DRAFT_NO_THINK=1
make prepare-goldset-draft DRAFT_CORPUS=<dir> DRAFT_MODEL=<hf-vllm-model> \
  DRAFT_BACKEND=vllm DRAFT_NO_THINK=1 DRAFT_NUM_CTX=16384
make prepare-goldset-draft DRAFT_CORPUS=<dir> DRAFT_MODEL=<local-model> \
  DRAFT_DOC_LIMIT=1 DRAFT_EXTRACT_MAX_CHARS=12000 DRAFT_CONCURRENCY=2 DRAFT_VERIFY_N=30
llb prepare-goldset-draft --corpus-root <dir> --model <local-model> \
  --max-tokens 2048 --temperature 0 --timeout 300 --no-think \
  --doc-limit 1 --extract-max-chars 12000 --concurrency 2 --verification-sample-size 30
llb prepare-goldset-draft --corpus-root <dir> --model <hf-vllm-model> \
  --backend vllm --no-think --num-ctx 16384 --doc-limit 1
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
`extract_max_chars`, `extract_concurrency`, `max_items`, seed), elapsed time, parse rate, page-span
coverage, grounded entity/event/claim/fact counts, dictionary-term yield, and quality gates. The
gates broadened past
"nonzero SRO facts": grounding counts if the extraction produced evidence of ANY kind
(`nonzero_grounded_extractions`), the gold set must be non-empty (`nonzero_draft_items`), and for
PDF corpora at least one citation-valid needle must exist (`has_citation_valid_needles`, marked
applicable by `pdf_citation_gate_applicable`). A single `passed` roll-up ANDs the required gates
(the needle gate only when page sidecars exist); `nonzero_grounded_facts` stays informational since
SRO relations power the GraphRAG store but no longer solely block a fact-sparse corpus. The
pipeline logs the roll-up (WARNING when it fails). Plain `prepare-goldset-draft` still writes a
failing bundle for inspection, while the PDF and mixed-corpus quickstart wrappers pass
`--require-passed-gates` so a zero-item or ungrounded draft exits non-zero before graph/validation
steps.

Every emitted gold item remains `verified=false`. The bundle must pass cross-check and human
verification before it can score real models.

### Yield-Max Drafting

Three opt-in knobs maximize meaningful questions from a corpus instead of stopping at a flat item
cap. They compose (a run can set all three) and all stay deterministic and resumable (the journal
meta pins them).

- **Coverage-target sampling** (`--coverage-target N`, `DRAFT_COVERAGE_TARGET=N`). `select_seeds`
  drafts up to `N` seeds per stratum bucket (relation / entity type / section / semantic kind)
  rather than stopping at `--max-items`; `--max-items` remains a safety ceiling.
  `coverage.coverage_report` writes a `coverage_matrix` into `pdf_ontology_report.json` recording,
  per stratum dimension, how many buckets exist, how many were drafted (and reached the target), and
  how many candidate seeds remain -- so an operator sees whether a draft exhausted the corpus's
  breadth or was cut short.
- **Multi-hop chain questions** (`--multi-hop`, `DRAFT_MULTI_HOP=1`). `graph_paths.walk_two_hop_paths`
  walks directed `A -r1-> B -r2-> C` chains over the knowledge graph (built in-run by reusing the
  extraction, or loaded from a persisted store via `--graph-dir`/`DRAFT_GRAPH_DIR`).
  `multi_hop.build_multi_hop_items` grounds each chain in the two hops' exact evidence spans, so a
  multi-hop item carries at least two grounded spans across sections or documents and passes
  span-exact validation by construction. `--multi-hop-max-paths` caps the walk (default 40). Every
  multi-hop item is labeled `multi-hop` / hard.
- **Near-duplicate suppression** (`--dedup-against <bundle[,bundle]>`, `DRAFT_DEDUP_AGAINST=`).
  `dedup.NearDuplicateFilter` drops a drafted question whose pinned-E5 (`multilingual-e5-base`,
  the RAG store's embedder) cosine similarity to any prior-bundle question is `>= 0.9`, so a
  coverage-target rerun does not re-draft paraphrases a reviewer already saw. The embedder is
  injectable, so the filter is unit-tested with a fake embedder. `pdf_ontology_report.json` gains a
  `dedup` block (threshold, prior question count, dropped ids); `provenance.json` records the prior
  bundles.

Every drafted item is tagged with a closed **question type** (factoid, definition, procedural,
numeric, comparative, multi-hop) and a **difficulty** label, recorded in item provenance and on the
`needle_items.jsonl` rows (not the `GoldItem` schema). `pdf_ontology_report.json` records the
`question_type_distribution`, `difficulty_distribution`, and -- when a retrieval index is supplied --
the retrieval-unique needle fraction per question type
(`retrieval_unique_needle_fraction_by_question_type`), so reviewers and the miss analyzer can filter
and compare by question type.

```bash
make prepare-goldset-draft DRAFT_CORPUS=<dir> DRAFT_MODEL=<model> \
  DRAFT_COVERAGE_TARGET=6 DRAFT_MULTI_HOP=1 DRAFT_DEDUP_AGAINST=<prior-bundle>
llb prepare-goldset-draft --corpus-root <dir> --model <model> \
  --coverage-target 6 --multi-hop --multi-hop-max-paths 40 \
  --dedup-against <prior-bundle>,<other-bundle> --graph-dir <graph-store>
```

## spaCy Adapter And Long Documents

`src/llb/prep/ontology/spacy_adapter.py` implements the Python-native NER adapter over spaCy
`uk_core_news` models. It is opt-in and lazy-imported. The adapter maps labels through the same
closed vocabulary as LLM extraction.

`LLMExtractionAdapter` chunks over-long documents, extracts per window, and merges entities and
facts while grounding offsets against the full original document. This keeps later sections from
being silently truncated by endpoint context limits.

For long local PDF corpora, extraction logs document and window progress. LLM window extraction can
run concurrently within one document: use `DRAFT_CONCURRENCY=<n>` with
`make prepare-goldset-draft`, `QUICKSTART_DRAFT_CONCURRENCY=<n>` with the PDF quickstart, or
`llb prepare-goldset-draft --concurrency <n>` / `--extract-concurrency <n>`. The default is `1`,
which preserves the prior sequential behavior. Parallel extraction uses one bounded worker pool per
document and stores completed windows back into their original indexes before merge, so grounding
and deduplication stay deterministic. Bundle provenance and `pdf_ontology_report.json` record the
effective `extract_concurrency` setting.

Server-side parallelism is still the local server's job. For Ollama, size `DRAFT_CONCURRENCY` to the
available `OLLAMA_NUM_PARALLEL` slots so calls share one loaded model instead of starting a second
model instance. Use the one-document probe twice, first with `DRAFT_CONCURRENCY=1` and then with the
target concurrency, and compare `elapsed_s`, parse rate, and calibration gates before running the
full PDF draft.

Smoke probe evidence on the local 16 GB Ollama host used `llama3.2:3b`, one PDF-derived document,
`DRAFT_EXTRACT_MAX_CHARS=60000`, and two extraction windows. Sequential extraction
(`DRAFT_CONCURRENCY=1`) wrote `.data/prepare-goldset/parallel-probe-c1` with `elapsed_s=17.937`;
parallel extraction (`DRAFT_CONCURRENCY=2`) wrote `.data/prepare-goldset/parallel-probe-c2` with
`elapsed_s=14.559` for the same 32,074 prompt tokens. This is speed-only evidence: the small smoke
model returned no grounded JSON (`parse_rate=0.0`, gates failed), so use the production drafter
probe before accepting a real PDF bundle.

Ollama reasoning models should use `--no-think`; the command routes through Ollama native
`/api/chat` so `think=false` is honored and JSON extraction is not spent on hidden reasoning.

vLLM-backed drafting is still `--endpoint local` (no egress), but sets `--backend vllm`. If
`--base-url` is omitted, `src/llb/cli/prep.py` starts `VllmLauncher` from
`src/llb/backends/vllm.py`, waits for `/v1/models`, writes vLLM logs under the draft bundle's
`vllm/` directory, and points the endpoint at `http://localhost:<port>/v1`. If `--base-url` is set,
the command uses that already-running OpenAI-compatible server. `--num-ctx` maps to vLLM
`--max-model-len` only when the command launches the server; use `--vllm-max-model-len` to override
that explicitly. Bundle provenance records `endpoint.backend=vllm` and the served `base_url`.
`--no-think` sends vLLM request extras through the existing OpenAI client: `chat_template_kwargs`
with `enable_thinking=false`, plus `include_reasoning=false` and `reasoning_effort=none`, so
reasoning-model output budget is available for JSON.

Passing vLLM probe evidence on the local 16 GB RTX 4060 Ti host used
`google/gemma-4-E4B-it-qat-w4a16-ct`, `DRAFT_BACKEND=vllm`, `DRAFT_NO_THINK=1`,
`DRAFT_NUM_CTX=4096`, a one-document probe corpus at `.data/vllm-draft-probe-corpus`, and output
bundle `.data/prepare-goldset/vllm-draft-probe`. The run launched vLLM, served
`http://localhost:8000/v1`, wrote `.data/prepare-goldset/vllm-draft-probe/vllm/vllm-8000.log`,
and stopped the server. `pdf_ontology_report.json` recorded `elapsed_s=17.737`, `parse_rate=1.0`,
6 grounded entities, 3 grounded facts, 1 claim, 2 events, 2 kept draft items, and `gates.passed=true`.
`provenance.json` records `endpoint.backend=vllm`, `endpoint.base_url=http://localhost:8000/v1`,
`endpoint.think=false`, 3 local calls, and zero egress/cost.

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
