# Host Validation

Host validation is the repeatable checklist for a CUDA workstation. It complements CI, which avoids
network, model downloads, and GPU-dependent paths.

## Core RAG Path

```bash
make validate-goldset
make build-index
make validate-retrieval RAG_K=10
make run-eval MODEL=<fitting-ua-model> BACKEND=ollama LIMIT=20 TELEMETRY=1
```

Expected properties:

- the committed fixture validates;
- retrieval clears the configured recall gate;
- `run-eval` writes a manifest and per-case scores;
- telemetry records throughput and peak VRAM when NVML is reachable;
- MLflow mirroring does not replace the canonical bundle.

## Backend Paths

Run one small cell for each backend available on the host:

```bash
llb run-eval --backend ollama --model <ollama-tag> --telemetry --limit 20
llb run-eval --backend vllm --model <hf-repo> --telemetry \
  --max-model-len 8192 --gpu-memory-utilization 0.80 --evict --limit 20
llb run-eval --backend llamacpp --model <gguf-source> --telemetry \
  --max-model-len 8192 --gpu-layers -1 --limit 20
```

Check that each backend records the same manifest shape. For vLLM, inspect contention and sampler
fields. For llama.cpp, inspect served context and `n_gpu_layers`.

On 12 GiB CUDA hosts, pin embeddings to CPU before a vLLM probe so the embedder does not compete
with the served model for the last few hundred MiB. Use the generated config so the offloaded 12B
target carries its `cpu_offload_gb` and `kv_offloading_size_gb` settings into `run-eval`:

```bash
make gen-serving-config
LLB_EMBED_DEVICE=cpu llb run-eval \
  --config "$DATA_DIR/llb/serving/gpu-12gb/run_eval_gemma_4_12b_vllm.yaml" \
  --evict --limit 1
```

## Robust Backend Checks

```bash
llb list-models --trust-config
llb preflight-vllm --force
llb detect-gpu-vram
llb gen-serving-config
```

When testing VRAM contention, prefer `--evict` or `--wait` before manual process intervention. The
contention guard should abort before launching a doomed vLLM server when headroom is insufficient.

## RTX PRO 3000 Blackwell 12 GiB Acceptance

The 2026-07-28 acceptance run used an NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU
(12,227 MiB, compute capability 12.0), driver 610.43.02, PyTorch 2.11.0+cu130, and vLLM 0.24.0.
`make detect-gpu-vram` selected tier 12, and `make gen-serving-config` persisted the detected GPU
identity and memory rather than an override-only tier.

The host run exposed and fixed one shared configuration gap. The generated serving YAML and the
`validate-retrieval` / `run-eval` CLI plus make paths now carry `corpus_root`; before that change,
the published gold set could be paired with the unrelated default corpus and a freshly rebuilt
store immediately read as stale. Regression coverage lives in
`tests/llb/inference/test_inference_generate.py`,
`tests/llb/rag/test_validate_retrieval_cli.py`, and
`tests/llb/eval/test_run_eval_cli.py`.

Acceptance results:

- The 250-item published gold set passed validation. A CPU-pinned e5-base rebuild wrote 311 chunks,
  and `make validate-retrieval RAG_K=10` scored the 82-item final split at recall@10 0.976 and MRR
  0.838.
- The generated Gemma 4 12B vLLM config ran one item with embeddings on CPU. The contention guard
  accepted 0.90 utilization with 11,696 MiB free; native sampling, Triton attention, Marlin W4A16,
  16 GiB CPU weight offload, and 32 GiB KV offload served a 16,384-token context at 3.32 tok/s and
  11,511 MiB peak VRAM. The load took 246.07 seconds, chiefly CUDA-graph capture. FlashInfer 0.6.12
  could not supply its sampler on SM 12.0 and the recorded native-sampler fallback worked. Artifact:
  `$DATA_DIR/run-eval/20260728T065519.474285Z-2f08bcd131d7/`.
- The 20-item Ollama path used the Ukrainian MamayLM Gemma 3 12B Q4_K_M model with CUDA embeddings.
  It scored objective 0.406, reliability 1.0, retrieval recall@5 0.900 / MRR 0.787, 39.16 tok/s,
  and 9,932 MiB peak VRAM. Artifact:
  `$DATA_DIR/run-eval/20260728T075902.053333Z-7e94edc3fe16/`.
- llama.cpp was not an available backend on this host (`llama-server` was absent), so no llama.cpp
  cell was claimed.
- The repository gate selects only current implementation coverage: obsolete unpublished-artifact
  compatibility checks were removed rather than skipped. It passes 2,226 tests with 43
  opt-in/slow tests deselected and zero runtime skips. Ruff format/check, mypy, Markdown lint, and
  the code-quality report also passed. `ollama ps` was empty after the evidence runs.

The recent paired embedder, context-ablation, and local drafting evidence reruns are recorded in
[RAG core](rag-core.md#the-recommendation-re-read-with-paired-uncertainty),
[RAG core](rag-core.md#context-ablation-evidence), and
[data prep](data-prep.md#sequential-local-qwengemma-draft-comparison).

## Category Smoke Path

Run representative category commands with committed samples:

```bash
llb bench-security --model <model> --backend <backend>
llb bench-tooling --model <model> --backend <backend>
llb bench-agentic --model <model> --backend <backend>
llb bench-summarization --model <model> --backend <backend>
llb bench-structured --model <model> --backend <backend>
llb bench-text-analysis --bundle samples/text_analysis_bundle_uk \
  --model <model> --backend <backend>
```

Each category should write a tier-specific manifest and per-case score series under
`$DATA_DIR/<category>/<run>/`.

## Judge Path

```bash
llb judge-smoke --judge-model <judge> --judge-base-url <url>
make calibration-score
```

Use the smoke check before long judged category or RAG runs. Use the calibration score to decide
whether `JUDGE_RHO` is admissible for the run.

## Platform Matrix

```bash
make platform-matrix
```

Use this only after the individual backend paths are known to work. The matrix compares backend
serve paths for a common logical model base, not arbitrary unrelated checkpoints.

## Quality Gate

Run the repository checks after host-specific validation:

```bash
make ci
make lint-md
scripts/code_quality.sh
```

`scripts/code_quality.sh` always prints the largest tracked Python files and largest tracked
non-Python files. Root-file, markdown, shell, and complexity sections are quiet when clean and
appear only when they have findings, missing optional tools, or failures.

`make test` is the full local precommit flow when slow tests are acceptable.

### Code quality checks

`make ci` checks Ruff formatting and lint, mypy, and the non-slow pytest suite. `make test` adds
the full local test flow and Markdown lint. `scripts/code_quality.sh` reports long source files,
cyclomatic complexity, and cognitive complexity so maintainers can split code at functional seams.
The ~250-line source-file target is soft; cohesive schemas and regular lookup families may remain
whole.

The D-grade cyclomatic-complexity cleanup keeps orchestration separate from validation, state
accumulation, and presentation. Ontology dedup now uses an embedded-candidate value object and
named matching/report helpers; the multi-hop expansion audit uses a check accumulator that builds
the final report. Retrieval validation passes an immutable request into
`cli/rag/retrieval_validation.py`, autonomous verification scoring lives in
`auto_rag/verification_auto.py`, and query-prep dependency checks are table-driven. The query
robustness integration test uses a module-level morphology-loader callable and named assertion /
artifact phases. The repository-wide Radon D-or-worse scan is empty; focused coverage lives in the
ontology, auto-RAG verification, query-prep, and query-robustness test suites.

The cognitive-complexity cleanup extends that separation across backend planning, review
workflows, conflict resolution and filtering, query robustness, incremental refresh, ontology
expansion, retrieval fusion, and reporting. Complex branches now live in named policy,
validation, selection, and rendering helpers. Stateful assembly uses
`rag/refresh/merge_assembly.py`; tree leaf filtering, robustness recovery, and hybrid-store
retrieval use focused owner modules. Launcher and morphology closures are module-level callable
adapters. The repository-wide Complexipy scan at the configured maximum of 15 is empty, and the
refactored source files remain within the 250-line soft target. Focused verification covers the
affected backend, review, conflict, evaluation, graph, ontology, retrieval, and scoring paths.
