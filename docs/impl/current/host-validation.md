# Host Validation

Host validation is the repeatable checklist for a CUDA workstation. It complements CI, which avoids
network, model downloads, and GPU-dependent paths.

## Core RAG Path

```bash
make validate-goldset
make build-index
make validate-retrieval RAG_K=10
make run-eval MODEL=llama3.2:3b BACKEND=ollama LIMIT=20 TELEMETRY=1
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

### Complexity baseline (2026-07 refactor)

A repository-wide refactor brought every `src/` function to cyclomatic grade C or better and
cognitive complexity <= 15 (complexipy). The techniques used -- and expected for new code --
are: extract nested loop bodies and validation blocks into named `_helper` functions, replace
closures that capture many locals with module-level functions or small callable classes
(e.g. `_SubprocessCellRunner`, `_WallClockBudget`), group 10+-parameter builders behind
dataclasses (e.g. `DraftSettings`, `_LoopContext`, `_CampaignHooks`), and share near-identical
branches through one parameterized helper (e.g. `_ledger_ref_status` for flat vs chain
ledgers). Remaining known findings are two D-grade TEST functions
(`test_ontology_draft.py`, `test_inference_generate.py`) and maintainability-index C grades on
the largest files (`goldset/verify.py`, `prep/pdf_corpus.py`, `scoring/external_rag.py`,
`goldset/verify_session.py`), which are size-driven and would need module splits to move.
