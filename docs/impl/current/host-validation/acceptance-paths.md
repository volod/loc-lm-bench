# Host Acceptance Paths

The cells to run on a CUDA workstation, in the order they are worth running: the core RAG
path first, then one cell per backend the host has, then the category, judge, and
cross-platform checks. Each states the properties to expect rather than a pass/fail number.

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

The `0.80` above is the value this card class is validated for, and it is a SEARCH input too, not
only an eval one. Both `joint-search` and `joint-search-long-run` take `--config`,
`--gpu-memory-utilization`, and `--max-model-len`, and carry them into every screen cell, every
finalist tune, and the confirmation run's public screen:

```bash
llb joint-search --candidates <vllm-candidates-yaml> --config <run-config-yaml> \
  --gpu-memory-utilization 0.80 --max-model-len 8192 --limit 20
```

Without them a search served every vLLM candidate at the `RunConfig` default 0.85, and the
pre-launch guard cannot correct that: it derates against OTHER processes' memory, so on a quiet
card the request reads as free. See
[the serving knobs a search carries](../rigor-board-judge/tuning-and-search.md#serving-knobs-a-search-carries).

On this 16 GiB host the embedder is what the guard actually trips over first when it shares the
process with a vLLM launch: with the embedder resident on the GPU the guard reported 12439 MB free
against a ~12609 MB need and ABORTED the launch, and the same command with `LLB_EMBED_DEVICE=cpu`
reported 14565 MB free and passed. Pin embeddings to CPU for any vLLM search cell on a 16 GiB card,
the same way the 12 GiB probe below does.

On 12 GiB CUDA hosts, pin embeddings to CPU before a vLLM probe so the embedder does not compete
with the served model for the last few hundred MiB. Use the generated config so the offloaded 12B
target carries its `cpu_offload_gb` and `kv_offloading_size_gb` settings into `run-eval`:

```bash
make gen-serving-config
LLB_EMBED_DEVICE=cpu llb run-eval \
  --config "$DATA_DIR/llb/serving/gpu-12gb/run_eval_gemma_4_12b_vllm.yaml" \
  --evict --limit 1
```

### A Failed Launch Names A Log That Still Exists

A launcher that owns a subprocess writes its server log inside the run's staging dir
(`$DATA_DIR/run-eval/.<timestamp>-<run id>.tmp/vllm/vllm-<port>.log`), which the failure path
deletes. Inside a screen cell or a tuning trial that staging dir is the ONLY copy, and nobody
reproduces the cell by re-running a command: the error named a path that had stopped existing
before anyone read it, so a failed Optuna trial recorded no cause. A FAILED launch now copies its
log to `$DATA_DIR/llb/logs/failed-<backend>-<port>-<UTC stamp>.log` before anything is torn down,
and the `RuntimeError` it raises names THAT path:

```text
RuntimeError: vLLM exited (code 1) during startup (startup log: <DATA_DIR>/llb/logs/failed-vllm-8000-<stamp>.log)
```

Preservation belongs to the launcher (`ServerLog`, `src/llb/backends/launch_log.py`), not to
`run-eval`, so it holds for every caller that starts one -- a screen cell, a tuning trial, the
drafting endpoint -- and for every backend that writes a startup log, vLLM and llama.cpp alike.
Each relaunch attempt keeps its own copy, because a relaunch truncates the log it reuses; the
`run-eval` teardown asks again while removing the staging dir and gets the same one copy back. A
SUCCESSFUL launch keeps nothing: the staging dir deleting a healthy cell's log is the temp dir
working as intended. When the launcher wrote no log at all (`log_dir` unset, stdout to `DEVNULL`)
the error names no path rather than inventing one, and when the copy itself fails it says the log
could not be preserved.

Probe it on a host by asking for a window the model cannot serve, which makes the engine exit
during startup config validation -- seconds, no weights loaded:

```bash
llb run-eval --backend vllm --model <hf-repo> --max-model-len 9000000 --limit 1
```

Verified on 2026-08-28 on the RTX 4060 Ti 16 GB CUDA host, vLLM 0.27.1, serving
`google/gemma-4-12B-it-qat-w4a16-ct` through `run_eval(..., emit=False)` -- the call a screen cell
and a tuning trial make, so the staging dir is created and deleted exactly as it is in a search.
The engine exited with code 1, the staging dir was gone when the call returned, and the single
7.3 KB preserved log named the cause on its last line: `User-specified max_model_len (9000000) is
greater than the derived max_model_len (max_position_embeddings=262144)`. The reading: a cell that
could not launch is now diagnosable from the run alone -- a bad sampled window reads as a config
error, a host flake as an allocation or driver error, without re-running hours of search to find
out which. What would overturn it: a launch that dies before the log handle opens, or a
`$DATA_DIR/llb/logs` that cannot be written -- both surface as an error that does NOT name a
readable log, which is the honest signal, not a silent loss.

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
