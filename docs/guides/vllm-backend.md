# Real backend: vLLM + telemetry (backend telemetry)

RAG core runs the loop on prebuilt Ollama. backend telemetry adds a **vLLM** launcher (serves
HF weights behind the same OpenAI-compatible interface) plus a real telemetry hook. This is
the heavy, GPU-host path. Prebuilt packages install through uv; an explicit local-checkout
mode handles CUDA source builds. Model weights are multi-GB. Module detail is in
[implementation/current.md](../impl/current.md).

## 1. Install vLLM (once, GPU host)

    make venv # installs prebuilt vLLM wheels automatically on CUDA hosts

Use `VENV_INSTALL_VLLM=0 make venv` for a lean environment without vLLM. To reinstall vLLM directly
or override the registry version, use the lower-level target:

    make build-vllm # binary-only install through uv's shared cache

The default path runs `uv pip install --only-binary :all:`. vLLM and all dependencies stay
in uv's standard shared cache (see `uv cache dir`) and are reusable by other uv projects.
Nothing is copied to `$DATA_DIR/wheels`. Override the registry version with:

    VLLM_SPEC='vllm==0.6.3' make build-vllm

To build an unpublished fork, clone it first and point the script at the clean checkout:

    git clone <repo-url> ../vllm
    VLLM_SOURCE_DIR=../vllm make build-vllm

Source mode installs build/runtime dependencies through uv, applies the canonical
`MAX_JOBS` cap, and exports only the locally built vLLM wheel under
`$DATA_DIR/wheels/vllm_<python+torch+cuda+gpu-arch>_git<revision>/`. The checkout must be
clean and point to its git root so the cache key identifies the exact source. `VLLM_SPEC=git+...`
and implicit local paths are rejected; use `VLLM_SOURCE_DIR` so source builds cannot be
confused with ordinary installs. The shell command is a thin bootstrap; `llb.build.vllm`
owns the Python implementation.

## 2. Cache weights + verify the model id

    make prep-models PREP_BACKEND=vllm # snapshot-downloads HF weights; a
    wrong/gated id is reported

`prep-models` is also the **id verification** step: a wrong repo id 404s and a gated repo
needs `HF_TOKEN` in `.env` (it is reported per-model, not fatal).

## 3. Run on the real backend, with telemetry

    make build-index                                  # if not already built
    llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry # the
    real-model validation reference run

or pick the model directly (cap the context so the KV cache fits -- see Gotchas):

    make run-eval BACKEND=vllm MODEL=google/gemma-4-E4B-it-qat-w4a16-ct
    TELEMETRY=1

The launcher starts `vllm serve <model>` (controlling `--gpu-memory-utilization` and
`--max-model-len`, recorded for VRAM comparability), waits for readiness, runs the eval, then
kills the server. `--telemetry` adds a steady-state pass and records into the manifest:
**tokens/sec** (fixed prompt set + warmup), **cold-start load time** (separate from
throughput), **peak VRAM** (NVML), **requested vs served context**, and **tokenizer
efficiency** (tokens per UA char). vLLM logs land under
`$DATA_DIR/run-eval/<UTC timestamp>-<run id>/vllm/`; if the engine dies during startup the
log is preserved to `$DATA_DIR/llb/logs/failed-*.log` (the run bundle is discarded).

Validated (real-model validation, RTX 4060 Ti 16 GB, vLLM 0.23.0): `gemma-4-E4B-it-qat-w4a16-ct`
scored 0.801 objective at **63.8 tok/s**, peak VRAM **15.7 GB** (gpu-mem-util 0.80),
cold load **112 s**, served ctx 8192.

The same launcher can serve ontology-assisted PDF drafting:

    make prepare-goldset-draft DRAFT_CORPUS=<dir> DRAFT_MODEL=<hf-vllm-model> \
      DRAFT_BACKEND=vllm DRAFT_NO_THINK=1 DRAFT_NUM_CTX=16384

When `DRAFT_BASE_URL` is unset, the draft command starts and stops `vllm serve`; when it is set,
the command uses the existing OpenAI-compatible endpoint. `DRAFT_NO_THINK=1` sends vLLM
`chat_template_kwargs.enable_thinking=false` through the request `extra_body` for reasoning models.

## Gotchas (from the real-model validation run)

- **flashinfer sampler is defaulted off.** vLLM JIT-compiles a flashinfer sampling kernel at
  startup; flashinfer 0.6.x's `sampling.cuh` calls `cub::BlockAdjacentDifference::FlagHeads`,
  removed from newer CCCL/CUB, so the build fails on consumer GPUs (sm_89). The launcher sets
  `VLLM_USE_FLASHINFER_SAMPLER=0` (greedy decoding does not need it); export
  `VLLM_USE_FLASHINFER_SAMPLER=1` to opt back in where the kernel builds.
- **Cap `max_model_len`.** A model's native window (e.g. 131072) makes vLLM over-reserve the
  KV cache and fail startup on 16 GB. The sample config caps it to 8192.
- **Free VRAM first.** vLLM's startup check needs `gpu-memory-utilization x total` VRAM free.
  A resident Ollama model (it keeps weights ~5 min) can fail the launch; unload it
  (`curl -s localhost:11434/api/generate -d '{"model":"<tag>","keep_alive":0}'`) or lower
  `gpu_memory_utilization`. A pre-launch guard is planned (plan.md robust backend prep).

## Config

vLLM serving knobs live in `RunConfig` (set in YAML or via flags): `backend: vllm`,
`vllm_host`, `vllm_port`, `gpu_memory_utilization`, `max_model_len`, `dtype`, `quantization`,
`measure_telemetry`. Use `make list-models` first to confirm a model fits this host.

## What needs your GPU

The launcher, telemetry, and build helper are built + unit-tested with fakes and now
**validated on a real model** (real-model validation above). The from-source build path
still runs only on a CUDA host. The real-model validation fit correction is already fed
back into `samples/models_uk.yaml`: w4a16 weights are under-estimated by `list-models`
(`params_b x bpw` ignores the high-precision large-vocab embedding -- measured 9.8 GiB
vs predicted ~4.2 GiB). An embedding-aware estimate is forward work
(plan.md robust backend prep).
