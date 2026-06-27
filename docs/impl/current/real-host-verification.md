# Real Host Verification

## Real-host verification (2026-06-25, RTX 4060 Ti 16 GB)

All three backends + every M5 category benchmark were validated on the real CUDA host (driver
595.71.05), complementing the fake-endpoint CI boards:
- **Core RAG eval:** `validate-goldset` PASS (250 items); `build-index` 311 chunks (dim 768);
  `validate-retrieval` recall@10=0.980, MRR=0.847 (> the 0.8 gate); `run-eval` llama3.2:3b on the
  final split with telemetry -> ~101 tok/s, peak VRAM 3994 MB, reliability 1.0, manifest + MLflow
  mirror written. The uncalibrated judge demoted correctly (objective ranks alone).
- **Three-backend run-path (same `run-eval` interface, source-built backends):**
  - **Ollama** (prebuilt daemon): llama3.2:3b -- ~101 tok/s, peak VRAM 3994 MB (above).
  - **vLLM 0.23.0** (`backend=vllm`, `samples/run_config_vllm_uk.yaml`): `google/gemma-4-E4B-it-qat-w4a16-ct`
    (w4a16, the M2.4 16 GB fit) -- the pre-launch VRAM guard fired (`gpu-memory-utilization 0.80`
    fits, 14990 MB free), then served at 63.5 tok/s, peak VRAM 14496 MB, cold load 114.0s, served
    ctx 8192; reliability 1.0. Ollama released the GPU cleanly first.
  - **llama.cpp** (`backend=llamacpp`, source-built `llama-server` under `.data/llb/llamacpp/build/bin`):
    `Qwen2.5-0.5B-Instruct` Q4_K_M GGUF (loaded via `-m`, all layers on GPU) -- 401.6 tok/s, peak
    VRAM 2318 MB, load 2.0s, served ctx 32768; reliability 1.0.
  All three stamped a ranked row + telemetry + manifest + MLflow mirror under the SAME executor.
- **M5.6 M4 run-path hardening (host-validated):**
  - **M4.1:** `list-models --trust-config` over the cached `config.json` of `gemma-4-E4B-it-w4a16`
    resolved it to `42/42 gpu` at ctx 131072 -- the sliding-window KV fits the full window where the
    old full-attention estimate forced an offload.
  - **M4.2:** the vLLM pre-launch guard read the free VRAM and the arch-derived KV headroom
    (`gpu-memory-utilization 0.80 fits, 15309 MB free`) before launching.
  - **M4.3:** `preflight-vllm --force` found the cached verdict STALE and re-probed -- the venv's
    flashinfer 0.2.5 builds + runs here (sampler=flashinfer, driver 595.71.05 recorded). A vLLM
    `--telemetry` run then recorded `sampler` + `flashinfer_version=0.2.5` in the manifest (the value
    is the sampler ACTUALLY used: `native`, because an explicit env flag won over the verdict).
  - **M4.5:** a real PARTIAL-offload split of the oversized `deepseek-r1:32b` Q4 GGUF (19.9 GB > 16
    GB) at `--gpu-layers 20` served end to end -- peak VRAM 7710 MB (only ~20 of ~64 layers on GPU,
    the rest in CPU RAM) at 3.36 tok/s, with `n_gpu_layers=20` recorded in the manifest. (Ollama's
    `gemma3:27b` GGUF would not load -- a llama.cpp-build arch-key mismatch, not a bug here.)
- **M5 category benches (llama3.2:3b, committed UA case sets):** agentic completion 0.500;
  security ASR 0.600 / defense 0.400 with a per-family breakdown; tooling call-accuracy 0.750;
  summarization reference-coverage 0.941 -- each stamped under its OWN Tier with a manifest + CI.
- **M5.3 gated trajectory-quality judge end-to-end:** with a real judge endpoint (gemma4:e4b on
  Ollama as the wiring smoke -- the calibrated production judge is gemma3:27b -- `--judge-rho
  0.628`) the judge was TRUSTED and recorded `trajectory_quality=0.5875` (CI [0.375, 0.85])
  ALONGSIDE the objective completion (0.500, unchanged), with a `JudgeStatus`
  (`metrics=["trajectory_quality"]`) persisted in the manifest -- confirming the objective-first
  "diagnostic alongside, never folded into the headline/ranking" contract on a real model.
- `make ci` (Ruff format + lint, mypy strict, lightweight pytest `-m "not slow"`) is green with
  zero warnings; after the M5 residuals it selects 615 tests (10 slow deselected; `make test` runs
  all 625 locally).

(Still host-pending: only the M5.6 host-dependent M4 hardening items the basic run did not exercise
-- Gemma sliding-window KV + cached-`config.json` arch override (M4.1), multi-GPU read + arch-derived
KV abort headroom (M4.2), flashinfer auto-pin + sampler-in-manifest (M4.3, the sampler stays off by
default), and a real PARTIAL-offload llama.cpp split (M4.5; only the all-on-GPU path is now confirmed)
-- see `plan.md`.)
