# Inference config examples

Serving settings for MamayLM, Lapa, Gemma 4, and Qwen3.6 on **12 / 16 / 24 / 32 GiB**
GPU tiers. Primary target ids are model families; each tier selects the largest concrete model
variant that fits. Configurations maximize model size and explicit context on GPU, not throughput.
loc-lm-bench scores **text only**.

## Generate configs for your machine

Templates and tier tables live in [samples/config-example/](../../samples/config-example/).
Generated artifacts are written under `.data/llb/serving/gpu-<tier>gb/` (gitignored).

### 1. Detect GPU tier

```bash
make detect-gpu-vram
# or: scripts/detect_gpu_vram.sh
# or: llb detect-gpu-vram
```

Maps `nvidia-smi` total VRAM to a supported tier: **12, 16, 24, or 32 GiB**
(16380 MiB -> 16 GiB, 32607 MiB -> 32 GiB, etc.).

### 2. Generate serve + run-eval artifacts

```bash
make gen-serving-config              # detect tier, render into .data/
make gen-serving-config GPU_GB=32     # override tier without re-detecting
# or: scripts/gen_serving_config.sh [12|16|24|32]
# or: llb gen-serving-config [--gpu-gb N]
```

Output directory (example for 16 GiB):

```text
.data/llb/serving/gpu-16gb/
  tier.json                  # index of generated files + models
  serve_mamaylm.sh           # start serving (ollama or vllm)
  serve_lapa.sh
  serve_gemma_4.sh
  serve_qwen3.6.sh
  run_eval_<target>.yaml     # llb RunConfig
  run_eval_<target>.sh       # llb run-eval --config ... --telemetry
  serve_gemma_4_12b_vllm.sh  # extra largest vLLM quant on this tier (when defined)
```

Read `tier.json` for the exact script names on your tier.

### 3. Serve and benchmark

From the repo root, after `make build-index` and `make prep-models` as needed:

```bash
.data/llb/serving/gpu-<tier>gb/serve_mamaylm.sh          # blocking serve
.data/llb/serving/gpu-<tier>gb/run_eval_mamaylm.sh       # eval + telemetry
```

Unload resident Ollama models before starting vLLM (`keep_alive: 0`; see operational
notes below).

## Host profiles (documented machines)

| Host | GPU tier | System RAM | Notes |
| ---- | -------- | ---------- | ----- |
| Dev / benchmark | 16 GiB | 128 GiB | RTX 4060 Ti; real-model validation vLLM reference (E4B w4a16) |
| HP Z2 Tower | 32 GiB | 64 GiB | RTX 5090; vLLM for MamayLM + Gemma 31B |

Use `make gen-serving-config` on either machine; override with `GPU_GB=` when testing
another tier.

---

## Target models

| HF repo | Chat variant | License |
| ------- | ------------ | ------- |
| [MamayLM 27B FP8][mamay-fp8] | instruct | Gemma Terms |
| [Lapa v0.1.2 Instruct][lapa] | instruct | Gemma Terms |
| [google/gemma-4-31B][gemma-31b] | **`gemma-4-31B-it`** | Apache 2.0 |
| [Qwen3.6 35B-A3B][qwen36] | same repo | Apache 2.0 |

[mamay-fp8]: https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-FP8-dynamic
[lapa]: https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct
[gemma-31b]: https://huggingface.co/google/gemma-4-31B
[qwen36]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B

### Traps

- **MamayLM FP8:** never pass `--quantization fp8` to vLLM (checkpoint is pre-quantized).
- **Gemma 4:** the family target serves the largest tier fit; when the tier picks 31B, serve
  **`gemma-4-31B-it`**, not the base checkpoint.
- **Qwen3.6 MoE:** vLLM loads all expert weights (~35B stored); Ollama GGUF on smaller GPUs.
- **Thinking mode:** Qwen3.6 defaults to thinking output; generated run configs use
  `temperature: 0.0` for reproducible scoring.

---

## Tier fit summary (from manifest)

Largest backend + model per target. Details and vLLM knobs:
[samples/config-example/manifest.yaml](../../samples/config-example/manifest.yaml).

| Tier | MamayLM | Lapa | Gemma 4 family target | Qwen3.6 35B-A3B | Extra vLLM on tier |
| ---- | ------- | ---- | -------------- | --------------- | ------------------ |
| 12 GiB | Ollama Q4_K_M GGUF | Ollama Q4_K_M GGUF | 31B Ollama Q4_0 GGUF | Ollama `iq3` | E4B w4a16 (util 0.80, ctx 8192) |
| 16 GiB | Ollama Q4_K_M GGUF | Ollama Q4_K_M GGUF | 31B Ollama Q4_0 GGUF | Ollama `iq3` | 12B w4a16 (util 0.85, ctx 8192) |
| 24 GiB | Ollama Q4_K_M GGUF | Ollama Q4_K_M GGUF | 31B vLLM w4a16 (0.90, 16384) | Ollama `iq4` | -- |
| 32 GiB | vLLM FP8 (0.90, 8192) | vLLM bf16 (0.90, 8192) | 31B vLLM w4a16 (0.90, 16384) | Ollama `iq4` | -- |

Qwen3.6 FP8 (~33 GiB weights) does not fit any tier through vLLM; use Ollama or 48 GiB+ GPU.

---

## vLLM memory tuning

**Goal:** largest checkpoint + explicit context without startup OOM.

vLLM keeps **weights on GPU** (no CPU offload). One pool sized by
`gpu_memory_utilization` holds weights, KV cache (from `max_model_len`), and workspace.

| GPU VRAM | Typical `gpu_memory_utilization` |
| -------- | ------------------------------- |
| 12 GiB | 0.80 |
| 16 GiB | 0.80-0.85 |
| 24 GiB | 0.88-0.90 |
| 32 GiB | 0.90-0.92 |

Generated vLLM serve scripts also set `--kv-cache-dtype fp8`, `--max-num-seqs 1`, and
text-only `--limit-mm-per-prompt`. **Always use explicit `--max-model-len`** from the
manifest (never `auto` in generated configs).

**If startup OOMs:** halve `max_model_len` in
[samples/config-example/manifest.yaml](../../samples/config-example/manifest.yaml), re-run
`make gen-serving-config`, and retry.

### Ollama vs vLLM

| Backend | Weights | When used on a tier |
| ------- | ------- | ------------------- |
| vLLM | GPU only | Largest quant that fits VRAM (24G+ Gemma, 32G MamayLM) |
| Ollama | GPU + RAM offload | Full-size targets on 12-24 GiB; Qwen3.6 on all tiers |

---

## Common setup

```bash
make build-vllm                        # once, GPU host
make prep-models PREP_BACKEND=vllm     # cache HF weights
make build-index                       # before run-eval
make gen-serving-config                # artifacts -> .data/llb/serving/
```

Environment: copy [`.env.example`](../../.env.example) to `.env` (or run `make venv`).
Variable names are defined in [`src/llb/env.py`](../../src/llb/env.py). Set `HF_TOKEN` for gated
MamayLM/Gemma weights, `VLLM_HOST` if vLLM is not on port 8000, and keep
`VLLM_USE_FLASHINFER_SAMPLER=0` on consumer GPUs (generated vLLM scripts read this). See
[vLLM backend guide](../guides/vllm-backend.md).

### Operational notes

1. **Free VRAM before vLLM** -- Ollama `keep_alive: 0` or stop Ollama.
2. **One large model at a time** on 64 GiB RAM hosts (e.g. do not keep `llama3.3:70b`
   resident while launching vLLM).
3. **Blackwell (RTX 50xx):** vLLM build needs CUDA 12.8+ / sm_120 support.

---

## Documented hosts (reference)

### 16 GiB GPU / 128 GiB RAM (dev machine)

RTX 4060 Ti, 16380 MiB, sm 89. Full-size target repos need **Ollama**; largest vLLM
quant on this tier is Gemma 4 12B w4a16 (manifest extra entry). real-model validation validated E4B
w4a16 at util **0.80**, ctx **8192**
([run_config_vllm_uk.yaml](../../samples/run_config_vllm_uk.yaml)).

### 32 GiB GPU / 64 GiB RAM (HP Z2 Tower)

RTX 5090, 32607 MiB, sm 120. **vLLM** for MamayLM 27B FP8 and Gemma 4 31B w4a16;
**vLLM** for Lapa v0.1.2 Instruct; **Ollama `iq4`** for Qwen3.6 (same class as installed
`qwen3:30b` at 18 GiB).

---

## Related

- [samples/config-example/](../../samples/config-example/) -- manifest + templates
- [vLLM backend guide](../guides/vllm-backend.md)
- [samples/models_uk.yaml](../../samples/models_uk.yaml) -- planner registry
