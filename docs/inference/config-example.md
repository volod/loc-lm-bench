# Inference config examples

Serving settings for MamayLM, Lapa, Gemma 4, Qwen3.6, and Mistral on **12 / 16 / 24 / 32 GiB**
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

### Automatic CUDA-host Draft Model Selection

`QUICKSTART_DRAFT_MODEL=auto` and `QUICKSTART_MODEL_SELECTION=auto` activate the PDF/mixed-corpus
drafter selector. This is deterministic host-profile selection, not a lookup of an old benchmark
result:

1. `detect-gpu-vram` reads total VRAM from `nvidia-smi` and selects the largest detected GPU.
2. Total VRAM maps to a supported tier: below 14 GiB -> 12, below 20 GiB -> 16, below 28 GiB ->
   24, otherwise 32. `QUICKSTART_GPU_GB=12|16|24|32` explicitly overrides detection.
3. The selector reads that tier from
   [`samples/config-example/manifest.yaml`](../../samples/config-example/manifest.yaml). It
   considers the `gemma-4` family target and any `gemma-4-*` extra targets.
4. For a CUDA tier, vLLM candidates rank ahead of Ollama/offload candidates. Within the same
   backend class, the larger parameter count wins.
5. A vLLM candidate is eligible only when its configured `max_model_len` meets
   `QUICKSTART_DRAFT_NUM_CTX` (16,384 by default). Ollama rows use the requested draft context at
   runtime and are not filtered by a manifest `max_model_len` field.
6. The selected row supplies the model, backend, GPU-memory fraction, maximum context, CPU weight
   offload, and CPU KV-offload settings to `prepare-goldset-draft`.

The current long-context automatic choices are:

| CUDA tier | Target | Model/backend | vLLM settings |
| ---: | --- | --- | --- |
| 12 GiB | `gemma-4-12b-vllm` | Gemma 4 12B w4a16 / vLLM | util 0.90, context 16384, CPU weights 16 GiB, CPU KV 32 GiB |
| 16 GiB | `gemma-4-12b-vllm` | Gemma 4 12B w4a16 / vLLM | util 0.85, context 16384, CPU weights 16 GiB, CPU KV 32 GiB |
| 24 GiB | `gemma-4` | Gemma 4 31B w4a16 / vLLM | util 0.90, context 16384 |
| 32 GiB | `gemma-4` | Gemma 4 31B w4a16 / vLLM | util 0.90, context 16384 |

If no GPU is detected, automatic selection does not prefer vLLM; it uses the 16 GiB manifest's
Ollama Gemma 4 row, which can offload through system RAM. Supplying `QUICKSTART_GPU_GB` deliberately
forces the corresponding CUDA profile even when detection is unavailable.

This `auto` mode chooses a curated, context-capable Gemma 4 drafter without spending hours on a
model comparison. Use `QUICKSTART_MODEL_SELECTION=benchmark` when the desired behavior is to run
the local candidate roster and select the best measured drafter for this host. Use `choose` for an
interactive local model, `frontier` for an explicitly authorized external route, or set
`QUICKSTART_DRAFT_MODEL=<model-id>` to bypass selection entirely.

Override precedence is explicit:

1. A concrete `QUICKSTART_DRAFT_MODEL` bypasses model selection.
2. `QUICKSTART_DRAFT_ENDPOINT=frontier` uses the external route and requires a concrete model.
3. Otherwise, `QUICKSTART_MODEL_SELECTION` chooses `auto`, `benchmark`, `choose`, or `frontier`.
4. Within `auto`, `QUICKSTART_GPU_GB` overrides detection and `QUICKSTART_DRAFT_NUM_CTX` filters
   vLLM candidates that cannot provide the requested context.

`auto` performs no model-choice prompt and does not depend on an existing benchmark artifact, so
it is safe in an unattended quickstart. The full corpus draft still has its deliberate compute and
data-egress confirmation gates; use `QUICKSTART_ASSUME_YES=1` in an already-approved unattended
run. That approval also covers a requested `benchmark` run and its recommended local model.

Automatic selection uses total VRAM and the curated manifest; it is not a live free-VRAM planner
or a performance measurement. Runtime serving checks can still reject a host whose memory is
occupied, and a selected vLLM profile requires the vLLM environment and model assets to be
prepared.

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
| [Mistral Small 3.1 24B][mistral] | instruct (w4a16 / FP8 / GGUF) | Apache 2.0 |

[mamay-fp8]: https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-FP8-dynamic
[lapa]: https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct
[gemma-31b]: https://huggingface.co/google/gemma-4-31B
[qwen36]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
[mistral]: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503

### Traps

- **MamayLM FP8:** never pass `--quantization fp8` to vLLM (checkpoint is pre-quantized).
- **Gemma 4:** the family target serves the largest tier fit; when the tier picks 31B, serve
  **`gemma-4-31B-it`**, not the base checkpoint.
- **Qwen3.6 MoE:** vLLM loads all expert weights (~35B stored); Ollama GGUF on smaller GPUs.
- **Thinking mode:** Qwen3.6 defaults to thinking output; generated run configs use
  `temperature: 0.0` for reproducible scoring.
- **Mistral quants:** both the 24 GiB `RedHatAI/...quantized.w4a16` and the 32 GiB
  `RedHatAI/...FP8-dynamic` checkpoints are compressed-tensors; vLLM auto-detects the quant, so
  never pass `--quantization`. Mistral Small 3.1 is multimodal, so the generated serve script keeps
  the text-only `--limit-mm-per-prompt '{"image": 0}'`.

---

## Tier fit summary (from manifest)

Largest backend + model per target. Details and vLLM knobs:
[samples/config-example/manifest.yaml](../../samples/config-example/manifest.yaml).

| Tier | MamayLM | Lapa | Gemma 4 family target | Qwen3.6 35B-A3B | Mistral Small 3.1 24B | Extra vLLM on tier |
| ---- | ------- | ---- | -------------- | --------------- | --------------------- | ------------------ |
| 12 GiB | Ollama Q4_K_M GGUF | Ollama Q4_K_M GGUF | 31B Ollama Q4_0 GGUF | Ollama `iq3` | Ollama Q4_K_M GGUF | 12B w4a16 (util 0.90, ctx 16384, CPU offload 16/32) |
| 16 GiB | Ollama Q4_K_M GGUF | Ollama Q4_K_M GGUF | 31B Ollama Q4_0 GGUF | Ollama `iq3` | Ollama Q4_K_M GGUF | 12B w4a16 (util 0.85, ctx 16384, CPU offload 16/32) |
| 24 GiB | Ollama Q4_K_M GGUF | Ollama Q4_K_M GGUF | 31B vLLM w4a16 (0.90, 16384) | Ollama `iq4` | vLLM w4a16 (0.90, 16384) | -- |
| 32 GiB | vLLM FP8 (0.90, 8192) | vLLM bf16 (0.90, 8192) | 31B vLLM w4a16 (0.90, 16384) | Ollama `iq4` | vLLM FP8 (0.90, 8192) | -- |

Qwen3.6 FP8 (~33 GiB weights) does not fit any tier through vLLM; use Ollama or 48 GiB+ GPU.
Mistral Small 3.1 24B serves vLLM on the 24 and 32 GiB tiers and Ollama (CPU offload) on 12/16 GiB.
The 24 GiB tier uses the w4a16 quant (~14 GiB weights, GPU-resident with KV room); the 32 GiB tier
upgrades to the higher-quality FP8 (~24 GiB weights), which leaves no KV room on a 24 GiB card.

---

## vLLM memory tuning

**Goal:** largest checkpoint + explicit context without startup OOM.

vLLM normally keeps **weights and KV cache on GPU**. One pool sized by
`gpu_memory_utilization` holds weights, KV cache (from `max_model_len`), and workspace. Generated
configs can opt into RAM-backed pressure relief with `cpu_offload_gb` for model weights and
`kv_offloading_size_gb` for a CPU KV offload buffer.

| GPU VRAM | Typical `gpu_memory_utilization` |
| -------- | ------------------------------- |
| 12 GiB | 0.80-0.90 |
| 16 GiB | 0.80-0.85 |
| 24 GiB | 0.88-0.90 |
| 32 GiB | 0.90-0.92 |

Generated vLLM serve scripts also set `--kv-cache-dtype fp8`, `--max-num-seqs 1`, and text-only
`--limit-mm-per-prompt`; rows with RAM offload also emit `--cpu-offload-gb` and
`--kv-offloading-size`. **Always use explicit `--max-model-len`** from the manifest (never `auto`
in generated configs).

**If startup OOMs:** halve `max_model_len` in
[samples/config-example/manifest.yaml](../../samples/config-example/manifest.yaml), re-run
`make gen-serving-config`, and retry.

### Ollama vs vLLM

| Backend | Weights | When used on a tier |
| ------- | ------- | ------------------- |
| vLLM | GPU, optionally GPU + RAM offload | Largest quant/context that fits host memory |
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
[vLLM backend guide](../guides/benchmarking/vllm-backend.md).

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
([run_config_vllm_uk.yaml](../../samples/configs/run_config_vllm_uk.yaml)).

### 12 GiB GPU / 64 GiB RAM (RTX PRO 3000 Blackwell laptop)

RTX PRO 3000 Blackwell laptop GPU, 12227 MiB, driver 610.43.02. The generated extra vLLM
target is Gemma 4 12B w4a16 at util **0.90**, ctx **16384**, `cpu_offload_gb=16`, and
`kv_offloading_size_gb=32`. A bounded PDF-drafter launch probe confirmed vLLM served that
long-context target with CPU/KV offload on this host; deliberately reducing the draft completion
budget to 512 tokens truncated extraction JSON, so keep `QUICKSTART_DRAFT_MAX_TOKENS=4096` for
production PDF drafting.

### 32 GiB GPU / 64 GiB RAM (HP Z2 Tower)

RTX 5090, 32607 MiB, sm 120. **vLLM** for MamayLM 27B FP8 and Gemma 4 31B w4a16;
**vLLM** for Lapa v0.1.2 Instruct; **Ollama `iq4`** for Qwen3.6 (same class as installed
`qwen3:30b` at 18 GiB).

---

## Related

- [samples/config-example/](../../samples/config-example/) -- manifest + templates
- [vLLM backend guide](../guides/benchmarking/vllm-backend.md)
- [samples/configs/models_uk.yaml](../../samples/configs/models_uk.yaml) -- planner registry
