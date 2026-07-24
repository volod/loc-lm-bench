# Platform Matrix And Vector Stores

The platform matrix compares a logical model family across serving backends on the same host and
gold split. The vector-store matrix compares local vector backends under the same chunking,
embedding, and source-span retrieval metric.

## Backend Matrix

`make platform-matrix` runs the same logical base across Ollama, vLLM, and llama.cpp when matching
artifacts are available for the host.

```bash
make platform-matrix
```

Useful overrides:

```text
PLATFORM_MATRIX_OLLAMA_MODEL
PLATFORM_MATRIX_VLLM_MODEL
PLATFORM_MATRIX_LLAMACPP_MODEL
PLATFORM_MATRIX_MAX_MODEL_LEN
PLATFORM_MATRIX_GPU_MEMORY_UTILIZATION
PLATFORM_MATRIX_LIMIT
PLATFORM_MATRIX_BACKENDS
PLATFORM_MATRIX_STRICT
```

The matrix uses `run-eval --telemetry`, so each row records objective quality, reliability,
tokens/sec, VRAM, load time, power, tokens per watt, and quality per watt.
By default the Make target runs the requested backend rows that can actually start on the host:
vLLM requires the `vllm` executable, and llama.cpp requires either
`$DATA_DIR/llb/llamacpp/build/bin/llama-server` or `llama-server` on `PATH`. Missing optional
backend binaries are logged as skips; set `PLATFORM_MATRIX_STRICT=1` to make those skips or row
failures fail the target.

The current default common base for a 16 GB CUDA host is Gemma 4 E4B IT:

- Ollama: `gemma4:e4b`;
- vLLM: `google/gemma-4-E4B-it-qat-w4a16-ct`;
- llama.cpp: `hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:q4_0-it`.

If a requested larger base has no matching artifact for one backend, prefer an actually comparable
common base over mixing unrelated checkpoints.

Quickstart validation on the 16 GiB RTX 4060 Ti host used
`.data/quickstart-leaderboard/run-eval/20260630T053945.651376Z-5544ffad36c2/manifest.json`:
Ollama `gemma4:e4b`, 20 final cases, objective `0.420`, reliability `0.750`, `60.04` tok/s,
peak VRAM `13717` MB, `120.03` W mean power, `0.5002` tokens/W, and retrieval
`recall@5=0.900`, `mrr=0.7875`. vLLM and llama.cpp rows were skipped because their serving
executables were not installed.

## Power Metrics

When `nvidia-smi` is reachable, telemetry records:

- `telemetry.mean_power_w`;
- `telemetry.peak_power_w`;
- `telemetry.power_samples`;
- `telemetry.tokens_per_watt`;
- `metrics.mean_power_w`;
- `metrics.tokens_per_watt`;
- `metrics.quality_per_watt`.

`quality_per_watt = objective_score * tokens_per_s / mean_power_w`. Keep raw
`tokens_per_watt` for serving efficiency and `quality_per_watt` for benchmark efficiency.

## GPU-Class Configs

`detect-gpu-vram` and `gen-serving-config` generate host-specific serving scripts and run configs
under `$DATA_DIR/llb/serving/gpu-<tier>gb/`.

```bash
llb detect-gpu-vram
llb gen-serving-config
llb gen-serving-config --gpu-gb 12
llb gen-serving-config --gpu-gb 24
llb gen-serving-config --gpu-gb 32
```

The generated directory contains `tier.json`, serve scripts, and `run-eval` YAML/scripts. Primary
tier targets are MamayLM, Lapa, Gemma 4, Qwen3.6, and Mistral; extra tier entries such as smaller
vLLM Gemma variants are emitted after those primary targets. This path lets another physical GPU
host contribute comparable manifest rows without hardcoding host paths.
Target ids are family-level keys; for example `gemma-4` generates `serve_gemma_4.sh` while the
tier manifest selects the concrete largest model variant that fits the host.
Concrete MamayLM references use v2.0 source names and labels: generated configs select the 16 GiB
Ollama GGUF or 32 GiB vLLM FP8 v2.0 source, prepare-model fixtures use the INSAIT v2.0 source names,
and recommendation fixtures use `mamaylm-v2-*` labels. The family key `mamaylm` remains only the
stable target id and file stem.

## Ukrainian Model Roster Refresh

The 2026-07-21 roster survey used the public
[lang-uk result set](https://huggingface.co/datasets/lang-uk/ukrainian-llm-leaderboard-results),
the [MamayLM v2.0 collection](https://huggingface.co/collections/INSAIT-Institute/mamaylm-v20-gemma-3),
and first-party model cards as the admission filter. The two useful additions are:

- `gemma-4-26b-a4b`: the 25.2B-total / approximately 3.8B-active Gemma 4 MoE represented in the
  public Ukrainian reasoning results. Its sources are Google bf16, Red Hat FP8 for vLLM,
  first-party `gemma4:26b` for Ollama, and Google's Q4_0 GGUF for llama.cpp.
- `qwen3.6-27b`: the official dense Qwen3.6 27B release, with bf16 and FP8 vLLM records,
  `qwen3.6:27b` for Ollama, and the Unsloth Q4_K_M GGUF for llama.cpp. The
  [official model card](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) identifies the FP8 artifact
  as vLLM-compatible and Apache-2.0.

Every logical entry in `samples/configs/models_uk.yaml` now has structured `license` and
`license_url` fields and resolves across vLLM, Ollama, and llama.cpp source records. Same-backend
quant lists are quality-ordered by bits per weight, so an official Q4 Ollama tag is preferred over
an IQ3 fallback when both are installed. `samples/config-example/manifest.yaml` adds concrete
`gemma-4-26b` and `qwen3.6-27b` tier targets: both use Ollama offload on 12/16 GiB, while Gemma 26B
uses FP8 vLLM on 32 GiB; Qwen 27B remains an Ollama target because its untied embedding overhead
keeps FP8 above the supported 32 GiB serving budget.

Planner and resolver fixes made the refreshed rows truthful:

- cached Hugging Face configs now fill `kv_dim`, `max_context`, and `kv_layers`; Qwen3.6's hybrid
  linear/full attention prices growing KV only on its full-attention layers;
- `list-models` counts a vLLM row runnable only when at least 2,048 tokens fit fully in GPU VRAM,
  while Ollama and llama.cpp may count a CPU-offloaded `ctx_max`;
- GGUF discovery normalizes `hf.co/<repo>:<quant>` before probing Hugging Face;
- sweep and joint search share an executable-readiness check, so a remote GGUF no longer becomes a
  runnable llama.cpp cell when `llama-server` is absent;
- Ollama benchmark calls use native `/api/chat` with `think=false`, keeping bounded scoring tokens
  in the answer for Qwen/Gemma reasoning templates. The OpenAI-compatible endpoint was rejected
  here because a live Qwen case spent 512 tokens on hidden reasoning and returned empty content;
  the native-path probe returned the expected `Kyiv` answer in 1.35 seconds with 3 completion
  tokens;
- joint-search forwards an explicit case limit into final-split pick scoring and evaluates
  identical goal configurations once, then writes the shared outcome to each goal's resume marker.

On the 16 GiB RTX 4060 Ti, `make list-models` reports 3 backend-runnable declared artifacts out of
14 quant-expanded vLLM/Ollama rows instead of the former misleading 9 of 9 hardware-only count.
Live `resolve-models` resolves all 10 logical candidates; the two additions select
`gemma4:26b` and `qwen3.6:27b` through Ollama offload. The official Qwen tag was prepared through
`make prep-models`; Ollama reports a 43 percent CPU / 57 percent GPU split at 4,096-token context.

On a 12 GiB RTX PRO 3000 Blackwell laptop GPU (12227 MiB, driver 610.43.02), the quickstart setup
generates and selects `$DATA_DIR/llb/serving/gpu-12gb/tier.json` from current host detection rather
than the presence of tier directories. The 12 GiB extra vLLM target is `gemma-4-12b-vllm`:
`google/gemma-4-12B-it-qat-w4a16-ct`, `gpu_memory_utilization=0.90`, `max_model_len=16384`,
`cpu_offload_gb=16`, and `kv_offloading_size_gb=32`. A bounded PDF-drafter launch probe on the same
host confirmed vLLM started with CPU/KV offload, reported 78,115 GPU KV-cache tokens, and allowed
4.77x concurrency for 16,384-token requests. The 512-token reduced probe returned useful extraction
content but hit the completion cap before closing JSON, so production PDF drafting keeps the default
`QUICKSTART_DRAFT_MAX_TOKENS=4096`. The resolver also prices vLLM candidates with the same serving
overhead used by the pre-launch contention guard and the default vLLM memory fraction, so sweeps do
not select vLLM rows that will be aborted immediately by the guard or by KV-cache allocation.

The Mistral family default is Mistral Small 3.1 24B (Apache-2.0, ungated, multilingual), served per
tier by the quant that fits GPU-resident: vLLM FP8
(`RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic`, ~24 GiB weights) on the 32 GiB tier,
vLLM w4a16 (`RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16`, ~14 GiB weights) on the
24 GiB tier, and Ollama's curated `mistral-small3.1:24b` (q4_k_m, CPU offload) on the 12/16 GiB
tiers. The curated Ollama tag is deliberate: the lmstudio/bartowski HF GGUF mirrors of this
checkpoint crash the Ollama 0.20 llama.cpp runner on load (exit status 2), while the curated tag is
tested against the runtime and serves the text path (we score text only). The planner registry
entry (`mistral-small-3.1-24b` in `samples/configs/models_uk.yaml`) lists BOTH vLLM quants under
`sources.vllm` (fp8 + w4a16); the resolver is embedding-aware (prices the untied 131k-token
embedding at bf16, so w4a16 lands at ~14.4 GiB and fp8 at ~23.6 GiB, not the flat
`params_b x bpw`) and picks the highest-quality quant whose serving window fits the GPU -- fp8 on
32 GiB, w4a16 on 24 GiB -- then the curated GGUF on 12/16 GiB (see [multi-quant
resolution](#multi-quant-vllm-resolution)). That makes the sweep path agree with the 32 GiB
serving tier (`samples/config-example/manifest.yaml`), which also serves the higher-quality fp8.

Smoke-validated on the 16 GiB RTX 4060 Ti host: `make list-models` rates the Mistral entry runnable
(w4a16 ~14.4 GiB weights, `ctx_gpu=828` so vLLM does not clear the GPU window -> offload), the
resolver picks `mistral-small3.1:24b` on Ollama, and a 3-case `run-eval --telemetry` on the
committed `ua_squad_postedited_v1` final split served via Ollama CPU offload with `recall@5=1.000`,
`reliability=1.000`, `12.7` tok/s, peak VRAM `15977` MB
(`.data/quickstart-leaderboard/run-eval/20260630T152748.480864Z-e1bb196e19d9/`). The vLLM w4a16
(24 GiB) and fp8 (32 GiB) rows are bigger-GPU-host runs, not exercised on this 16 GiB box.

## Multi-Quant vLLM Resolution

A logical model entry can declare SEVERAL vLLM quants under `sources.vllm` as a list of records
(each with its own `quant`/`source`/`min_vram_gb`, inheriting the shared arch from the parent).
`candidate_sources` (`src/llb/backends/resolver.py`) orders those quants highest-bits-per-weight
first, so the existing "first runnable wins" rule picks the best-quality quant whose `ctx_gpu >=
MIN_SERVING_CTX` on the host, then falls through to the Ollama/llama.cpp offload. For Mistral that
yields fp8 on a 32 GiB card, w4a16 on a 24 GiB card, and the curated GGUF on 12/16 GiB -- one entry,
the right quant per host -- so the sweep/host-fit path matches the per-tier serving config.
Model-prep expansion (`_expand_prepare_sources`) mirrors the shape: each listed quant becomes its
own prep artifact (`<name>-vllm-<quant>`), so `prep-models` caches every quant that fits the card.
`make list-models` likewise expands a multi-quant entry into one fit row per quant
(`_expand_quant_variants` in `src/llb/cli/models/prep.py`), so the host-fit table shows the fp8 row the
resolver would pick on a big card -- not just the parent quant -- while `resolve-models` still
prints the single chosen backend. Single-source entries are unchanged throughout.

## Model-Prep Disk Preflight

`prep-models` / `prep-serving-targets` reuse any artifact already in its backend store and refuse a
download up front when the destination filesystem cannot hold it, so a multi-GiB pull never fails an
hour in (`src/llb/backends/prepare/stores.py`). The check is reuse-aware: a vLLM repo whose `config.json`
is already in the HF hub cache, or an Ollama tag the running daemon serves, skips the precheck and
re-uses the cache. The Ollama reuse signal is authoritative -- it asks the daemon via the same
`/api/tags` probe the resolver uses, so a tag in any store the daemon is configured with counts,
falling back to an on-disk blob-store scan only when the daemon is unreachable. Otherwise the check
requires free space `>= estimate * 1.15 + 2048 MiB`, where the estimate is the embedding-aware
planner weight size; an unknown free-space probe (`0`) never blocks.
Store roots resolve from `OLLAMA_MODELS`, else the first existing of `~/.ollama/models` and the
systemd-package `/usr/share/ollama/.ollama/models` (so a service install is probed where it
actually writes), and from `HF_HUB_CACHE` / `HF_HOME` / `--cache-dir` (default
`~/.cache/huggingface/hub`). `--dry-run` previews the disk plan (`[disk: ...]`) without
downloading.

## llama.cpp Binary Lookup

The llama.cpp launcher first checks the project-managed binary under
`$DATA_DIR/llb/llamacpp/build/bin/llama-server`, then falls back to `PATH`. This lets
`make build-llamacpp` feed `run-eval --backend llamacpp` without requiring a shell profile edit.

## Vector-Store Seam

`src/llb/rag/vector_index.py` defines the `VectorIndex` protocol and backend dispatch:

```text
faiss
chroma
qdrant
lancedb
```

`RagStore` owns chunk records and source offsets. Vector indexes only map query embeddings to
build-order ids plus similarity. That design keeps `.retrieve(question, k)` and source-span
metrics unchanged across backends.

Adapters live under `src/llb/rag/stores/`:

- `base.py`: shared id shaping and persistence helpers;
- `chroma.py`: Chroma adapter;
- `qdrant.py`: Qdrant adapter;
- `lancedb.py`: LanceDB adapter.

Optional extras pin validated client APIs: `[rag-chroma]`, `[rag-qdrant]`, and `[rag-lancedb]`.
The default `make venv` installs the Chroma and Qdrant extras so the full local test suite
exercises their live adapter round-trips without optional-dependency skips. LanceDB remains an
opt-in adapter lane; its live refresh-equivalence parameter is marked `opt_in_env` and therefore
deselected, rather than skipped, by regular CI.

## Vector-Store Commands

```bash
llb build-index --corpus-root <bundle>/corpus --vector-store faiss
llb build-index --corpus-root <bundle>/corpus --vector-store chroma
llb build-index --corpus-root <bundle>/corpus --vector-store qdrant
llb build-index --corpus-root <bundle>/corpus --vector-store lancedb
llb validate-retrieval --goldset <bundle>/goldset.jsonl --k 10
make compare-vector-stores GOLDSET=<bundle>/goldset.jsonl RAG_K=10 \
  VECTOR_BACKENDS=faiss,chroma,qdrant,lancedb NOISE_FLOOR=1 \
  COMPARE_STORES_OUT=<report>.json
```

When `--goldset <bundle>/goldset.jsonl` is passed and `<bundle>/corpus/` exists,
`compare-vector-stores` uses the sibling corpus automatically. Pass `--corpus-root` when the paths
are separate. `NOISE_FLOOR=1` adds the
[measurement floor](rag-core.md#measurement-floor---noise-floor) per backend, which is what says
whether a backend-to-backend delta is a ranking at all.

Use one isolated `DATA_DIR` per validation run when you need to keep persisted stores for multiple
backends.

Measured backend comparison (CUDA host, 2026-07-24, pinned e5-base, `recursive` 800/120, 40-item
accepted converted-PDF goldset, k=10; report at
`$DATA_DIR/compare-embeddings/floor-reread/vector-stores-floor.json`):

| backend | recall@10 | MRR | fragile | floor recall@10 |
| --- | ---: | ---: | ---: | ---: |
| `faiss` | 0.925 | 0.852 | 0/40 | +/-0.000 |
| `chroma` | 0.925 | 0.852 | 0/40 | +/-0.000 |
| `qdrant` | 0.925 | 0.852 | 0/40 | +/-0.000 |

The three backends are indistinguishable: identical recall@10, identical MRR, a zero floor in each,
and a leader-to-runner-up gap of 0.000 that the report states does not clear the floor. The
`best (recall@k): chroma` line the table prints is label order, not a recommendation -- which is the
reason the floor was wired into this lane. That is the designed outcome of the
[vector-store seam](#vector-store-seam): the adapters map query embeddings to ids and `RagStore`
owns the chunk records, so the source-span metric SHOULD be backend-invariant on a corpus this
size, and now it is measured rather than assumed. Choose the backend on operational grounds (build
time, footprint, deployment), not on these rows. `lancedb` is not in the row set because its
optional extra is not installed on this host.

## Embedding Bake-off

`compare-vector-stores` fixes the embedder and varies the backend; `compare-embeddings` fixes the
backend + chunking and varies the EMBEDDER, ranking candidates on recall@k / MRR plus embed
throughput, index size, dimension, and device. See [RAG core](rag-core.md) (Embedder Conventions And
Bake-off) for the module map, the per-family query/passage conventions, the store/query embedder
fingerprint guard, and the opt-in Cohere API-row egress gate.

```bash
make compare-embeddings GOLDSET=<bundle>/goldset.jsonl RAG_K=10 NOISE_FLOOR=1
llb compare-embeddings --goldset <bundle>/goldset.jsonl --k 10 --noise-floor \
  --models intfloat/multilingual-e5-base,intfloat/multilingual-e5-large,BAAI/bge-m3 \
  --baseline intfloat/multilingual-e5-base
make build-index EMBEDDING_MODEL=intfloat/multilingual-e5-base   # apply an ADOPTED embedder
```

Every candidate row carries a PAIRED delta interval against `--baseline` plus the win/loss/tie
ledger, and the report ends in an explicit adopt-or-retain verdict rather than a point-estimate
rank; see [RAG core](rag-core.md#paired-uncertainty-and-the-adopt-or-retain-verdict).

Recommended embedder for the 16 GB host: `intfloat/multilingual-e5-base`, the current default. The
2026-07-10 `embedding-bakeoff-full-corpus` evidence (four local candidates over a verified 44-item
quickstart-PDF accepted goldset, 1139 chunks) put it ahead on recall@10 (0.955 vs 0.932 for
e5-large and bge-m3) with ~1.8x the embed throughput of the 1024-dim pair (69 vs 38 chunks/s on
GPU) and the smallest index (4.99 MB vs 6.10 MB); e5-large was the MRR winner (0.795 vs 0.740) and
tied e5-base at recall@20. That goldset is no longer on disk, and the 2026-07-24 floor re-read on
the accepted goldset that survives does NOT reproduce the ranking -- `bge-m3` leads there by 0.050
recall@10 against a zero floor, and e5-base ties e5-large. The 2026-07-24 paired re-read then
settles the reading: on that accepted goldset the verdict is RETAIN (`bge-m3` +0.050
`[-0.050, +0.150]`, 3 wins / 1 loss / 36 ties), while on the committed 250-item UA fixture the
verdict is ADOPT `e5-large` (+0.020 `[+0.004, +0.040]`, 5 wins / 0 losses) -- two corpora, two
different separated candidates, so the default stays put until an accepted operator-corpus ledger
separates one. Tables and the full reading are in
[RAG core](rag-core.md#the-recommendation-re-read-with-paired-uncertainty). The paraphrase/STS
`lang-uk` model collapses on every run (recall@10 0.455 / 0.475 / 0.856) and is the one row that
separates from the baseline in the negative direction on both corpora. Embed VRAM peaked ~4 GB, so
all candidates fit the 16 GB host.
