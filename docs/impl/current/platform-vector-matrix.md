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

## Provider-Aware Bounded Model Downloads

`make download-model` caches very large open models without handing an unbounded snapshot to a
provider SDK. The shared implementation under `src/llb/backends/model_download/` resolves a moving
provider reference once, persists the immutable revision and file identities under
`.llb-model-download/state.json`, and transfers verified HTTP ranges into the requested target.
Every completed chunk is flushed, read back, SHA-256 checked, and atomically checkpointed. A later
invocation resumes from the verified tail; `MODEL_DOWNLOAD_VERIFY_COMPLETED=1` performs the deeper
whole-cache scan needed to detect storage corruption after a prior session.

The provider adapters preserve each provider's usable local shape:

- `huggingface` (alias `hf`) pins the Hub commit and verifies LFS SHA-256 or Git blob SHA-1 before
  publishing each normal snapshot file. `HF_TOKEN` supports gated/private repositories.
- `ollama` resolves the public Ollama OCI manifest, verifies every content-addressed blob, writes the
  standard `blobs/sha256-*` store, and publishes the tag manifest last under
  `manifests/registry.ollama.ai/`. Point `OLLAMA_MODELS` at the target when serving that store.
- `github-release` (alias `github`) pins a release and downloads its assets through the GitHub API.
  It refuses assets that lack the provider's server-side SHA-256 digest instead of recording an
  unverifiable cache. `GITHUB_TOKEN` raises API limits and permits accessible private releases.

For example, a bounded Hugging Face session is:

```text
make download-model \
  MODEL_DOWNLOAD_PROVIDER=huggingface \
  MODEL_DOWNLOAD_ID=moonshotai/Kimi-K3 \
  MODEL_DOWNLOAD_TARGET=<model-dir> \
  MODEL_DOWNLOAD_SESSION_GIB=32 \
  MODEL_DOWNLOAD_CHUNK_MIB=64
```

Use `MODEL_DOWNLOAD_DRY_RUN=1` to resolve the immutable revision and exact provider-reported size
without creating the target. `MODEL_DOWNLOAD_MAX_MIBPS` is an explicit ceiling; the adaptive
`MODEL_DOWNLOAD_BANDWIDTH_FRACTION` defaults to 0.8 and paces each verified chunk against an EWMA of
the observed end-to-end transfer rate. The per-invocation byte budget may stop inside a large shard,
so no provider file has to fit inside one session.

Disk protection is evaluated before every network chunk. The remaining free space must cover the
next chunk plus the larger of `MODEL_DOWNLOAD_MIN_FREE_GIB` and
`MODEL_DOWNLOAD_MIN_FREE_PERCENT` of the destination filesystem (defaults: 1 GiB and 5%). Thus a
large data volume keeps a proportional safety margin while a small volume still retains an absolute
floor. A nonblocking target lock prevents concurrent writers; state replacement and final-file
publication are atomic; 429 responses honor `Retry-After`/rate-limit reset hints; transient network
and server failures use bounded retries; 401/403 errors preserve the last checkpoint and never
persist tokens. Set `MODEL_DOWNLOAD_VERIFY_ONLY=1` for an offline integrity pass after the snapshot
has been cached.

Live provider validation used metadata-only reads for all three adapters: Kimi-K3 resolved to 118
files / 1.42 TiB at commit `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; Ollama
`tinyllama:latest` resolved to 6 files / 608.16 MiB at manifest
`sha256:2644915ede352ea7bdfaff0bfac0be74c719d5d5202acb63a6fb095b52f394a4`; and the
latest `ggml-org/llama.cpp` GitHub release resolved to 25 checksum-bearing assets / 2.40 GiB.
Two real Kimi sessions capped at 10.49 KiB each then exercised cross-process resume: the second
started the partial `README.md` at the prior 5,429-byte checkpoint and advanced it to 16,166 bytes
without re-fetching completed files. Separate 10.49 KiB live transfer smokes validated the immutable
Ollama blob and GitHub asset URLs; the Ollama manifest remained unpublished while its model blob was
partial, as required by the store-visibility contract. The temporary partial-download roots were
removed after validation.

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
make compare-vector-stores CONFIG=<run>.yaml VECTOR_BACKENDS=faiss,chroma,qdrant NOISE_FLOOR=1
```

When `--goldset <bundle>/goldset.jsonl` is passed and `<bundle>/corpus/` exists,
`compare-vector-stores` uses the sibling corpus automatically. Pass `--corpus-root` when the paths
are separate. With `CONFIG=` the YAML owns the gold set and the split, and `GOLDSET=` / `SPLIT=`
are forwarded only when the caller actually sets them -- the same rule `compare-embeddings` uses,
so a config-targeted backend comparison cannot silently score the default gold set against the
config's corpus. `NOISE_FLOOR=1` adds the [measurement
floor](rag-core/retrieval-metrics.md#measurement-floor---noise-floor) per backend, which is what
says whether a backend-to-backend delta is a ranking at all.

Use one isolated `DATA_DIR` per validation run when you need to keep persisted stores for multiple
backends.

### Paired Backend Evidence

Every backend row also carries a PAIRED delta interval against the incumbent backend over shared
resample index sets, its win/loss/tie ledger, and an explicit adopt-or-retain verdict -- the same
lane, statistics, and verdict vocabulary an embedder swap is decided on
([RAG core](rag-core/paired-verdicts.md#paired-uncertainty-and-the-adopt-or-retain-verdict)). The
point estimate alone cannot say whether a backend gap is real, and `best (recall@k)` is label order
when the backends tie, so the verdict rather than that line is the recommendation.

`VECTOR_BASELINE=` names the incumbent (default `faiss` when it is in `VECTOR_BACKENDS=`, else the
first selected backend; naming a backend that was not scored exits 2 rather than pairing against a
different row). `VECTOR_RESAMPLES=`, `VECTOR_CONFIDENCE=`, and `VECTOR_SEED=` set the draw. The
report JSON gains `uncertainty`, `paired_items` (the per-item ledger the reading is recomputable
from), `backends.<row>.paired_vs_baseline`, and `verdict`.

A backend comparison has one reading the embedder lane rarely sees: a ledger with zero wins AND
zero losses on every metric. That is stronger than the `flat` reading printed per column -- `flat`
says this item set did not separate the lanes, while zero discordant pairs says they returned the
same evidence for every question, so no larger gold set would separate them either. The verdict
names that case explicitly instead of reporting an unresolved measurement.

Modules and tests: `src/llb/cli/rag/compare_stores.py` (command + the shared
`resolve_paired_baseline`, reused by `compare-retrieval`), `src/llb/rag/compare.py` (scoring +
paired attachment), `src/llb/rag/retrieval_comparison_uncertainty.py` (the verdict);
`tests/llb/rag/test_compare_retrieval_cli.py` and `tests/llb/rag/test_compare_retrieval_core.py`
cover the columns, the baseline resolution, and the verdict over fake stores.

### Measured Backend Comparison

Measured on the CUDA host (RTX 4060 Ti, 16,380 MiB) on 2026-08-19 with the pinned e5-base encoder,
`recursive` 800/120, k=10, 2000 resamples at 95% confidence, seed 13, on the same two corpora the
embedder bake-off is read on. Configs and reports:
`$DATA_DIR/compare-vector-stores/paired-uncertainty-{pdf,fixture}.yaml` and
`$DATA_DIR/compare-vector-stores/paired-uncertainty-{pdf,fixture}/report.json`. The earlier
floor-only reading is at `$DATA_DIR/compare-embeddings/floor-reread/vector-stores-floor.json`.

40-item accepted converted-PDF goldset:

| backend | recall@10 | MRR | recall delta vs faiss | w/l/t | floor recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `faiss` | 0.925 | 0.852 | +0.000 `[0.000, 0.000]` | 0/0/40 | +/-0.000 |
| `chroma` | 0.925 | 0.852 | +0.000 `[0.000, 0.000]` | 0/0/40 | +/-0.000 |
| `qdrant` | 0.925 | 0.852 | +0.000 `[0.000, 0.000]` | 0/0/40 | +/-0.000 |

250-item committed UA fixture (`samples/goldsets/ua_squad_postedited_v1/`):

| backend | recall@10 | MRR | recall delta vs faiss | w/l/t | floor recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `faiss` | 0.980 | 0.847 | +0.000 `[0.000, 0.000]` | 0/0/250 | +/-0.000 |
| `chroma` | 0.980 | 0.847 | +0.000 `[0.000, 0.000]` | 0/0/250 | +/-0.000 |
| `qdrant` | 0.980 | 0.847 | +0.000 `[0.000, 0.000]` | 0/0/250 | +/-0.000 |

**Verdict on both corpora: RETAIN `faiss`.** Not "the gap is small" -- the paired ledger is empty
on both item sets (no discordant pair on recall, MRR, coverage, or intactness), so the point-estimate
`best (recall@k): chroma` line is label order and no larger gold set would separate the three
backends. Coverage@k and intact@k are identical too. That is the designed outcome of the
[vector-store seam](#vector-store-seam): the adapters map query embeddings to ids and `RagStore`
owns the chunk records, so the source-span metric SHOULD be backend-invariant, and it is now
measured on two corpora rather than assumed. Choose the backend on operational grounds (build time,
footprint, deployment), not on these rows. `lancedb` is not in the row set because its optional
extra is not installed on this host.

One caveat the re-reads surfaced: the PDF lane was read three times and one of the three runs put
`chroma` one item ahead of `faiss` on MRR (+0.001, 1/0/39) while the other two were itemwise
identical (`report-repeat.json` beside the recorded report). The recall ledger was empty in all
three. A single discordant item cannot reach the 95% reporting level -- which needs six
([evidence gate](rag-core/paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading))
-- so it cannot move the verdict; read it as rebuild noise in the approximate adapter indexes, not
as a backend property. The measurement floor jitters scores within ONE build and therefore reports
+/-0.000 here; between-build variation is a separate axis this lane does not price.

## Embedding Bake-off

`compare-vector-stores` fixes the embedder and varies the backend; `compare-embeddings` fixes the
backend + chunking and varies the EMBEDDER, ranking candidates on recall@k / MRR plus embed
throughput, index size, dimension, and device. See [RAG core](rag-core.md) (Embedder Conventions And
Bake-off) for the module map, the per-family query/passage conventions, the store/query embedder
fingerprint guard, and the opt-in Cohere API-row egress gate.

```bash
make compare-embeddings GOLDSET=<bundle>/goldset.jsonl RAG_K=10 NOISE_FLOOR=1
make compare-embeddings GOLDSET=... EMBED_ALLOW_REMOTE_CODE=1   # opt into trust_remote_code rows
make compare-embeddings GOLDSET=... EMBED_DTYPE=float32         # one declared precision for every row
make compare-embeddings-legacy CONFIG=<run.yaml>                # the transformers 4.x scoring pass
llb compare-embeddings --goldset <bundle>/goldset.jsonl --k 10 --noise-floor \
  --models intfloat/multilingual-e5-base,intfloat/multilingual-e5-small,\
intfloat/multilingual-e5-large,BAAI/bge-m3 \
  --baseline intfloat/multilingual-e5-base
make build-index EMBEDDING_MODEL=intfloat/multilingual-e5-base   # apply an ADOPTED embedder
```

The default roster is nine ids: the four incumbents, the current multilingual retrieval generation
(`multilingual-e5-large-instruct`, `gte-multilingual-base`, `jina-embeddings-v3`,
`Qwen3-Embedding-0.6B`), and the paraphrase/STS control. A candidate with no declared query/passage
convention fails the run before any store is built, a `trust_remote_code` candidate is skipped with
its reason recorded unless `EMBED_ALLOW_REMOTE_CODE=1`
([RAG core](rag-core/embedders.md#roster-screening)), and a candidate whose repository code targets
a different transformers major is routed to `compare-embeddings-legacy`. Every scored candidate
must reproduce its own model card before a store is built for it
([RAG core](rag-core/stack-and-card-parity.md)).

Every candidate row carries a PAIRED delta interval against `--baseline` plus the win/loss/tie
ledger, and the report ends in an explicit adopt-or-retain verdict rather than a point-estimate
rank; see [RAG core](rag-core/paired-verdicts.md#paired-uncertainty-and-the-adopt-or-retain-verdict).

Recommended embedder for the 16 GB host: `intfloat/multilingual-e5-base`, the current default. On a
12 GiB Blackwell host that must keep embeddings on CUDA beside a served generator,
`intfloat/multilingual-e5-small` is the measured cheap alternative (~3x warm encode, lower VRAM)
when quality is flat -- still RETAIN until a paired adopt clears ([RAG
core](rag-core/embedders.md#blackwell-sub-base-encoder-roster-e5-small)). The 2026-07-10
`embedding-bakeoff-full-corpus` evidence (four local candidates over a verified 44-item
quickstart-PDF accepted goldset, 1139 chunks) put it ahead on recall@10 (0.955 vs 0.932 for e5-large
and bge-m3) with ~1.8x the embed throughput of the 1024-dim pair (69 vs 38 chunks/s on GPU) and the
smallest index (4.99 MB vs 6.10 MB); e5-large was the MRR winner (0.795 vs 0.740) and tied e5-base
at recall@20. That goldset is no longer on disk, and the 2026-07-24 floor re-read on the accepted
goldset that survives does NOT reproduce the ranking -- `bge-m3` leads there by 0.050 recall@10
against a zero floor, and e5-base ties e5-large. The 2026-07-24 paired re-read then settles the
reading: on that accepted goldset the verdict is RETAIN (`bge-m3` +0.050 `[-0.050, +0.150]`, 3 wins
/ 1 loss / 36 ties), while on the committed 250-item UA fixture the verdict is ADOPT `e5-large`
(+0.020 `[+0.004, +0.040]`, 5 wins / 0 losses) -- two corpora, two different separated candidates,
so the default stays put until an accepted operator-corpus ledger separates one. Tables and the full
reading are in [RAG core](rag-core/embedders.md#the-recommendation-re-read-with-paired-uncertainty).
Where `bge-m3` DOES separate on the PDF corpus is first-hit rank (MRR +0.064), and the end-to-end
[scoped first-hit-rank
bar](rag-core/first-hit-rank-adoption.md#the-scoped-first-hit-rank-adoption-bar) measures that this
rank gain reaches the answer under a reranker (+0.052 objective) or a small `top_k` (+0.034) but is
free at the shipped k=10 default (-0.010) -- so `--adoption-bars recall_at_k,mrr` adopts `bge-m3`
for those two configurations while the recall@k-only default is retained for the shipped one. The
paraphrase/STS `lang-uk` model collapses on every run (recall@10 0.455 / 0.475 / 0.856) and is the
one row that separates from the baseline in the negative direction on both corpora. Embed VRAM
peaked ~4 GB, so all candidates fit the 16 GB host.

The 2026-08-16 roster refresh added the current multilingual generation and did NOT change that
call: on a 16,380 MiB RTX 4060 Ti both corpora still verdict **RETAIN `e5-base`**, with
`multilingual-e5-large-instruct` tying `bge-m3` on the fixture (+0.012, 3 wins / 0 losses) and
losing an item on the PDF corpus, and `Qwen3-Embedding-0.6B` tying the baseline on one corpus and
sitting below it on the other. Peak VRAM across the seven scored candidates stayed under 5.1 GiB.
`gte-multilingual-base` and `jina-embeddings-v3` are scored in the transformers 4.x
[legacy pass](rag-core/stack-and-card-parity.md#the-legacy-transformers-pass) instead, and neither
changes the call either: gte sits `-0.024` recall below the incumbent and jina `+0.008` with an
interval spanning zero, at 2.1x its peak VRAM. The 3.4x throughput lead of
`multilingual-e5-large-instruct` turns out to be its float16 upload -- at a declared
`EMBED_DTYPE=float32` it lands within 1.4% of `e5-large`. Full tables are in
[RAG core](rag-core/embedders.md#the-refreshed-candidate-roster-2026-08-16) and
[the scoring stack and card-parity gate](rag-core/stack-and-card-parity.md).

## Reranker Bake-off

`compare-rerankers` fixes the encoder, the chunking, and the candidate pool and varies the
CROSS-ENCODER, ranking candidates on recall@k / MRR / first-hit rank plus the two columns a reranker
is actually chosen on -- rerank latency per query and the VRAM it holds beside a resident generator.
See [RAG core](rag-core/reranker-bakeoff.md) for the lane, the shared-pool design, the reranker-off
row, the per-candidate process isolation, and the full tables.

```bash
make compare-rerankers GOLDSET=<bundle>/goldset.jsonl CORPUS=<bundle>/corpus SPLIT= RAG_K=10 \
  NOISE_FLOOR=1 RERANK_GENERATOR_VRAM_MB=<mb>   # declares the budget the fit gate reads
make compare-rerankers GOLDSET=... RERANK_ALLOW_REMOTE_CODE=1   # opt into trust_remote_code rows
make compare-rerankers-legacy GOLDSET=... CORPUS=... SPLIT=      # the transformers 4.x scoring pass
make run-eval MODEL=<m> RERANKER=BAAI/bge-reranker-v2-m3        # apply a chosen reranker
```

Recommended reranker for the 16 GB host: `BAAI/bge-reranker-v2-m3`, the current default. The
2026-08-16 bake-off (five candidates, both corpora, `e5-base` + `recursive@800/120`, pool 30, k=10,
measured with a 12B UA generator holding 8,278 MiB) verdicts **RETAIN** on both: `Qwen3-Reranker-0.6B`
ties it on recall and sits 0.017 MRR below at twice the latency, and `mxbai-rerank-base-v2` is below
it on both bars with intervals that exclude zero on the 250-item fixture. What the reranker buys is
first-hit RANK -- against no reranking, +0.094 MRR `[+0.063, +0.127]` but only +0.020 recall@10 --
for ~530-600 ms per query and a ~4.5 GiB scoring peak, which fits beside the generator on this host.
`jina-reranker-v2-base-multilingual` and `gte-multilingual-reranker-base` are scored in the
transformers 4.x [legacy pass](rag-core/stack-and-card-parity.md#the-legacy-transformers-pass), the
same packaging hole the encoder roster hit. Both are below the incumbent on both bars on the
fixture, but at ~7x lower latency (73 and 80 ms per query against 546) and about a third of its
resident VRAM -- a latency/quality frontier, not a swap candidate. Full tables are in
[RAG core](rag-core/reranker-bakeoff.md#what-the-bake-off-measured-2026-08-16-cuda-host) and
[the scoring stack and card-parity gate](rag-core/stack-and-card-parity.md#the-two-rerankers).
