# RTX PRO 3000 Blackwell 12 GiB Acceptance

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
- The generated Gemma 4 12B vLLM config (`google/gemma-4-12B-it-qat-w4a16-ct`, seed 13, k=5) ran
  ONE item with embeddings on CPU -- a serving smoke, not a quality reading. The contention guard
  accepted 0.90 utilization with 11,696 MiB free of 12,227 MiB (weight floor 7,817 MiB, not
  derated); native sampling, Triton attention, Marlin W4A16, 16 GiB CPU weight offload, and 32 GiB
  KV offload served the full requested 16,384-token context at 3.32 tok/s steady, 51.99 W mean, and
  11,511 MiB peak VRAM. The load took 246.07 seconds, chiefly CUDA-graph capture. FlashInfer 0.6.12
  could not supply its sampler on SM 12.0 and the recorded native-sampler fallback worked. The one
  item scored objective 0.200 with recall@5 1.0 and reliability 1.0; retrieve took 11.75 s against
  1.44 s of generation. Reading: the 12 GiB tier SERVES a 12B W4A16 model at its declared window
  with headroom to spare, and 3.32 tok/s is the price of the CPU/KV offload it needs to do so. n=1
  supports the serving claim and nothing about quality. Lookup key: run
  `serving-12gb-gemma-4-12b-vllm`, run id `2f08bcd131d7`.
- The 20-item Ollama path used the Ukrainian MamayLM Gemma 3 12B Q4_K_M model
  (`hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M`, seed 13, k=5) with CUDA
  embeddings on the 20-item final split. It scored objective 0.406, reliability 1.0, retrieval
  recall@5 0.900 / MRR 0.787, 39.16 tok/s steady at 79.42 W mean (0.493 tokens/W), and 9,932 MiB
  peak VRAM, with retrieve 0.53 s against generate 1.44 s per item. Reading: the Ollama lane is the
  usable everyday path on this tier -- ~12x the vLLM lane's throughput at 2 GiB less peak VRAM,
  because a Q4_K_M GGUF fits without the offload the W4A16 config needed. Objective 0.406 on n=20 is
  a smoke figure, not a leaderboard one. Lookup key: run `rag-eval`, run id `7e94edc3fe16`.
- llama.cpp was not an available backend on this host (`llama-server` was absent), so no llama.cpp
  cell was claimed.
- The repository gate selects only current implementation coverage: obsolete unpublished-artifact
  compatibility checks were removed rather than skipped. It passes 2,226 tests with 43
  opt-in/slow tests deselected and zero runtime skips. Ruff format/check, mypy, Markdown lint, and
  the code-quality report also passed. `ollama ps` was empty after the evidence runs.

The recent paired embedder, context-ablation, and local drafting evidence reruns are recorded in
[RAG core](../rag-core/embedders.md#the-recommendation-re-read-with-paired-uncertainty),
[RAG core](../rag-core/embedders.md#blackwell-encoder-throughput-decomposition),
[RAG core](../rag-core/context-ablation-evidence.md), and
[data prep](../data-prep/drafting-lanes.md#sequential-local-qwengemma-draft-comparison).

Encoder throughput on this host (2026-07-29): `EMBED_ENCODER_THROUGHPUT=1` over the 311-chunk UA
fixture at the 80 W power limit. Warm CUDA rates are ~638 chunks/s for e5-small, 208 for e5-base,
~62 for e5-large and BGE-M3, and ~334 for the paraphrase model; cold load (~5.7 s) dominated the
earlier one-pass rates, but the e5-base vs large spread survives warm measurement. Prefer warm
chunks/s for host cost columns. e5-small is the named cheap CUDA alternative (~3.05x base, lower
peak VRAM) when quality is flat; the paired verdict still RETAINs e5-base on n=82. Lookup keys: run
id `1d36908e745c` (full roster) and `c79df0776706` (VRAM after the release fix); the per-encoder
numbers those two carry are tabulated on the RAG-core page linked below rather than repeated here.
See [RAG core](../rag-core/embedders.md#blackwell-sub-base-encoder-roster-e5-small).
