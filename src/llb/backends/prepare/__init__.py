"""Prepare candidate models for the local backends.

Two backends, two stores:
  - ollama: Ollama manages its own model store, so we shell out to `ollama pull <tag>`.
  - vllm:   vLLM loads HF weights from the standard Hugging Face cache, so we snapshot-
            download each repo ONCE (via the base `huggingface_hub` dep -- no torch/vLLM
            needed just to cache). A later vLLM launch reuses the cached snapshot.

The host GPU is detected first; oversized models are skipped (vLLM) or flagged
(Ollama, which can offload to CPU). The plan/decision logic is pure and unit-testable;
the side-effecting `ollama_pull` / `hf_cache` are injectable.

Manifest entry (YAML, see `samples/configs/models_uk.yaml`):
  - name: <label>
    backend: ollama | vllm
    source: <ollama-tag> | <hf-repo-id>
    min_vram_gb: <int>      # rough floor to serve it on this hardware class
    notes: <free text>

Submodules: `base` (constants + type aliases), `manifest` (load candidate/serving targets),
`stores` (backend stores, reuse detection, disk preflight), `planning` (pure action/source-expansion
logic), `fetch` (the side-effecting pull/cache), and `run` (the `prepare_models` orchestrator).
"""
