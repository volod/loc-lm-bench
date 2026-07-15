# Goldset Leaderboard Quickstart Granular Commands

This is the command-chain layer of the goldset leaderboard track: the wrapper form
(`make quickstart-goldset`) and the track's purpose are in the
[Quick Start](quick-start.md); per-stage depth lives in
[Run RAG core](../benchmarking/run-rag-core.md), [platform matrix](../benchmarking/platform-matrix.md),
and [MLflow analysis](../benchmarking/mlflow-analysis.md).

The granular commands below are the same operations without the wrapper orchestration:

```sh
# Purpose: create or update the local Python environment.
# Default input: pyproject.toml extras from EXTRAS, plus prebuilt vLLM wheels on CUDA hosts.
# Output/result: .venv is ready for CLI, RAG, tracking, board, prep, vLLM, and test commands.
make venv

# Optional lean environment when vLLM is not needed.
VENV_INSTALL_VLLM=0 make venv

# Purpose: isolate all quickstart leaderboard artifacts.
# Default input: none.
# Output/result: run bundles, indexes, serving configs, MLflow, and board data stay under this root.
export DATA_DIR=.data/quickstart-leaderboard

# Purpose: detect the supported CUDA host tier and generate largest-per-tier serve/eval scripts.
# Default input: samples/config-example/manifest.yaml, current nvidia-smi GPU.
# Output/result: $DATA_DIR/llb/serving/gpu-<tier>gb/ with tier.json, serve scripts, run configs.
make detect-gpu-vram
make gen-serving-config

# Purpose: chunk and embed the committed fixture corpus into the default FAISS RAG store.
# Default input: CORPUS=samples/goldsets/ua_squad_postedited_v1/corpus.
# Output/result: chunk records, vector index, and store metadata under $DATA_DIR/llb/rag/.
make build-index

# Purpose: check whether retrieval can find gold source spans before model scoring.
# Default input: GOLDSET=committed fixture, RAG_K=10, index from $DATA_DIR/llb/rag/.
# Output/result: prints n, recall@10, MRR, and PASS or retrieval-bottleneck status.
make validate-retrieval

# Purpose: resolve and prepare candidate model families for this host.
# Default input: samples/configs/models_uk.yaml.
# Output/result: host fit table and pulled/cached runnable candidates. prep-models reuses any
# artifact already in its backend store and refuses a download up front when the cache filesystem
# lacks room for it (no failing an hour into a multi-GiB pull); --dry-run previews the disk plan.
make list-models
make prep-models

# Purpose: run one isolated evaluation cell per runnable candidate model and backend.
# Default input: samples/configs/models_uk.yaml, GOLDSET=committed fixture, SPLIT=final.
# Output/result: run bundles in $DATA_DIR/run-eval/ plus resume markers in
# $DATA_DIR/sweep/qs-committed/cells/; qs-committed is only the user-chosen sweep name.
make sweep SWEEP_ID=qs-committed

# Purpose: compare one logical model base across Ollama, vLLM, and llama.cpp with telemetry.
# Default input: committed fixture, current platform-matrix model defaults, LIMIT=20.
# Output/result: available backend rows under $DATA_DIR/run-eval/; missing vLLM/llama.cpp
# executables are logged as skips unless PLATFORM_MATRIX_STRICT=1.
make platform-matrix

# Purpose: turn the sweep into host-adaptive operator picks + a model-comparison chart.
# Default input: $DATA_DIR/run-eval/ final-split bundles, detected CUDA tier (RECOMMEND_GPU_GB= to
# simulate another tier; RECOMMEND_MIN_CASES= to drop partial runs).
# Output/result: best RAG accuracy, best efficiency (quality/W), best model for THIS host, RAG
# health, and $DATA_DIR/recommend/{summary.md,comparison.png}.
make recommend RECOMMEND_MIN_CASES=50

# Purpose: run security tests as a separate benchmark tier; do not mix ASR with RAG quality.
# Default input: samples/benchmarks/security_cases_uk.json, SECURITY_MODEL=MamayLM 27B GGUF,
# SECURITY_BACKEND=ollama.
# Output/result: ASR, defense rate, refusal-appropriateness, per-family ASR, and security bundle.
make bench-security

# Purpose: prepare prompt-system candidates, review/pin one, then compare final prompt runs.
# Default input: committed fixture corpus; reviewer supplies the pinned prompt id.
# Output/result: prompt package under $DATA_DIR/prompt-system/<run>/ and prompt comparison board.
make prompt-system-prepare PROMPT_SYSTEM_CORPUS=samples/goldsets/ua_squad_postedited_v1/corpus
make prompt-system-review PROMPT_SYSTEM_RUN_DIR=<prompt-run-dir> PROMPT_SYSTEM_ACTION=pin \
  PROMPT_SYSTEM_ID=<prompt-id>
make run-eval PROMPT_SYSTEM_ID=<prompt-id> PROMPT_PACKAGE=<prompt-run-dir>
make prompt-system-compare

# Purpose: inspect canonical run bundles in the local leaderboard UI.
# Default input: $DATA_DIR/run-eval/ plus screen, category, and prompt-system artifacts.
# Output/result: Streamlit serves http://127.0.0.1:8501 until stopped.
make board

# Purpose: inspect the MLflow mirror of canonical evaluation runs.
# Default input: $DATA_DIR/run-eval/ manifests and the local $DATA_DIR/mlflow/ store.
# Output/result: syncs and serves the loc-lm-bench MLflow UI at http://127.0.0.1:5000.
make mlflow
```
