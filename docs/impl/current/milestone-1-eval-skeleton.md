# Milestone 1 Current State

## Milestone 1 -- modules + how to run

The walking skeleton: one model, fixed config, retrieve -> generate -> score -> ranked row
+ manifest. It is compile-free (prebuilt Ollama, which still uses the GPU; no vLLM/flash-attn
source build). Every heavy collaborator (FAISS, sentence-transformers, langgraph, mlflow,
pyarrow, pynvml) is lazy-imported, so the base install imports the whole package; the real
run needs a running Ollama (the `[rag,eval]` deps are installed by `make venv`).

### The flow

    [gold set] --> retrieve (FAISS, pinned embedding)
                     |  recall@k / MRR vs source spans
                     |  (validates retrieval; not a rank axis)
                     v
                  generate (LangGraph node -> OpenAI-compatible chat -> Ollama)
                     |  classify: ok / empty / malformed / refusal / timeout /
                     |            backend_error / retrieval_miss
                     v
                  score: reference answer-correctness (objective)
                         [+ gated judge: scored into the blend only when JUDGE_RHO >= threshold,
                          else demoted to a diagnostic and objective ranks]
                     v
                  aggregate -> ranked row (Pareto tie-break: tok/s, then VRAM)
                     v
                  persist manifest.json + scores.{parquet,jsonl} FIRST, then
                  MLflow mirror

### Canonical run config -- `llb.config.RunConfig`
One Pydantic object flows through retrieval, generation, scoring, and the manifest, so a
run is reproducible from a single record. `RunConfig.load(path)` reads YAML (see
`samples/run_config_uk.yaml`); CLI flags override individual fields. Configuration forbids
unknown keys, validates numeric and cross-field chunking constraints, and revalidates every
CLI override. `llb.paths` loads the project `.env`, honors `DATA_DIR`, and resolves all
relative paths from the project root rather than the caller's current directory.

### CLI -- `llb` (`llb.main` -> `llb.cli`, Typer)

`main.py` is a thin entry point; the Typer app root is `llb.cli.app` and commands live in per-area
modules under `llb.cli/` (`eval`, `models`, `prep`, `rag`, `bench`, `inference`, `ui`, with shared
helpers in `helpers.py`). Heavy collaborators are lazy-imported at call time, so the package still
imports in the base install. Representative commands:

    llb prep-models # detect GPU; pull Ollama tags + cache vLLM weights
    llb list-models # which candidates can run here (GPU+RAM, context)
    llb build-index # chunk + embed the corpus -> FAISS store ([rag])
    llb build-index --strategy markdown --mode parent_child # structure-aware +
    parent-child
    llb validate-retrieval --k 10 # recall@k / MRR of the pinned embedding
    ([rag])
    llb run-eval --model llama3.2:3b # one ranked row + manifest (Ollama +
    [rag,eval])
    llb run-eval --config samples/run_config_uk.yaml --judge-rho 0.7 # records
    gate status
    llb run-eval --split calibration --worksheet ws.csv # pre-fill a
    calibration worksheet
    llb run-eval --score-semantic # also record semantic correctness

Or via make: `make prep-models`, `make build-index`, `make validate-retrieval`,
`make run-eval MODEL=... LIMIT=...`. The make targets default `GOLDSET` and `CORPUS` to the
committed post-edited public fixture, so they run without regeneration or network access;
override both for another set. A missing gold set or a set with no `verified: true` items in
the split fails with an actionable message rather than a raw traceback.

### Model preparation -- `llb.backends.{hardware,prepare}` (`prep-models`)
Reads a candidate-models manifest (`samples/models_uk.yaml`), detects the host GPU via
`nvidia-smi`, then prepares each model by backend: `ollama pull <tag>` (Ollama owns its
store) or a one-time Hugging Face `snapshot_download` for vLLM weights (uses the base
`huggingface_hub` dep -- no torch/vLLM needed just to cache; a gated repo needs `HF_TOKEN`).
Oversized models are skipped for vLLM and flagged for Ollama (which offloads to CPU);
`--force` overrides, `--dry-run` shows the plan. The plan/fit logic is pure and tested.

### Model feasibility planner -- `llb.backends.planner` (`list-models`)
Lists which candidate models can be benchmarked on THIS host, optimizing for ABILITY TO
RUN rather than speed. The memory budget is GPU VRAM + system RAM (detected via
`nvidia-smi` + `/proc/meminfo`); a model that does not fit in VRAM alone can still run by
splitting layers between GPU and CPU. For each model it estimates the weights footprint
(embedding-aware -- see M4.1 below), the KV cache per token (`2 x n_layers x kv_dim x 2B`, batch=1,
no parallelism), and reports the max context fully on GPU (`ctx_gpu`), the max context
using GPU+RAM offload (`ctx_max`), the GPU/CPU layer split, and a verdict
(gpu / offload / no). `--context N` plans for a fixed context instead of the maximum.
All values are planning estimates from `samples/models_uk.yaml`; the real fit test is a
launch (Milestone 2).

    make list-models # plan at the max context the host can hold
    make list-models CONTEXT=8192    # plan at a target context

### RAG store + retrieval metrics -- `llb.rag.{store,embedding,index,retrieval}`
`RagStore.build` chunks the corpus (reusing `rag.chunking`), embeds with the PINNED
`Embedder` (e5 query/passage prefixes applied), and indexes with a FAISS inner-product
index; `.retrieve(question, k)` returns chunk dicts (doc id + char offsets). `retrieval`
scores recall@k / MRR by SOURCE-SPAN overlap -- it validates the embedding (Premise 4,
recall@10 >= 0.8) and is reported as context, never as a model-ranking axis.

Two retrieval modes (`--mode`): `flat` indexes the `chunk_size` chunks directly;
`parent_child` indexes small `child_chunk_size` children for precise matching but returns
their larger PARENT chunk for generation context (retrieve a child -> surface its parent,
deduped). Both return offset-bearing chunks, so the span metric is mode-agnostic and Optuna
can compare flat vs parent-child.

### Backends -- `llb.backends.{base,openai_client,ollama,vllm}`
`BackendLauncher` is the seam (Premise 1): all backends speak OpenAI-compatible HTTP, so
only the launcher + telemetry hook are backend-specific. `openai_client.chat_once` maps
transport failures to normalized tokens (`timeout` / `backend_error`). M1 ships the prebuilt
`OllamaLauncher`; M2 adds `VllmLauncher` (M2.1) -- it starts `vllm serve <model>` as a
subprocess (controlling + recording `gpu-memory-utilization` / `max-model-len`), waits for
readiness, serves chat through the same `chat_once`, and kills the server on stop. It is a
subprocess CLI, so the module imports in the base install and is tested by injecting the
process factory + HTTP probe (no vLLM/CUDA needed). llama.cpp slots in the same way later.
The launcher seeds the subprocess env via `launch_env`, which defaults
`VLLM_USE_FLASHINFER_SAMPLER=0` (only when unset, so an explicit value wins): flashinfer
JIT-compiles a sampling kernel at startup that fails to build on consumer CUDA toolchains
(its `sampling.cuh` calls `cub::BlockAdjacentDifference::FlagHeads`, removed from newer
CCCL/CUB), and greedy decoding does not need it. When a launch fails, the runner preserves
the backend's startup log to `$DATA_DIR/llb/logs/failed-*.log` before discarding the staging
bundle, so a dead engine stays diagnosable.

### Eval graph -- `llb.eval.graph`
A LangGraph retrieve -> generate flow (the first of the ~3 DRY templates). The node
closures and `classify_response` are pure and unit-tested; only `build_rag_graph` imports
langgraph. Each case ends in exactly one typed status, recorded separately.

### Scoring -- `llb.scoring.{correctness,judge,aggregate}`
`correctness` ranks models by reference answer-correctness (exact / token-F1 / contains,
Unicode-normalized for casing and punctuation); `score` is token-F1. An optional
semantic-similarity signal (cosine via the pinned embedder) captures paraphrases and UA
morphology when `--score-semantic` is set -- it is recorded separately because blending
weights require calibration. `judge` enforces the gate (Premise 2): the DeepEval G-Eval judge
only may enter aggregate ranking at calibration rho >= threshold; below it, objective score
ranks alone. `run-eval` invokes it when configured and trusted, records per-case scores, and
keeps the row objective-only otherwise. `aggregate` produces the ranked row
(quality, then tok/s, then VRAM; infeasible models listed without a rank).

### Tracking -- `llb.tracking.manifest`
The immutable `manifest.json` + per-case `scores.{parquet,jsonl}` are written FIRST; the
MLflow mirror runs after, best-effort, and a mirror failure never loses a completed run.
All runs share the local SQLite store and artifact root under `$DATA_DIR/mlflow/`, enabling
cross-run comparison without putting mutable MLflow state inside immutable run bundles.
`make mlflow` serves that store locally at `http://127.0.0.1:5000`; override its bind address
or port with `MLFLOW_HOST` and `MLFLOW_PORT`. Before serving, it idempotently reconciles all
canonical run directories: missing records are created and old mirror schemas are enriched
with grouped quality/retrieval/telemetry/hardware/judge metrics, unique run names, canonical
run-id tags, and the manifest plus per-case scores under the `canonical/` artifact path. See
the [MLflow analysis guide](../../guides/mlflow-analysis.md).
Parquet when `pyarrow` ([track]) is present, JSONL otherwise. The full run bundle, including
backend logs, is assembled in a hidden sibling staging directory and atomically renamed to
its final timestamped directory only after both canonical files succeed. Failed writes leave
no partially published run, and existing canonical artifacts are never overwritten.

### Executor -- `llb.executor.{cases,reporting,runner,vram}`
`vram` is the basic NVML reclaim gate (injectable reader; raises `VramNotReclaimed` when
freed VRAM stays above tolerance). `runner.run_eval` orchestrates the single-model run;
every heavy collaborator is injectable, so the whole vertical runs end to end in a unit
test with fakes (`tests/test_runner.py`). The runner filters out unverified gold items,
separates case execution, telemetry, aggregation, persistence, and reporting, and uses the
steady-state telemetry rate in the leaderboard when telemetry is enabled.

### Typed contracts and enforced formatting
`llb.contracts` defines the records crossing package boundaries: chunks and source spans,
retrieval and telemetry metrics, model manifests and plans, case scores, leaderboard rows,
and persisted run paths. External YAML model entries are validated by Pydantic before being
converted to those contracts. Mypy checks all production modules with generic type arguments
required, while Ruff formatting and linting are enforced by `make ci` and GitHub Actions.

### Milestone 1 status

- **M1.1** (`RunConfig` + Typer CLI (`build-index`, `validate-retrieval`, `run-eval`)): DONE
- **M1.2** (pinned-embedding FAISS RAG store (`build-index`)): DONE
- **M1.3** (recall@k / MRR by source-span overlap): DONE
- **M1.4** (LangGraph retrieve->generate over Ollama + typed failure taxonomy): DONE
- - **M1.5** (objective answer-correctness (+ semantic) + judge gate seam): CORE (executor judge
- wiring -> M3.8)
- **M1.6** (canonical manifest + scores, MLflow mirror): DONE
- **M1.7** (minimal sequential runner + NVML VRAM gate): DONE
- **M1.8** (`run-eval` prints one ranked row (SQuAD-uk seed)): DONE

Residual M1 work is scoped forward in [`plan.md`](../plan.md): human judge calibration (M3.8). The
map-reduce / multi-hop eval templates (M1.4-rest) are now DELIVERED under M5.0 (see Milestone 5
below). The optional semantic-similarity correctness signal is built (`--score-semantic`).
