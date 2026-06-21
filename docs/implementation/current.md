# loc-lm-bench — Implemented (current state)

A snapshot of what exists and runs **today**. Forward work lives in
[`plan.md`](plan.md); the full spec is [`spec.md`](../design/spec.md).

**Status:**
- **Milestone 0 (data prep) complete:** schema, validator, disjoint splits, SQuAD
  ingestion, a real 250-item Ukrainian gold set (HPLT/ua-squad), judge-calibration
  stats, a chunking RAG-store builder.
- **Milestone 1 (eval skeleton) complete:** a canonical `RunConfig` + Typer
  CLI, a pinned-embedding FAISS RAG store + source-span retrieval metrics, a LangGraph
  retrieve -> generate flow over an OpenAI-compatible backend (Ollama), objective
  answer-correctness + a gated judge, a canonical manifest + scores record (MLflow
  mirror), and a minimal sequential runner with a VRAM gate. `llb run-eval` prints one
  ranked row. The skeleton is **compile-free** -- prebuilt Ollama (which uses the GPU),
  no vLLM/flash-attn source build -- so the loop is proven before that build (M2).
- **Milestone 2 (real backend + telemetry) complete:** a vLLM launcher (serves HF weights
  behind the same OpenAI-compatible interface), a per-backend telemetry hook (steady-state
  tokens/sec, peak VRAM, served vs requested context, load time, tokenizer efficiency), and a
  MAX_JOBS-capped vLLM build script -- now **validated end to end on a real model**:
  `google/gemma-4-E4B-it-qat-w4a16-ct` served via vLLM 0.23.0 on the RTX 4060 Ti 16 GB,
  scored under the executor with real telemetry (~64 tok/s, peak VRAM 15.7 GB, cold load
  ~112 s, served ctx 8192). See the [vLLM guide](../guides/vllm-backend.md).

Two host-aware model utilities: `prep-models` prepares candidate models (pulls Ollama
tags, caches vLLM Hugging Face weights once), and `list-models` reports which candidates
can actually run here (GPU VRAM + system RAM, KV-cache-aware, with a GPU/CPU layer split).

220 tests passing; Ruff format/lint and mypy are clean. CI enforces formatting, linting,
static typing, and unit tests only (no GPU / network / heavy extras); every heavy dependency
is lazy-imported so the base install stays importable.

## Dev setup

Requires [`uv`](https://docs.astral.sh/uv/). `make venv` creates `.venv` (Python 3.11),
installs the `llb` package editable with ALL extras, and seeds `.env` from `.env.example`,
so a fresh checkout can run every command without a follow-up `uv pip install`.

    make            # list targets
    make venv       # .venv (py3.11) + package + all extras + .env (idempotent; RECREATE_VENV=1 to rebuild)
    make test       # pytest (220 tests)
    make format     # apply canonical Ruff formatting to src/ and tests/
    make ci         # format check + lint + mypy + tests
    make demo-eval  # idempotent end-to-end: venv -> gold set -> index -> validate -> prep-models -> run-eval+telemetry
    make mlflow     # review all mirrored experiment runs at http://127.0.0.1:5000

`make demo-eval` runs the whole pipeline in order and is **idempotent** -- the venv is reused
(deps updated; `RECREATE_VENV=1` to rebuild), an existing gold set / index is reused, and
cached model downloads are skipped. It tees per-step output to `.data/llb/logs/pipeline-
<ts>.log` and, on failure, names the failing step + log path. It needs a running Ollama for
the final telemetry run. Every command shares one runtime (`llb.runtime`): Ctrl-C shuts down
cleanly (exit 130, backends killed via their context manager) and an unexpected crash is
logged with a traceback (`LLB_LOG=debug` for more) instead of a raw stack dump.

Extras (`rag, eval, track, board, prep, telemetry, goldset, dev`) are all installed by
`make venv`; trim with `EXTRAS=` (e.g. `make venv EXTRAS=rag,eval`). GitHub CI installs
only `.[dev]` (it never runs `make venv`), so it stays light. vLLM/torch/flash-attn are
hardware-matched and installed via a separate path per AGENTS.md, never declared here.

Gitignored: `.data/` (runtime output), `.env` (secrets), `.venv/`.

## Repo layout (current)

    pyproject.toml                 # package "llb": deps + extras, pytest/ruff config
    Makefile                       # venv, test, gen-rag-items, validate-goldset,
                                   #   ingest-squad, ingest-uk-squad, build-rag-store, calibration-worksheet
    .env.example                   # DATA_DIR + frontier-API key placeholders
    samples/                       # COMMITTED DATA (kept separate from code)
      rag_items_uk.json            #   sample RAG spec: source docs + item defs
      squad_uk_fixture.json        #   SQuAD-format UA fixture (tests/demo)
      corpus/ip_regulation_uk.md   #   substantial UA domain doc (IP regulation) for chunking
    scripts/
      shared/common.sh             # shared bootstrap + canonical max_jobs() helper (AGENTS.md)
      gen_rag_items.sh             # thin entrypoint -> llb.prep.gen_rag_items
      build_vllm.sh                # uv-shared prebuilt install or one checkout-built wheel
    src/llb/
      build/vllm.py               # uv-shared installer + checkout wheel builder
      config.py                    # RunConfig (Pydantic) -- the canonical run config
      contracts.py                 # shared TypedDict boundary contracts
      paths.py                     # project root, .env, and DATA_DIR path resolution
      main.py                      # Typer CLI: build-index, validate-retrieval, run-eval
      runtime.py                   # shared CLI runtime: graceful Ctrl-C (exit 130) + crash logging
      goldset/schema.py            # GoldItem + SourceSpan (Pydantic), load/dump
      goldset/splits.py            # deterministic disjoint split assignment
      goldset/validate.py          # corpus-grounded validator + CLI
      prep/gen_rag_items.py        # spec -> seed gold set
      prep/ingest_squad.py         # SQuAD-format (local or HF) -> canonical gold items
      judge/calibration.py         # Spearman rho + CI + trust decision + worksheet
      rag/chunking.py              # fixed/sentence/recursive/markdown/semantic chunking (offset-exact)
      rag/{embedding,index,store}.py  # pinned embedder + FAISS index + store (flat / parent-child)
      rag/retrieval.py             # recall@k / MRR by source-span overlap (pure)
      backends/{base,openai_client,ollama,vllm}.py  # launcher iface + chat call + Ollama + vLLM
      backends/{hardware,prepare,planner,telemetry}.py  # GPU/RAM detect + pull/cache + plan + telemetry
      eval/graph.py                # LangGraph retrieve->generate flow + failure taxonomy
      scoring/{correctness,judge,aggregate}.py  # objective + semantic + gated judge + N-model board
      tracking/{manifest,mlflow,server}.py  # canonical artifacts + MLflow mirror/UI
      executor/{cases,reporting,runner,vram,isolation}.py  # per-case work + reporting + sweep isolation
      backends/resolver.py         # M3.2 AvailabilityResolver (discovery + backend priority + fit)
      optimize/tuner.py            # M3.4 two-stage Optuna (tuning-split search -> stage-2 entry)
      screen/public.py             # M3.1 Tier-1 lm-eval-harness-uk adapter (logprob/generation tracks)
      prep/frontier.py             # M3.5 prepare-goldset + prepare-synthetic-corpus (litellm)
      board/{data,app}.py          # M3.7 thin Streamlit leaderboard over the run bundles
    tests/                         # 220 tests across the above

Shared runtime data is gitignored under `$DATA_DIR/llb/` (default `.data/llb/`):
`corpus/`, `goldset/*.jsonl`, `rag/` (chunks + FAISS index), and
`calibration_worksheet.csv`. Immutable eval artifacts are isolated per invocation under
`$DATA_DIR/run-eval/<UTC timestamp>-<run id>/` (`manifest.json`,
`scores.{parquet,jsonl}`, and optional `vllm/` logs).

## Milestone 0 -- modules + how to run

### Canonical gold-item schema — `llb.goldset.schema`
Pydantic `GoldItem` + `SourceSpan`. Fields: `id, lang, question, reference_answer,
source_doc_id, source_spans[{doc_id, char_start, char_end, text}], provenance, verified,
split`. Labels are SOURCE-SPAN (char offsets, not chunk ids), so they survive `chunk_size`
tuning. `provenance` and `split` are enforced literals. Only `verified: true` items score
models. `load_goldset` / `dump_goldset` handle JSONL (UTF-8).

### Splits — `llb.goldset.splits`
`assign_splits(ids, ratios, seed)` -> deterministic, disjoint `calibration / tuning / final`.

### Validator (M0 acceptance) — `llb.goldset.validate`
Checks every span resolves to its labeled text on disk, ids unique, splits disjoint.

    make validate-goldset          # PASS on the sample set

### Sample generator — `llb.prep.gen_rag_items`
Reads `samples/rag_items_uk.json`, computes spans, writes + validates a seed gold set. Its
six synthetic, hand-authored demo fixtures are explicitly verified so `make demo-eval` can
score them; imported public datasets remain unverified until human review.

    make gen-rag-items             # -> .data/llb/goldset/sample_rag_items.jsonl (6 items)

### SQuAD ingestion (M0.3) — `llb.prep.ingest_squad`
Maps SQuAD-format UA QA (flattened, nested, or HF rows where `answers` is a dict-string) ->
canonical items (`provenance: public-reused`, `verified: false`), span from the answer
offset with a `find()` fallback. Local file or HF dataset (streams when `--max-items` set).

    make ingest-uk-squad                       # HPLT/ua-squad -> 250-item real gold set
    make ingest-squad                          # the bundled fixture (4 items)
    make ingest-squad SQUAD_JSON=path.json     # a local SQuAD-uk export
    python -m llb.prep.ingest_squad --hf-dataset <id> --hf-split train   # needs HF_TOKEN (goldset extra via make venv)

The current real set is `.data/llb/goldset/goldset_uk.jsonl` (250 items, splits
cal=86/tun=82/fin=82, 239 corpus docs). All `verified: false` pending human review.

### RAG chunking / store builder — `llb.rag.chunking`
Five strategies, every chunk anchored to `doc_id` + char offsets so retrieval scores against
source-span gold labels. We reuse the langchain family where it preserves offsets, and roll
our own where it does not (the span metric is the hard constraint):
- `fixed`, `sentence` -- pure-Python (zero deps), the always-available fallbacks.
- `recursive` -- langchain `RecursiveCharacterTextSplitter` (`add_start_index` -> exact
  offsets); falls back to the pure paragraph->sentence->char split when `[rag]` is absent.
- `markdown` -- structure-aware: headers parsed from the SOURCE (offset-exact), header
  breadcrumbs recorded in chunk `metadata`, long sections sub-split recursively.
- `semantic` -- native: embed sentences with the PINNED embedder, break at distance spikes
  (offset-exact -- langchain's `SemanticChunker` rejoins text and loses offsets, so we do
  not use it). Needs the embedder (`[rag]`).

All langchain use is lazy; `fixed` / `sentence` / `markdown` work without `[rag]`. `--embed`
(with `[rag]`) also builds a per-strategy FAISS index.

    make build-rag-store                       # chunk samples/corpus, all strategies
    python -m llb.rag.chunking --corpus-root <dir> --out-dir .data/llb/rag \
        --strategy markdown --size 800 --overlap 120 [--embed]

On the bundled IP doc: recursive 10 / markdown 8 chunks (markdown carries h1/h2 breadcrumbs).

### Judge calibration (M0.5 stats) — `llb.judge.calibration`
Spearman rho (no scipy), bootstrap CI, trust decision (`rho >= 0.6` else demote). Two
worksheet emitters: a blank one, and a pre-filled one driven from a run, so the human only
adds `human_rating`.

    make calibration-worksheet                            # blank worksheet from the calibration split
    llb run-eval --split calibration --worksheet ws.csv   # worksheet with model answers pre-filled
    python -m llb.judge.calibration score --ratings ws.csv   # rho + CI + decision

## Milestone 0 status

| Step | What | State |
|------|------|-------|
| M0.1 schema | Pydantic `GoldItem` / `SourceSpan` | DONE |
| M0.2 sample generator | `gen_rag_items` + sample spec | DONE |
| M0.3 real gold set | `ingest_squad` + 250 items from HPLT/ua-squad | DONE |
| M0.4 splits | deterministic disjoint partition | DONE |
| M0.5 calibration stats | rho + CI + blank/pre-filled worksheet | DONE (code) |
| chunking | fixed/sentence/recursive RAG-store builder | DONE |
| acceptance | validator PASS (sample + fixture + 250-item set), suite green | DONE |

Remaining (blocked on a judge choice or human input; scoped forward in [`plan.md`](plan.md)):
- **Judge-calibration close-out (plan M3.8):** the stats, the gate, and a pre-filled
  worksheet (`run-eval --split calibration --worksheet`) all exist. Closing the loop is
  blocked on choosing the judge (OQ2) + the Ragas scorer + human ratings; then
  `calibration score` gates at rho >= 0.6 and `run-eval --judge-rho` lets the judge in.
- **Human verification + screen datasets (plan M3.9):** the 250 public-reused items are
  `verified: false` pending human review; Belebele-uk (MCQ) wires into the Tier-1 screen,
  not the source-span gold set.

## Milestone 1 -- modules + how to run

The walking skeleton: one model, fixed config, retrieve -> generate -> score -> ranked row
+ manifest. It is compile-free (prebuilt Ollama, which still uses the GPU; no vLLM/flash-attn
source build). Every heavy collaborator (FAISS, sentence-transformers, langgraph, mlflow,
pyarrow, pynvml) is lazy-imported, so the base install imports the whole package; the real
run needs a running Ollama (the `[rag,eval]` deps are installed by `make venv`).

### The flow

    [gold set] --> retrieve (FAISS, pinned embedding)
                     |  recall@k / MRR vs source spans (validates retrieval; not a rank axis)
                     v
                  generate (LangGraph node -> OpenAI-compatible chat -> Ollama)
                     |  classify: ok / empty / malformed / refusal / timeout /
                     |            backend_error / retrieval_miss
                     v
                  score: reference answer-correctness (objective) [+ gated judge]
                     v
                  aggregate -> ranked row (Pareto tie-break: tok/s, then VRAM)
                     v
                  persist manifest.json + scores.{parquet,jsonl} FIRST, then MLflow mirror

### Canonical run config — `llb.config.RunConfig`
One Pydantic object flows through retrieval, generation, scoring, and the manifest, so a
run is reproducible from a single record. `RunConfig.load(path)` reads YAML (see
`samples/run_config_uk.yaml`); CLI flags override individual fields. Configuration forbids
unknown keys, validates numeric and cross-field chunking constraints, and revalidates every
CLI override. `llb.paths` loads the project `.env`, honors `DATA_DIR`, and resolves all
relative paths from the project root rather than the caller's current directory.

### CLI — `llb` (`llb.main`, Typer)

    llb prep-models                         # detect GPU; pull Ollama tags + cache vLLM weights
    llb list-models                         # which candidates can run here (GPU+RAM, context)
    llb build-index                         # chunk + embed the corpus -> FAISS store ([rag])
    llb build-index --strategy markdown --mode parent_child   # structure-aware + parent-child
    llb validate-retrieval --k 10           # recall@k / MRR of the pinned embedding ([rag])
    llb run-eval --model llama3.2:3b        # one ranked row + manifest (Ollama + [rag,eval])
    llb run-eval --config samples/run_config_uk.yaml --judge-rho 0.7
    llb run-eval --split calibration --worksheet ws.csv   # pre-fill a calibration worksheet
    llb run-eval --score-semantic                         # also record semantic correctness

Or via make: `make prep-models`, `make build-index`, `make validate-retrieval`,
`make run-eval MODEL=... LIMIT=...`. `make run-eval` defaults `GOLDSET=` to the verified
sample set (seeded by `make gen-rag-items` / `make demo-eval`) so it runs out of the box;
override `GOLDSET=` for another set. A missing gold set or a set with no `verified: true`
items in the split fails with an actionable message rather than a raw traceback.

### Model preparation — `llb.backends.{hardware,prepare}` (`prep-models`)
Reads a candidate-models manifest (`samples/models_uk.yaml`), detects the host GPU via
`nvidia-smi`, then prepares each model by backend: `ollama pull <tag>` (Ollama owns its
store) or a one-time Hugging Face `snapshot_download` for vLLM weights (uses the base
`huggingface_hub` dep -- no torch/vLLM needed just to cache; a gated repo needs `HF_TOKEN`).
Oversized models are skipped for vLLM and flagged for Ollama (which offloads to CPU);
`--force` overrides, `--dry-run` shows the plan. The plan/fit logic is pure and tested.

### Model feasibility planner — `llb.backends.planner` (`list-models`)
Lists which candidate models can be benchmarked on THIS host, optimizing for ABILITY TO
RUN rather than speed. The memory budget is GPU VRAM + system RAM (detected via
`nvidia-smi` + `/proc/meminfo`); a model that does not fit in VRAM alone can still run by
splitting layers between GPU and CPU. For each model it estimates the weights footprint
(params x bits-per-weight), the KV cache per token (`2 x n_layers x kv_dim x 2B`, batch=1,
no parallelism), and reports the max context fully on GPU (`ctx_gpu`), the max context
using GPU+RAM offload (`ctx_max`), the GPU/CPU layer split, and a verdict
(gpu / offload / no). `--context N` plans for a fixed context instead of the maximum.
All values are planning estimates from `samples/models_uk.yaml`; the real fit test is a
launch (Milestone 2).

    make list-models                 # plan at the max context the host can hold
    make list-models CONTEXT=8192    # plan at a target context

### RAG store + retrieval metrics — `llb.rag.{store,embedding,index,retrieval}`
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

### Backends — `llb.backends.{base,openai_client,ollama,vllm}`
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

### Eval graph — `llb.eval.graph`
A LangGraph retrieve -> generate flow (the first of the ~3 DRY templates). The node
closures and `classify_response` are pure and unit-tested; only `build_rag_graph` imports
langgraph. Each case ends in exactly one typed status, recorded separately.

### Scoring — `llb.scoring.{correctness,judge,aggregate}`
`correctness` ranks models by reference answer-correctness (exact / token-F1 / contains,
Unicode-normalized for casing and punctuation); `score` is token-F1. An optional
semantic-similarity signal (cosine via the pinned embedder) captures paraphrases and UA
morphology when `--score-semantic` is set -- it is recorded separately because blending
weights require calibration. `judge` enforces the gate (Premise 2): the Ragas judge only
enters the blend at calibration rho >= threshold, else it is demoted and the objective score
ranks alone. `aggregate` produces the ranked row (quality, then tok/s, then VRAM; infeasible
models listed without a rank).

### Tracking — `llb.tracking.manifest`
The immutable `manifest.json` + per-case `scores.{parquet,jsonl}` are written FIRST; the
MLflow mirror runs after, best-effort, and a mirror failure never loses a completed run.
All runs share the local SQLite store and artifact root under `$DATA_DIR/mlflow/`, enabling
cross-run comparison without putting mutable MLflow state inside immutable run bundles.
`make mlflow` serves that store locally at `http://127.0.0.1:5000`; override its bind address
or port with `MLFLOW_HOST` and `MLFLOW_PORT`. Before serving, it idempotently reconciles all
canonical run directories: missing records are created and old mirror schemas are enriched
with grouped quality/retrieval/telemetry/hardware/judge metrics, unique run names, canonical
run-id tags, and the manifest plus per-case scores under the `canonical/` artifact path. See
the [MLflow analysis guide](../guides/mlflow-analysis.md).
Parquet when `pyarrow` ([track]) is present, JSONL otherwise. The full run bundle, including
backend logs, is assembled in a hidden sibling staging directory and atomically renamed to
its final timestamped directory only after both canonical files succeed. Failed writes leave
no partially published run, and existing canonical artifacts are never overwritten.

### Executor — `llb.executor.{cases,reporting,runner,vram}`
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

| Step | What | State |
|------|------|-------|
| M1.1 | `RunConfig` + Typer CLI (`build-index`, `validate-retrieval`, `run-eval`) | DONE |
| M1.2 | pinned-embedding FAISS RAG store (`build-index`) | DONE |
| M1.3 | recall@k / MRR by source-span overlap | DONE |
| M1.4 | LangGraph retrieve->generate over Ollama + typed failure taxonomy | DONE |
| M1.5 | objective answer-correctness (+ semantic) + gated judge | DONE (Ragas scorer -> M3.8) |
| M1.6 | canonical manifest + scores, MLflow mirror | DONE |
| M1.7 | minimal sequential runner + NVML VRAM gate | DONE |
| M1.8 | `run-eval` prints one ranked row (SQuAD-uk seed) | DONE |

Residual M1 work is scoped forward in [`plan.md`](plan.md): the Ragas judge scorer + UA
metric-prompt localization (M3.8, needs the judge choice + calibration ratings) and the
map-reduce / multi-hop eval templates (deferred until the text-analysis benchmark needs
them). The optional semantic-similarity correctness signal is now built (`--score-semantic`).

## Milestone 2 -- real backend + telemetry (complete)

A real vLLM backend behind the same interface, a steady-state telemetry hook, and the
MAX_JOBS-capped build entrypoint -- validated end to end on a real model (see the
[vLLM guide](../guides/vllm-backend.md) and `samples/run_config_vllm_uk.yaml`).

### vLLM launcher — `llb.backends.vllm` (M2.1)
`VllmLauncher` + `build_vllm_command` (pure). Documented under Backends above (incl. the
`launch_env` flashinfer-sampler default and the on-failure log preservation). The thin
`scripts/build_vllm.sh` entrypoint sources `scripts/shared/common.sh`, exports its canonical
`max_jobs()` result (`min(cores//2, RAM_GiB//14)`, AGENTS.md), and delegates to
`llb.build.vllm`. The default binary-only install and all ordinary dependencies use uv's
shared cache. Only a wheel built from `VLLM_SOURCE_DIR=<clean-git-checkout>` is exported
under `$DATA_DIR/wheels/vllm_<abi-key>_git<revision>/`. Weights are cached by `prep-models`.

    make build-vllm                                   # prebuilt wheel via uv shared cache
    VLLM_SOURCE_DIR=../vllm make build-vllm           # one ABI-keyed checkout wheel
    make prep-models PREP_BACKEND=vllm                # cache HF weights (verifies repo ids)
    llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry   # the M2.4 run

### Telemetry hook — `llb.backends.telemetry` (M2.2)
`measure_throughput` runs the steady-state protocol (fixed UA prompt set + fixed
max_new_tokens + N warmup iters) over `launcher.chat`, so tokens/sec is comparable across
models; cold-start `load_time_s` is recorded separately by launchers that own the backend
lifecycle, and remains null for an already-running external daemon such as Ollama.
`VramSampler` polls NVML (injected reader) for peak VRAM. `collect_telemetry` assembles the manifest record:
steady tokens/sec, tokenizer efficiency (tokens/UA-char), peak VRAM, requested-vs-served
context, load time, gpu-memory-utilization, and detected GPU. Wired into `run-eval`
behind `config.measure_telemetry` (`--telemetry`); recorded under `manifest.telemetry`.

### M2.4 real-model validation (RTX 4060 Ti 16 GB)
`google/gemma-4-E4B-it-qat-w4a16-ct` served via vLLM 0.23.0 and scored under the executor
produced a real ranked row + full telemetry: objective quality 0.801, **63.8 tok/s** steady,
peak VRAM **15.7 GB** (at gpu-memory-utilization 0.80), cold load **112 s**, served context
8192, tokenizer 0.33 tok/UA-char. vLLM resolves `Gemma4ForConditionalGeneration` +
`compressed-tensors` natively; attention falls back to TRITON (Gemma-4 heterogeneous head
dims), the flashinfer sampler is disabled (see `launch_env`), and `max_model_len` is capped
so the KV cache fits (the native 131072 window would over-reserve and fail startup).

Planner-vs-measured fit: the model's **weights load 9.8 GiB**, ~2.3x the planner's ~4.2 GiB
estimate (`params_b x bpw`). w4a16 quantizes only the linear layers while Gemma's 256k-token
embedding stays high-precision, so `list-models` under-estimates w4a16 weights. The measured
floor + caveat are recorded in `samples/models_uk.yaml`; an embedding-aware estimator is
forward work ([`plan.md`](plan.md) Milestone 4).

### Milestone 2 status

| Step | What | State |
|------|------|-------|
| M2.1 | `VllmLauncher` + `build_vllm_command` + MAX_JOBS build helper / script | DONE |
| M2.2 | telemetry hook (steady tokens/sec, peak VRAM, served ctx, load time, tok/char) | DONE |
| M2.3 | candidate list in `samples/models_uk.yaml`; vLLM repo ids verified via `prep-models` | DONE |
| M2.4 | validated on a real vLLM-served model (gemma-4-E4B-it-w4a16) w/ real telemetry | DONE |

Residual (non-blocking, forward in [`plan.md`](plan.md) Milestone 4): an embedding-aware VRAM
estimate for w4a16/int4 (the 2.3x weight under-estimate above), a pre-launch VRAM-contention
guard, and surfacing the vLLM serving knobs as `run-eval` CLI flags.

## Milestone 3 -- two-tier + scale + rigor (code complete; close-outs human-gated)

All nine M3 modules are built and unit-tested; the only remaining work is non-code: the judge
calibration close-out (needs the judge choice OQ2 + human ratings) and the gold-set human
verification (flip `verified: true` on reviewed items). The CLI grew `resolve-models`, `sweep`,
`tune`, `prepare-goldset`, `prepare-synthetic-corpus`, `screen-public`, and `board`.

### AvailabilityResolver -- `llb.backends.resolver` (M3.2)
`resolve(spec, vram, ram)` picks the backend that can actually serve a model on THIS host:
it adds DISCOVERY (does each source exist?) and a PRIORITY+FIT decision on top of the
feasibility planner. For each candidate `(backend, source)` -- the single declared backend or
a `sources: {backend: source}` map for the same logical model across backends -- it probes
availability (vLLM -> HF repo exists, Ollama -> tag pulled/in library, llama.cpp -> repo has a
`*.gguf`), then plans the fit and chooses by the fixed order vLLM > Ollama > llama.cpp. Fit is
offload-aware: vLLM must hold a serving window (`MIN_SERVING_CTX`, default 2048) fully on GPU
(`ctx_gpu`), while Ollama / llama.cpp may split layers to CPU RAM (`ctx_max`) -- so an oversized
bf16 model that vLLM cannot offload resolves to its GGUF on Ollama, exactly the rule the model
notes describe. Judging fit at a serving context (not the host max) is what keeps gemma-4-E4B
on vLLM even though it needs a sliver of offload at 131072. Every probe is injectable
(`ResolverProbes`), so the decision logic is pure and unit-tested without network.

    llb resolve-models                       # chosen backend per candidate (live probes)
    llb resolve-models --offline             # skip probes; assume declared sources exist
    llb resolve-models --context 8192        # resolve fit at a target context

On this host it resolves gemma-4-E4B / gemma-4-12B (w4a16) to vLLM and llama3.2-3b to Ollama;
the bf16/fp8 UA models resolve to nothing until a GGUF/Ollama `source` is declared for them.
Residual: each spec carries one `quant`, so per-source quant (vLLM bf16 vs Ollama q4) is not
yet modeled; the live HF/Ollama probes are not exercised in CI.

### N-model board rigor -- `llb.scoring.aggregate` (M3.6)
`rank_board` generalizes the single-model ranker to N models with four guards against
weight-gaming and noise-driven flips:
- **Average-rank headline.** Models are ranked on each shared quality signal (objective
  always; the gated judge only when trusted AND present for all; semantic only when present
  for all), and the per-signal ranks are averaged (`average_ranks`). This is robust to the
  arbitrary judge weight -- two models can tie on average rank even when a weighted blend
  would order them. The weighted blend (`headline_quality`) is kept as the tie-breaker view.
- **Confidence intervals.** `bootstrap_mean_ci` puts a percentile bootstrap CI on each model's
  per-case objective scores; adjacent models whose CIs overlap are flagged `unresolved` (the
  rank flip is not statistically resolved).
- **Pareto front.** `pareto_front` marks models not dominated on (quality up, tokens/sec up,
  peak VRAM down).
- **No tier mixing.** `rank_board` raises if asked to rank Tier-1 `screen` and Tier-2
  `private` results in one board (`TIER_SCREEN` / `TIER_PRIVATE` on `ModelResult`).
`format_board` renders it as ASCII (`*` = Pareto, `~` = CI-overlap/unresolved). The M1
`rank_results` / `format_table` single-row path is unchanged and still used by `run-eval`.

### Hard-isolation sweep -- `llb.executor.isolation` (M3.3)
`run_sweep(configs)` runs one (model, config) cell per PROCESS so a leak or crash in one cell
cannot bias the next: the default `CellRunner` shells out to `python -m llb.main run-eval
--config <cell> --split <s>`, so the vLLM server AND the whole CUDA context die with the cell.
Between cells it gates two things and records a third:
- **VRAM reclaim gate** (`executor.vram.assert_reclaimed`): wait for used VRAM to return to the
  pre-cell baseline within tolerance, else raise `VramNotReclaimed` and abort the whole sweep
  (a leak would bias every later cell). It runs only for `GATE_BACKENDS` (vLLM / llama.cpp) that
  own their VRAM; Ollama keeps weights warm by design, so gating it would falsely abort.
- **Thermal cooldown** (`cool_down`): wait until the hottest GPU is <= a threshold, capped at a
  max wait so a warm room cannot stall the sweep; throughput is only comparable at like clocks.
- **GPU telemetry** (`sample_gpu` via nvidia-smi): temp / power / SM+mem clocks per cell.
The sweep is RESUMABLE: each cell has a stable `cell_key` (a hash of its reproducibility-
relevant config, ignoring `run_name`) and writes a marker under `$DATA_DIR/sweep/<id>/cells/`,
so a re-run skips finished cells. Every side-effect (subprocess, NVML reader, GPU sampler,
sleep) is injectable; 10 unit tests cover it without a GPU. New CLI `sweep` resolves each
manifest model to a backend (M3.2) and runs the isolated cells:

    llb sweep --goldset .data/llb/goldset/sample_rag_items.jsonl --sweep-id run1   # run
    llb sweep --sweep-id run1                                                       # resume (skips done)

Validated on this host: an Ollama cell ran as a subprocess, recorded GPU temp, and a re-run
skipped it (resume). Residual: the sweep generates one cell per model at the default RAG
config; the RAG-parameter search space is driven by Optuna (M3.4).

### Two-stage Optuna tuning -- `llb.optimize.tuner` (M3.4)
`two_stage(base_config)` keeps the leaderboard honest by SPLIT discipline: stage 1 searches the
RAG/backend space on the disjoint `tuning` split, stage 2 scores ONLY the winning config on the
full `final` split, and only that stage-2 run is the leaderboard entry. The embedding is pinned
(never a search dimension). The search space is the M1 chunking machinery: strategy x
chunk_size x overlap-fraction (so overlap < size always holds) x top_k x retrieval_mode x
child_chunk_size. Over-context configs are PRUNED before they run -- `fits_context` estimates
the retrieved prompt tokens (`top_k x chunk_size / CHARS_PER_TOKEN` + headroom + completion) and
prunes when they exceed the model's effective window, so the prune depends on the RAG params,
not just the model. The study uses a persistent SQLite backend under `$DATA_DIR/optuna/` with
`load_if_exists`, so a killed search resumes. `optuna` is lazy-imported (the `[track]` extra);
the search-space + fit helpers are pure, and the per-trial evaluation + the stage-2 runner are
injectable (15 unit tests, no GPU). New CLI `tune`:

    llb tune --model llama3.2:3b --backend ollama --trials 30 --study uk1 \
        --goldset .data/llb/goldset/sample_rag_items.jsonl

Validated on this host (3 trials, Ollama): stage 1 picked markdown/size=960/top_k=6, then
stage 2 scored it on the final split as the leaderboard row. Residual: the search space does
not yet sample backend serving knobs (e.g. `max_model_len`); a pruning callback that also kills
mid-trial on `VramNotReclaimed` would tie M3.4 into the M3.3 sweep.

### Frontier prep utilities -- `llb.prep.frontier` (M3.5)
Two GPU-free, litellm-backed data-prep utilities that emit UNVERIFIED material for human review
(only `verified=True` items ever score a model):
- `prepare_goldset` drafts (question, reference_answer, exact source span) triples from real
  corpus docs. Every drafted span is RE-GROUNDED against the doc (`build_drafted_items` keeps
  only spans that are a verbatim substring, with exact offsets), so a label can never point at
  text that is not there; items are written `verified=false`, provenance `frontier-drafted`,
  with deterministic splits.
- `prepare_synthetic_corpus` generates synthetic docs with structured PLANTED labels and a hard
  guard that the planter model is NOT the eval judge (a model grading answers it authored is
  circular). It writes the docs, a `planted_labels.jsonl`, and a `provenance.json` recording
  planter vs judge.
`litellm` is lazy (the `[prep]` extra) and the completion call is injectable, so prompt
building, fenced/prose JSON parsing (`parse_json_block`), span grounding, and the planter!=judge
guard are pure and unit-tested without a key. New CLIs: `prepare-goldset`,
`prepare-synthetic-corpus`.

### Streamlit board -- `llb.board` (M3.7)
A thin leaderboard page over the canonical run bundles. The loading half (`board.data`) is pure
and unit-tested: `load_run_records` reads each `$DATA_DIR/run-eval/<ts>/manifest.json` (skipping
staging `.tmp` dirs) plus its per-case `scores` into `ModelResult`s (per-case objectives ->
the bootstrap CI), `best_per_model` keeps the highest-objective run per model, and
`config_summary` extracts the best config. `board.app` is a thin Streamlit view: the M3.6
`rank_board` (average-rank, Pareto `*`, CI-overlap `~`) plus best-config-per-model; deep
inspection stays in the MLflow UI. New CLI `board` (shells out to `streamlit run`; needs the
`[board]` extra). Verified on the real run bundles -- llama3.2:3b and gemma-4-E4B both land on
the Pareto front with bootstrap CIs. Residual: the page shows only objective quality (no judge
column until M3.8 close-out) and does not yet separate Tier-1 screen boards from Tier-2.

### Tier-1 public screen -- `llb.screen.public` (M3.1 + M3.9 Belebele wiring)
`run_screen(model, backend, base_url)` drives lm-eval-harness-uk through its `local-completions`
model against the already-launched OpenAI-compatible endpoint (no model loaded twice). It splits
into two TRACKS that are never cross-ranked: a **logprob** track (vLLM exposes token logprobs,
so MCQ tasks score by loglikelihood -- Belebele-uk + others) and a **generation** track
(Ollama / llama.cpp generate text only -- SQuAD-uk-style QA). `assert_single_track` refuses to
combine them (a loglikelihood accuracy is not comparable to a generation exact-match), mirroring
the Tier-1/Tier-2 guard in `aggregate`. COVERAGE is first-class: `parse_results` records which
requested tasks produced a result and marks the report `complete=False` when any are missing, so
a screen is never silently partial. lm-eval is heavy/external, so the run is injected (`runner=`)
and task selection, command building, parsing, and coverage are unit-tested without it. New CLI
`screen-public` (launches vLLM or uses the running Ollama / an explicit `--base-url`). The
default task lists wire Belebele-uk into the logprob track and SQuAD-uk into the generation
track (M3.9); task ids are overridable per harness build. Residual: the default UA task ids are
best-effort (the harness fork's exact names vary) and the live lm-eval path is unrun here.

### Ragas judge scorer -- `llb.scoring.judge` (M3.8, scorer half)
The trust GATE already existed (`run_judge` / `judge_is_trusted`: the judge only enters the
blend at calibration rho >= threshold, else it is demoted and objective correctness ranks
alone). M3.8 fills in the scorer it routes to: `ragas_scorer` computes Ragas **faithfulness**
(answer vs retrieved context) + **answer-relevancy** (answer vs question) with UA-localized
metric instructions. The pure halves -- `to_ragas_samples` (our records -> Ragas
`SingleTurnSample` fields), `extract_scores` (tolerating the 0.1 `answer_relevancy` vs 0.2
`response_relevancy` key), and the UA prompt text -- are unit-tested via an injected
`evaluate_fn`; the default `_default_ragas_evaluate` wires the real Ragas `evaluate` (lazy
`[rag]` extra, litellm judge). The calibration CLOSE-OUT stays blocked on choosing the judge
(OQ2) and producing human ratings, so live Ragas validation is still pending and the gate keeps
the judge demoted until then.

### Milestone 3 status

| Step | What | State |
|------|------|-------|
| M3.1 | `screen/` Tier-1 lm-eval-harness-uk adapter (logprob vs generation track, per-task coverage) | DONE |
| M3.2 | `backends/AvailabilityResolver` (discovery + vLLM>Ollama>llamacpp priority + offload-aware fit) | DONE |
| M3.3 | `executor/` hard isolation (process-per-cell, VRAM gate, thermal cooldown, resume, GPU telemetry) | DONE |
| M3.4 | `optimize/` two-stage Optuna (tuning-split search, over-context prune, persistent SQLite, stage-2 entry) | DONE |
| M3.5 | `prep/` frontier utils (`prepare-goldset` re-grounded drafts; `prepare-synthetic-corpus` planter!=judge) | DONE |
| M3.6 | `scoring/aggregate` N-model rigor (average-rank, Pareto, bootstrap CIs, no tier mixing) | DONE |
| M3.7 | `board/` thin Streamlit (rank + best-config-per-model + CIs over the run bundles) | DONE |
| M3.8 | `scoring/judge.ragas_scorer` (faithfulness + answer-relevancy, UA prompts); calibration close-out blocked on OQ2 + human ratings | CODE |
| M3.9 | Belebele-uk -> logprob screen, SQuAD-uk -> generation screen (M3.1); gold-set human verification still pending | CODE |
