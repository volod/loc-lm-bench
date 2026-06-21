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
- **Milestone 2 (real backend + telemetry) -- code complete:** a vLLM launcher (serves HF
  weights behind the same OpenAI-compatible interface), a per-backend telemetry hook
  (steady-state tokens/sec, peak VRAM, served vs requested context, load time, tokenizer
  efficiency), and a MAX_JOBS-capped vLLM build script. The from-source build + validating on
  a real model (M2.4) need a CUDA host (see the [vLLM guide](../guides/vllm-backend.md)); the
  code is unit-tested with fakes.

Two host-aware model utilities: `prep-models` prepares candidate models (pulls Ollama
tags, caches vLLM Hugging Face weights once), and `list-models` reports which candidates
can actually run here (GPU VRAM + system RAM, KV-cache-aware, with a GPU/CPU layer split).

164 tests passing; Ruff format/lint and mypy are clean. CI enforces formatting, linting,
static typing, and unit tests only (no GPU / network / heavy extras); every heavy dependency
is lazy-imported so the base install stays importable.

## Dev setup

Requires [`uv`](https://docs.astral.sh/uv/). `make venv` creates `.venv` (Python 3.11),
installs the `llb` package editable with ALL extras, and seeds `.env` from `.env.example`,
so a fresh checkout can run every command without a follow-up `uv pip install`.

    make            # list targets
    make venv       # .venv (py3.11) + package + all extras + .env (idempotent; RECREATE_VENV=1 to rebuild)
    make test       # pytest (164 tests)
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
      scoring/{correctness,judge,aggregate}.py  # objective + semantic + gated judge + ranking
      tracking/{manifest,mlflow,server}.py  # canonical artifacts + MLflow mirror/UI
      executor/{cases,reporting,runner,vram}.py  # per-case work + reporting + orchestration
    tests/                         # 164 tests across the above

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
`make run-eval MODEL=... LIMIT=...`.

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

## Milestone 2 -- real backend + telemetry (code complete)

A real vLLM backend behind the same interface, a steady-state telemetry hook, and the
MAX_JOBS-capped build entrypoint. The code is unit-tested with fakes; the from-source build +
serving a real model run on a CUDA host -- see the [vLLM guide](../guides/vllm-backend.md).

### vLLM launcher — `llb.backends.vllm` (M2.1)
`VllmLauncher` + `build_vllm_command` (pure). Documented under Backends above. The thin
`scripts/build_vllm.sh` entrypoint sources `scripts/shared/common.sh`, exports its canonical
`max_jobs()` result (`min(cores//2, RAM_GiB//14)`, AGENTS.md), and delegates to
`llb.build.vllm`. The default binary-only install and all ordinary dependencies use uv's
shared cache. Only a wheel built from `VLLM_SOURCE_DIR=<clean-git-checkout>` is exported
under `$DATA_DIR/wheels/vllm_<abi-key>_git<revision>/`. Weights are cached by `prep-models`.

    make build-vllm                                   # prebuilt wheel via uv shared cache
    VLLM_SOURCE_DIR=../vllm make build-vllm           # one ABI-keyed checkout wheel
    make run-eval BACKEND=vllm MODEL=google/gemma-4-12B-it-qat-w4a16-ct TELEMETRY=1

### Telemetry hook — `llb.backends.telemetry` (M2.2)
`measure_throughput` runs the steady-state protocol (fixed UA prompt set + fixed
max_new_tokens + N warmup iters) over `launcher.chat`, so tokens/sec is comparable across
models; cold-start `load_time_s` is recorded separately by launchers that own the backend
lifecycle, and remains null for an already-running external daemon such as Ollama.
`VramSampler` polls NVML (injected reader) for peak VRAM. `collect_telemetry` assembles the manifest record:
steady tokens/sec, tokenizer efficiency (tokens/UA-char), peak VRAM, requested-vs-served
context, load time, gpu-memory-utilization, and detected GPU. Wired into `run-eval`
behind `config.measure_telemetry` (`--telemetry`); recorded under `manifest.telemetry`.

### Milestone 2 status

| Step | What | State |
|------|------|-------|
| M2.1 | `VllmLauncher` + `build_vllm_command` + MAX_JOBS build helper / script | DONE (code) |
| M2.2 | telemetry hook (steady tokens/sec, peak VRAM, served ctx, load time, tok/char) | DONE (code) |
| M2.3 | candidate list seeded in `samples/models_uk.yaml` | PARTIAL (finalize + verify ids) |
| M2.4 | validate on one real vLLM-served model | TODO (needs a CUDA host) |

Remaining (need a CUDA host; scoped in [`plan.md`](plan.md)): the actual `build-vllm`,
finalizing + verifying the candidate HF repo ids (`make prep-models --backend vllm` 404s a
wrong id), and the M2.4 real-model run (then feed fit corrections back into `planner.py`).
