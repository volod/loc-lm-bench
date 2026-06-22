# loc-lm-bench — Implemented (current state)

A snapshot of what exists and runs **today**. Forward work lives in
[`plan.md`](plan.md); the full spec is [`spec.md`](../design/spec.md).

**Status:**
- **Milestone 0 (data prep) complete:** schema, validator, disjoint splits, SQuAD
  ingestion, a committed 250-item post-edited Ukrainian development fixture, a manual
  gold-set skeleton, judge-calibration stats, and a chunking RAG-store builder.
- **Milestone 1 (eval skeleton) complete:** a canonical `RunConfig` + Typer
  CLI, a pinned-embedding FAISS RAG store + source-span retrieval metrics, a LangGraph
  retrieve -> generate flow over an OpenAI-compatible backend (Ollama), objective
  answer-correctness + judge gate/scorer seams, a canonical manifest + scores record (MLflow
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
- **Milestone 3 core delivered:** backend resolution, process-isolated resumable sweeps,
  two-stage RAG tuning, public-screen and frontier-prep adapters, N-model ranking, and a
  final-only Streamlit board. The audit fixes prevent tuning/calibration leakage into the
  board, make sweep markers interruption-safe, align drafted document ids with the RAG index,
  and harden external JSON/metric parsing. Full design acceptance gaps remain in `plan.md`.

Two host-aware model utilities: `prep-models` prepares candidate models (pulls Ollama
tags, caches vLLM Hugging Face weights once), and `list-models` reports which candidates
can actually run here (GPU VRAM + system RAM, KV-cache-aware, with a GPU/CPU layer split).

297 tests passing; Ruff format/lint and mypy are clean. CI enforces formatting, linting,
static typing, and unit tests only (no GPU / network / heavy extras); every heavy dependency
is lazy-imported so the base install stays importable.

## Dev setup

Requires [`uv`](https://docs.astral.sh/uv/). `make venv` creates `.venv` (Python 3.11),
installs the `llb` package editable with ALL extras, and seeds `.env` from `.env.example`,
so a fresh checkout can run every command without a follow-up `uv pip install`.

    make            # list targets
    make venv       # .venv (py3.11) + package + all extras + .env (idempotent; RECREATE_VENV=1 to rebuild)
    make test       # pytest (297 tests)
    make format     # apply canonical Ruff formatting to src/ and tests/
    make ci         # format check + lint + mypy + tests
    make demo-eval  # idempotent end-to-end: venv -> gold set -> index -> validate -> prep-models -> run-eval+telemetry
    make mlflow     # review all mirrored experiment runs at http://127.0.0.1:5000

`make demo-eval` runs the whole pipeline in order and is **idempotent** -- the venv is reused,
the committed gold set is validated, the index is rebuilt from its matching committed corpus,
and cached model downloads are skipped. It tees per-step output to `.data/llb/logs/pipeline-
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
      squad_uk_fixture.json        #   tiny SQuAD-format parser fixture
      goldsets/ua_squad_postedited_v1/
        goldset.jsonl              #   250 canonical verified public-development items
        corpus/                    #   250 exact source documents
        source.json                #   pinned revision, source digest, selection rule
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
      prep/ua_squad_source.py      # pinned reviewed-source identity + selection policy
      prep/goldset_skeleton.py     # timestamped from-scratch authoring template
      prep/published_goldset.py    # strict builder for the pinned committed fixture
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
    tests/                         # 297 tests across the above

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
models. Verification may be a local review or acceptance of a pinned upstream post-edited
fixture; `provenance` and fixture metadata preserve the distinction. `load_goldset` /
`dump_goldset` handle JSONL (UTF-8).

### Splits — `llb.goldset.splits`
`assign_splits(ids, ratios, seed)` -> deterministic, disjoint `calibration / tuning / final`.

### Validator (M0 acceptance) — `llb.goldset.validate`
Checks every span resolves to its labeled text on disk, ids unique, splits disjoint.

    make validate-goldset          # PASS on the committed public fixture

### Sample generator — `llb.prep.gen_rag_items`
Reads `samples/rag_items_uk.json`, computes spans, writes + validates a six-item synthetic
format fixture. It remains useful for parser and tiny smoke checks but is no longer the
default demo gold set.

    make gen-rag-items             # -> .data/llb/goldset/sample_rag_items.jsonl (6 items)

### SQuAD ingestion (M0.3) — `llb.prep.ingest_squad`
Maps SQuAD-format UA QA (flattened, nested, or HF rows where `answers` is a dict-string) ->
canonical items, with spans from the answer offset and a `find()` fallback. Drafts start with
`provenance: public-reused`, `verified: false`. The default ID-keyed verification ledger then
adopts matching canonical items from `ua_squad_postedited_v1`, including their reviewed corpus
files; unmatched drafts remain false. Local file or HF dataset (streams when `--max-items` set).
The HF loader accepts an explicit revision and normalizes both flattened rows and the pinned
source's nested SQuAD article rows.

    make ingest-uk-squad GOLDSET_MODE=development  # reproduce the pinned reviewed set
    make ingest-uk-squad GOLDSET_MODE=skeleton     # editable SQuAD template + instructions
    make ingest-uk-squad GOLDSET_MODE=draft        # reserved; reports planned M4.4
    make ingest-squad                          # the bundled fixture (4 items)
    make ingest-squad SQUAD_JSON=path.json     # a local SQuAD-uk export
    python -m llb.prep.ingest_squad --hf-dataset <id> --hf-split train   # needs HF_TOKEN (goldset extra via make venv)

The stable public development fixture is
`samples/goldsets/ua_squad_postedited_v1/goldset.jsonl`: 250 verified items and 250 distinct
documents, split cal=86/tun=82/final=82. It is a deterministic subset of the pinned
`FIdo-AI/ua-squad` validation export. The upstream card states that Ukrainian translations
were post-edited and answer spans aligned; `source.json` and the fixture README record the
revision, source SHA-256, selection rule, verification basis, attribution, and data license.
The pinned selection was reviewed by a human and all 250 items are `verified: true`.

`--verified-goldset <path>` replaces the default ledger and may be repeated to combine reviewed
sets. This is the review handoff for M3.5 `prepare-goldset` and planted-label outputs after a
human flips accepted entries to true; each ledger JSONL has a sibling `corpus/`. Canonical item
replacement, rather than a boolean-only flip, prevents a reused ID from certifying changed
content. `--no-verification-ledger` explicitly disables adoption. A zero-match import warns and
stays unverified.

The default development target uses one code-owned profile, `--pinned-development-source`, so
the Makefile cannot drift from the fixture metadata. It loads `FIdo-AI/ua-squad` revision
`943ef27daea65e400350ef1875d07c7e97288177`, split `validation`, then applies the exact fixture
selection: first grounded QA per distinct context, in source order. Live acceptance generated
250/250 verified items with 86/82/82 calibration/tuning/final splits; all canonical items and
all 250 corpus files exactly matched the committed fixture. This closes M3.9 and provides a
stable regenerated bundle for initial model tests. Normal initial tests should still use the
committed fixture through `make demo-eval`, which is offline and avoids unnecessary downloads.

`llb.prep.goldset_skeleton` writes an editable SQuAD example and instructions under
`$DATA_DIR/goldset-skeleton/<timestamp>/`. The complete manual is
[`docs/guides/goldset-from-scratch.md`](../guides/goldset-from-scratch.md).

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

### Judge calibration (M0.5 stats + M3.8 close-out scaffolding) — `llb.judge.calibration`
Spearman rho (no scipy), bootstrap CI, trust decision (`rho >= 0.6` else demote). Two
worksheet emitters: a blank one, and a pre-filled one driven from a run. The pre-filled
worksheet fills `model_answer` and (when a judge is configured) the `judge_rating` column,
running the judge **ungated** -- calibration measures whether the judge agrees with humans, so
the gate is irrelevant here; the human only adds `human_rating`. When the judge backend is
unavailable the column is left blank and the run logs a warning rather than failing.

The three Make targets drive the loop over the verified committed gold set
(`GOLDSET` defaults to `samples/goldsets/ua_squad_postedited_v1` -- all 86 calibration items
are `verified: true`, so M3.9 is already satisfied for it; no re-review needed):

    make calibration-worksheet                       # blank worksheet (86 calibration rows)
    make calibration-run JUDGE_MODEL=<served-model-id> \
        JUDGE_BASE_URL=http://127.0.0.1:8000/v1       # answers + judge_rating -> CAL_WS
    # human fills the human_rating column in CAL_WS, then:
    make calibration-score RATINGS=<filled.csv>      # rho + bootstrap CI + trust decision

Equivalent direct CLI:

    llb run-eval --split calibration --worksheet ws.csv --judge-model <id> \
        --judge-base-url http://127.0.0.1:8000/v1      # pre-fill answers + judge
    python -m llb.judge.calibration score --ratings ws.csv                  # rho + CI + decision

What still gates close-out is collecting the independent human `human_rating` column. The
maintained DeepEval metric engine, Ukrainian prompts, local endpoint adapter, targets, ungated
judge run, and calibration scoring are implemented and tested.

#### Judge model (OQ2 decided) + bias disclosure

The v1 judge is a **local Gemma-4 model**, chosen over a frontier API for **no corpus
data-egress and reproducibility**. The id is configured through `judge_model` /
`--judge-model` / `JUDGE_MODEL` and must match the id exposed by the local OpenAI-compatible
endpoint. `judge_base_url` / `--judge-base-url` / `JUDGE_BASE_URL` keeps that endpoint separate
from the candidate backend. Existing `hosted_vllm/` and `ollama_chat/` prefixes remain accepted
and are stripped before requests.

| GPU VRAM | Judge (served model id) | Notes |
|---|---|---|
| 12 GB | `ollama_chat/gemma-4-e4b-it` | smallest Gemma 4 via GGUF/CPU offload; the 12B will not fit |
| 16 GB (this box) | `hosted_vllm/google/gemma-4-12B-it-qat-w4a16-ct` | biggest Gemma 4 that fits; the configured default |
| 32 GB | `hosted_vllm/google/gemma-4-12B-it` | bf16 12B (higher fidelity) + headroom to co-host judge + a candidate |

On 16 GB a 12B judge normally cannot co-reside with a vLLM candidate. Use Ollama GGUF/CPU
offload, a smaller test judge, or another local host while generating the calibration worksheet.

**Bias (disclosed, not eliminated).** This judge is **not independent of the candidate pool**:
Gemma-4 (E4B/12B) are candidates, and MamayLM v2 + Lapa are Gemma-3 fine-tunes -- so the judge
shares architecture, tokenizer, and pretraining lineage with most of the pool and may
**self-prefer Gemma-family answers** over the non-Gemma ones (Qwen3.6, Llama 3.2). The bias can
move the *ranking*, not just absolute scores. It is accepted because: (1) the judge is **gated**
(Premise 2) -- it enters ranking only when calibration rho >= 0.6 against the human-verified set,
else it is demoted to a diagnostic and objective correctness ranks alone; (2) the headline blend
keeps objective reference-correctness weighted; (3) the disclosure (`JUDGE_BIAS_NOTE` in
`scoring/judge.py`) travels with the run; and (4) a **non-Gemma cross-check judge** (e.g.
Qwen3.6 or a frontier model) can re-score the same calibration split to quantify the family
delta, with the board's judge-cohort guard preventing mixed cohorts in one board. The spec also
cautions a small local judge may not clear the gate for Ukrainian -- a 12B is borderline; if rho
< 0.6 the judge stays demoted, which is the gate working as designed.

## Milestone 0 status

| Step | What | State |
|------|------|-------|
| M0.1 schema | Pydantic `GoldItem` / `SourceSpan` | DONE |
| M0.2 sample generator | `gen_rag_items` + sample spec | DONE |
| M0.3 stable public gold set | pinned post-edited UA-SQuAD fixture (250 items/docs) | DONE |
| M0.4 splits | deterministic disjoint partition | DONE |
| M0.5 calibration stats | rho + CI + blank/pre-filled worksheet | DONE (code) |
| chunking | fixed/sentence/recursive RAG-store builder | DONE |
| acceptance | validator PASS (sample + fixture + 250-item set), suite green | DONE |

Remaining (blocked on a judge choice or human input; scoped forward in [`plan.md`](plan.md)):
- **Judge-calibration close-out (plan M3.8):** the stats, the gate, the executor judge
  wiring, the chosen judge (OQ2 -- a local Gemma-4 model, bias disclosed above), and the full
  pre-filled-worksheet scaffolding (model answers + ungated `judge_rating` via
  `make calibration-run` / `run-eval --worksheet --judge-model`, scored by
  `make calibration-score`) all exist and are unit-tested. The only required residual is external:
  collecting the human `human_rating` column over the verified calibration split.

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
                  score: reference answer-correctness (objective)
                         [judge scorer/gate exists but is not wired here yet]
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
    llb run-eval --config samples/run_config_uk.yaml --judge-rho 0.7  # records gate status
    llb run-eval --split calibration --worksheet ws.csv   # pre-fill a calibration worksheet
    llb run-eval --score-semantic                         # also record semantic correctness

Or via make: `make prep-models`, `make build-index`, `make validate-retrieval`,
`make run-eval MODEL=... LIMIT=...`. The make targets default `GOLDSET` and `CORPUS` to the
committed post-edited public fixture, so they run without regeneration or network access;
override both for another set. A missing gold set or a set with no `verified: true` items in
the split fails with an actionable message rather than a raw traceback.

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
(embedding-aware -- see M4.1 below), the KV cache per token (`2 x n_layers x kv_dim x 2B`, batch=1,
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
weights require calibration. `judge` enforces the gate (Premise 2): the DeepEval G-Eval judge
only may enter aggregate ranking at calibration rho >= threshold; below it, objective score
ranks alone. `run-eval` invokes it when configured and trusted, records per-case scores, and
keeps the row objective-only otherwise. `aggregate` produces the ranked row
(quality, then tok/s, then VRAM; infeasible models listed without a rank).

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
| M1.5 | objective answer-correctness (+ semantic) + judge gate seam | CORE (executor judge wiring -> M3.8) |
| M1.6 | canonical manifest + scores, MLflow mirror | DONE |
| M1.7 | minimal sequential runner + NVML VRAM gate | DONE |
| M1.8 | `run-eval` prints one ranked row (SQuAD-uk seed) | DONE |

Residual M1 work is scoped forward in [`plan.md`](plan.md): human judge calibration (M3.8),
plus map-reduce / multi-hop eval templates (deferred
until the text-analysis benchmark needs them). The optional semantic-similarity correctness
signal is built (`--score-semantic`).

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

Planner-vs-measured fit: the model's **weights load 9.8 GiB**, ~2.3x the old flat ~4.2 GiB
estimate (`params_b x bpw`). w4a16 quantizes only the linear layers while Gemma's 256k-token
embedding stays high-precision, so the flat product under-estimated w4a16 weights. The
embedding-aware estimator that fixes this is now delivered (M4.1 below); the measured floor is the
regression anchor in `samples/models_uk.yaml`.

### Milestone 2 status

| Step | What | State |
|------|------|-------|
| M2.1 | `VllmLauncher` + `build_vllm_command` + MAX_JOBS build helper / script | DONE |
| M2.2 | telemetry hook (steady tokens/sec, peak VRAM, served ctx, load time, tok/char) | DONE |
| M2.3 | candidate list in `samples/models_uk.yaml`; vLLM repo ids verified via `prep-models` | DONE |
| M2.4 | validated on a real vLLM-served model (gemma-4-E4B-it-w4a16) w/ real telemetry | DONE |

Residual (non-blocking, forward in [`plan.md`](plan.md) Milestone 4): the embedding-aware VRAM
estimate is now DONE (M4.1 below); still open are a pre-launch VRAM-contention guard (M4.2) and
surfacing the vLLM serving knobs as `run-eval` CLI flags (M4.3).

## Milestone 3 -- two-tier + scale + rigor (core + depth hardening delivered)

The M3 core components are built and unit-tested. The CLI grew `resolve-models`, `sweep`,
`tune`, `prepare-goldset`, `prepare-synthetic-corpus`, `screen-public`, and `board`. A
post-implementation audit confirmed the component boundaries and found that the full design
acceptance is not yet closed: screen-to-finalist orchestration, process-isolated Optuna trials
with backend-parameter search, judge integration/calibration, and human gold-set verification
remain forward work. The delivered behavior and those boundaries are stated below; residual
work stays in [`plan.md`](plan.md).

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
`rank_board` rejects duplicate model configs so callers must select exactly one config per
model before rank calculation; this avoids silently overwriting average ranks by model key.

### Hard-isolation sweep -- `llb.executor.isolation` (M3.3)
`run_sweep(configs)` runs one (model, config) cell per PROCESS so a leak or crash in one cell
cannot bias the next: the default `CellRunner` shells out to `python -m llb.main run-eval
--config <cell> --split <s>`, so the vLLM server AND the whole CUDA context die with the cell.
The per-cell isolation contract is ONE reusable primitive, `isolate_cell(work, backend=...)`,
shared by the sweep, the public screen (`screen.run_screen_isolated`), and every Optuna trial
(`optimize.tuner.with_isolation`) -- so "process per cell + gate + cooldown" is defined once.
Between cells it gates two things and records a third:
- **PID-attributed VRAM reclaim gate** (M3.3): snapshot the VRAM baseline + the set of PIDs
  already holding VRAM, run the cell, then `wait_for_reclaim`. If VRAM does not return to
  baseline, `classify_residual` ATTRIBUTES the residual: a PID that APPEARED during the cell and
  still holds VRAM is a `leaked` cell -> raise `VramNotReclaimed` and abort the whole sweep; a
  pre-existing process that merely grew is a `baseline_shift` -> tolerated (logged), so an
  unrelated desktop process can no longer falsely abort the sweep. The gate runs only for
  `GATE_BACKENDS` (vLLM / llama.cpp) that own their VRAM; Ollama keeps weights warm by design.
  Without a `pid_usage_reader` it stays conservative (any over-tolerance residual aborts).
- **Thermal cooldown** (`cool_down`): wait until the hottest GPU is <= a threshold, capped at a
  max wait so a warm room cannot stall the sweep; throughput is only comparable at like clocks.
- **GPU telemetry** (`sample_gpu` via nvidia-smi): temp / power / SM+mem clocks per cell.
The sweep is RESUMABLE: each cell has a stable `cell_key` (a hash of its reproducibility-
relevant config, ignoring `run_name`) and atomically publishes a marker under
`$DATA_DIR/sweep/<id>/cells/`, so a re-run skips finished cells. A truncated/invalid marker is
treated as unfinished and rerun instead of crashing or falsely skipping the cell. Every side
effect (subprocess, NVML reader, GPU sampler, sleep) is injectable. New CLI `sweep` resolves each
manifest model to a backend (M3.2) and runs the isolated cells:

    llb sweep --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl \
        --sweep-id run1                                                        # run
    llb sweep --sweep-id run1                                                       # resume (skips done)

Validated on this host: an Ollama cell ran as a subprocess + resumed on re-run; and a real
vLLM cell (gemma-4-E4B) ran through the live PID-attributed gate (`nvml_reader` +
`nvml_process_reader`), reclaiming to baseline (residual 2 MB, no leaked PID) -- the marker +
bundle recorded it. The CLIs (`sweep`, `screen-public --isolated`, `tune --isolate`) wire
best-effort NVML readers. Residual: the sweep generates one cell per model at the default RAG
config; the RAG-parameter search space is driven by Optuna (M3.4).

### Two-stage Optuna RAG tuning -- `llb.optimize.tuner` (M3.4)
`two_stage(base_config)` keeps the leaderboard honest by SPLIT discipline: stage 1 searches the
RAG space for one fixed model/backend on the disjoint `tuning` split, stage 2 scores ONLY the
winning config on the full `final` split, and only that stage-2 run is the leaderboard entry.
The embedding is pinned (never a search dimension). The search space is the M1 chunking
machinery: strategy x
chunk_size x overlap-fraction (so overlap < size always holds) x top_k x retrieval_mode x
child_chunk_size. Over-context configs are PRUNED before they run -- `fits_context` estimates
the retrieved prompt tokens (`top_k x chunk_size / CHARS_PER_TOKEN` + headroom + completion) and
prunes when they exceed the model's effective window, so the prune depends on the RAG params,
not just the model. The study uses a persistent SQLite backend under `$DATA_DIR/optuna/` with
`load_if_exists`, so a killed search resumes. `optuna` is lazy-imported (the `[track]` extra);
the search-space + fit helpers are pure, and the per-trial evaluation + the stage-2 runner are
injectable and tested without a GPU. New CLI `tune`:

    llb tune --model llama3.2:3b --backend ollama --trials 30 --study uk1 \
        --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl

Validated on this host (3 trials, Ollama): stage 1 picked markdown/size=960/top_k=6, then
stage 2 scored it on the final split as the leaderboard row. The backend is fixed for a study;
backend serving knobs are not sampled, and trials currently execute in-process rather than
through M3.3 isolation. These are spec-depth gaps, not delivered behavior.

### Frontier prep utilities -- `llb.prep.frontier` (M3.5)
Two GPU-free, litellm-backed data-prep utilities that emit UNVERIFIED material for human review
(only `verified=True` items ever score a model):
- `prepare_goldset` drafts (question, reference_answer, exact source span) triples from real
  corpus docs. Every drafted span is RE-GROUNDED against the doc (`build_drafted_items` keeps
  only spans that are a verbatim substring, with exact offsets), so a label can never point at
  text that is not there; items are written `verified=false`, provenance `frontier-drafted`,
  with deterministic splits. Document ids use corpus-relative paths, matching the RAG index
  and avoiding collisions when nested directories contain the same filename.
- `prepare_synthetic_corpus` generates synthetic docs with structured PLANTED labels and a hard
  guard that the planter model is NOT the eval judge (a model grading answers it authored is
  circular). It writes the docs, a `planted_labels.jsonl`, and a `provenance.json` recording
  planter vs judge.
`litellm` is lazy (the `[prep]` extra) and the completion call is injectable, so prompt
building, fenced/prose JSON parsing (`parse_json_block`), span grounding, and the planter!=judge
guard are pure and unit-tested without a key. Malformed top-level JSON shapes and non-object
entries are skipped with a warning instead of crashing a long prep run. New CLIs: `prepare-goldset`,
`prepare-synthetic-corpus`. Accepted outputs can become custom verification ledgers by retaining
their stable IDs, flipping only human-approved entries to `verified=true`, and passing the JSONL
to the ingester with `--verified-goldset`.

### Streamlit board -- `llb.board` (M3.7)
A thin leaderboard page over the canonical run bundles. The loading half (`board.data`) is pure
and unit-tested: `load_run_records` reads each `$DATA_DIR/run-eval/<ts>/manifest.json` (skipping
staging `.tmp` dirs) plus its per-case `scores` into `ModelResult`s (per-case objectives ->
the bootstrap CI). Run manifests now record the evaluated split, and the board accepts only
`final` runs; for legacy manifests it infers the split from case rows. This prevents tuning or
calibration scores from leaking onto the leaderboard. `best_per_model` keeps the
highest-objective final run per model, and
`config_summary` extracts the best config. `board.app` is a thin Streamlit view: the M3.6
`rank_board` (average-rank, Pareto `*`, CI-overlap `~`) plus best-config-per-model; deep
inspection stays in the MLflow UI. New CLI `board` (shells out to `streamlit run`; needs the
`[board]` extra). Verified on the real run bundles -- llama3.2:3b and gemma-4-E4B both land on
the Pareto front with bootstrap CIs. Residual: the page shows only objective quality (no judge
column until M3.8 close-out) and does not yet separate Tier-1 screen boards from Tier-2.

### Tier-1 public screen -- `llb.screen.public` (M3.1 + M3.9 dataset wiring)
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
track (M3.9); task ids are overridable per harness build. Task selection de-duplicates
user-supplied/default ids, stderr fields can never be selected as headline metrics, and model
names are sanitized before they become output filenames.

`run_screen_isolated` (M3.1) runs a screen under the SAME isolation contract as a Tier-2 sweep
cell by REUSING the executor primitives: it snapshots VRAM, runs the screen (whose backend lives
in its own process), then -- for VRAM-owning backends (vLLM) -- asserts the freed VRAM returns
to baseline (`VramNotReclaimed` aborts) and applies the capped thermal cooldown; Ollama is never
gated (it keeps weights warm). `screen-public --isolated` (with `--max-model-len` to cap the
vLLM KV cache) wires it and writes a `<model>.isolation.json` (VRAM residual + cooldown) beside
the report. The `local-completions` command is TRACK-aware: the logprob track points lm-eval at
the model's HF tokenizer (loglikelihood needs it), the generation track sets
`tokenizer_backend=None` (an Ollama tag is not a HF repo); the runner reads lm-eval's
`<out>/<model>/results_*.json`.

Validated LIVE against lm-eval 0.4.12 on this host: the generation track on Ollama
(`llama3.2:3b`, `global_piqa_prompted_ukr_cyrl`, coverage 1/1) and the logprob track on vLLM
(`gemma-4-E4B`, `belebele_ukr_Cyrl` + `arc_uk` + `hellaswag_uk` + `m_mmlu_uk` + global_piqa,
coverage 5/5) -- the latter exercising the VRAM-reclaim gate (residual reclaimed to baseline
after the cell). The default task ids are confirmed UA tasks; `squad_uk` (which does not exist
upstream) was replaced by `global_piqa_prompted_ukr_cyrl`.

### DeepEval Ukrainian judge -- `llb.scoring.judge` (M3.8)
The trust GATE already existed (`run_judge` / `judge_is_trusted`: the judge only enters the
blend at calibration rho >= threshold, else it is demoted and objective correctness ranks
alone). `deepeval_scorer` uses maintained DeepEval 4 G-Eval metrics for **faithfulness**
(answer vs retrieved context) and **answer relevancy** (answer vs question), with fixed Ukrainian
evaluation steps and a Ukrainian JSON result template. `LocalModel` connects to any local
OpenAI-compatible endpoint; no cloud provider or embedding call is required. The dependency is
lazy under `[rag]`, while the endpoint and model are recorded in each manifest without secrets.

Ragas 0.4.3 was evaluated first but failed to import against the project's current LangChain
stack because it imports modules removed by current LangChain. The project does not pin old
LangChain, install shims, or retain Ragas in the lock graph. DeepEval 4.0.6 imports in the current
environment, and the test suite executes its real G-Eval engine with the local model transport
replaced by an in-process OpenAI-compatible fake. `llb judge-experiment` / `make
judge-experiment` adds endpoint-level smoke validation through three fixed Ukrainian cases and
writes the served-model metadata, exact prompts, cases, and scores under
`$DATA_DIR/judge-experiment/<timestamp>/result.json`. No judge server was running on this
development host, so no live model scores are claimed. See the
[local judge guide](../guides/judge-experiments.md).

The scorer is called by `executor.run_eval` in both the gated ranking path and ungated
calibration path, and the board loads judge metrics (M3.7). The required calibration close-out
residual is only collecting human ratings and passing rho/CI.

### Milestone 3 depth/acceptance hardening
On top of the core modules, the spec-depth requirements landed:
- **Per-source model metadata (M3.2).** `sources:` accepts per-backend records (`source` +
  its own `quant`/arch/`min_vram_gb`); the resolver prices each artifact independently, so the
  bf16 UA models (MamayLM/Lapa) now resolve to their q4 GGUF on Ollama. `BackendCandidate`
  carries the priced `quant`.
- **Backend-aware Optuna (M3.4).** `suggest_overrides` samples `gpu_memory_utilization` /
  `max_model_len` ONLY for vLLM; a MEASURED OOM during a trial prunes it (vs the pre-run
  estimate); equal-quality trials tie-break by higher throughput; an `on_trial` hook mirrors
  each trial as a nested MLflow child run.
- **Prep provenance + grounding (M3.5).** `ProvenanceLog` records per-call model/tokens/cost
  into a `*.provenance.json`; `ground_span` adds a casefold/whitespace-normalized fallback that
  still maps to EXACT offsets; synthetic corpora are written under `out/corpus/` (ready for
  `build-index`) with an explicit `synthetic: true` tag.
- **Statistical completeness (M3.6).** Per-case objective/semantic/judge bootstrap CIs; the
  rank-uncertainty `unresolved` flag is computed on the per-case HEADLINE blend (`per_case_quality`),
  not objective alone; `ranking_policy_note` prints the signals + judge weight so the blend is
  never silently applied.
- **Board completion (M3.7).** Loads per-case judge/semantic series; renders Tier-1 screens
  SEPARATELY from the Tier-2 board (`load_screen_reports`); picks each model's best config by the
  ranking policy (`best_per_model(judge_trusted=...)`); `rank_board` rejects an incompatible
  judge cohort.
- **Judge integration + calibration scaffolding (M3.8).** `run_judge` is wired into
  `executor.run_eval`: it builds per-case (question, answer, retrieved-contexts) records, scores
  with the GATED judge, persists per-case `judge_score` + an aggregate in the manifest, and
  enters the blend ONLY when trusted. The calibration close-out adds an **ungated** path
  (`_judge_ratings`): `run-eval --split calibration --worksheet --judge-model` (and
  `make calibration-run`) pre-fills the worksheet's `model_answer` and `judge_rating` columns by
  running the judge regardless of trust, so the human only adds `human_rating`; the judge
  backend being unavailable degrades to a blank column + warning rather than a hard failure.
  `make calibration-score` then computes rho/CI/decision. The loop runs over the verified
  committed gold set (86/86 calibration items `verified:true`), so it needs no re-review (M3.9).
- **Isolation contract (M3.3).** One shared `isolate_cell` primitive (sweep + screen + Optuna
  trial) runs the LIVE PID-attributed reclaim gate -- `classify_residual` over a `nvml_process_reader`
  PID-set diff distinguishes a `leaked` cell (a PID that appeared during the cell still holds VRAM)
  from a tolerated `baseline_shift` -- plus the capped cooldown; the sweep also writes a
  `thermal.json` into the run BUNDLE. Live-validated on a real vLLM sweep.
- **Tier handoff (M3.1).** `select_finalists` is a deterministic per-track top-N policy (tracks
  never cross-ranked); the new `pipeline` command chains finalists -> two-stage tune -> final board.

### Milestone 3 status

| Step | What | State |
|------|------|-------|
| M3.1 | Tier-1 adapter + finalist policy + `pipeline` + `run_screen_isolated`; live-validated on Ollama (generation) and vLLM (logprob, VRAM gate exercised); UA task ids confirmed against lm-eval 0.4.12 | DONE |
| M3.2 | AvailabilityResolver + per-source artifact metadata (own quant/arch priced) | DONE |
| M3.3 | `isolate_cell` shared by sweep + screen + Optuna; live PID-attributed reclaim gate (leak vs baseline shift); thermal flag in run bundle | DONE (live-validated: real vLLM sweep, residual 2 MB reclaimed) |
| M3.4 | two-stage RAG tuning + backend-aware serving params, measured-OOM prune, throughput tie-break, nested-MLflow hook | DONE |
| M3.5 | frontier drafts + planter guard + per-call cost provenance + fuzzy-but-exact grounding + synthetic build-index bundle | DONE |
| M3.6 | average-rank, Pareto, per-case objective/semantic/judge CIs, headline-CI rank-uncertainty, policy-visible blend | DONE |
| M3.7 | final-only board + judge/semantic load, Tier-1/Tier-2 separation, best-by-policy, judge-cohort guard | DONE |
| M3.8 | maintained DeepEval G-Eval scorer + Ukrainian prompts + local endpoint smoke artifact; gate + `run_judge` wired into `run_eval`; local Gemma-4 judge choice and bias disclosure; pre-filled calibration worksheet + rho/CI commands | DONE (implementation); close-out gated only on human `human_rating` collection |
| M3.9 | committed human-reviewed fixture + pinned reproducible development importer + ID-keyed canonical adoption/custom ledgers + public task defaults | DONE (live importer acceptance: 250/250 verified, exact item/corpus match) |

## Milestone 4 -- robustness + ontology data prep + third backend (in progress)

### Embedding-aware VRAM estimate -- `llb.backends.planner` (M4.1)
The weights estimate is no longer a flat `params_b x bpw`. Partial quants (w4a16 / int4 / fp8)
quantize only the linear layers while the token embedding + norms stay high-precision; with a
256k-token vocab that premium is large. `weights_mib_detailed(params_b, quant_bpw, hi_params,
embed_bpw)` prices the high-precision mass at `embed_bpw` (default 16) and only the remainder at
the quant bpw. The high-precision mass is `hi_precision_params(spec)`: an explicit
`hi_precision_params_b` wins (for quirks the vocab formula misses, e.g. Gemma 3n Per-Layer
Embeddings), else -- ONLY for a partial quant (`PARTIAL_QUANT_FORMATS`) -- the token embedding
(`vocab_size x hidden_size`, plus the untied head) derived from the spec. GGUF k-quants and
bf16/fp32 get no premium (they quantize uniformly). The detailed estimate flows through
`plan_model`, so the `AvailabilityResolver` fit (M3.2) and Optuna's over-VRAM prune (M3.4)
inherit it for free.

Arch fields come from the spec or, when omitted, a cached `config.json`: `enrich_arch(spec)`
reads `vocab_size` / `hidden_size` / `num_hidden_layers` / `tie_word_embeddings` via
`arch_from_config` (handles Gemma's nested `text_config`) using `cached_config_path` (HF cache,
never downloads). It fills only missing fields (curated YAML wins) and skips non-HF sources
(Ollama tags, `hf.co/...:Q4_K_M`). `list-models` / `resolve-models` enrich specs before planning.

Validated against the M2.4 measurement: gemma-4-E4B-it-w4a16 now estimates **9.81 GiB** (the
measured floor is 9.8 GiB) vs the old flat ~4.2 GiB; the Gemma 12B w4a16 gains a ~1.3 GiB
embedding premium (6.3 -> 7.6 GiB) and the 27B fp8 ~1.3 GiB. The E4B floor + the new arch fields
are recorded in `samples/models_uk.yaml` as the regression anchor.

Possible further improvements: the E4B high-precision mass is measurement-anchored
(`hi_precision_params_b`) rather than derived from the Gemma 3n PLE shapes in `config.json`;
sliding-window KV (Gemma 3/4) is still estimated as full attention (conservative at long ctx);
and `enrich_arch` fills gaps rather than letting `config.json` override curated values.

### Pre-launch VRAM-contention guard -- `llb.executor.contention` (M4.2)
Before a VRAM-owning backend (vLLM) starts, `run-eval` runs a guard so a resident process can no
longer trip vLLM's startup free-memory check (the original M2.4 failure: Ollama held ~2.8 GB, so
`gpu-memory-utilization x total` exceeded free VRAM). `plan_guard(total, free, requested_util,
weight_floor)` (pure) caps `gpu-memory-utilization` at `(free - margin) / total` (rounded down,
only ever lowered) -- the non-destructive default AUTO-DERATE -- and returns a `ContentionReport`
{total, free, safe_util, target, residents, derated, fits, action, note}. It ABORTS with an
actionable message when even the derated target cannot hold the M4.1 weight floor + vLLM's ~2 GB
non-weight serving overhead (`DEFAULT_VLLM_OVERHEAD_MB`: CUDA context, peak activations, CUDA-graph
capture) + a minimal KV working set; without that overhead term the guard would derate into a
doomed launch (the live finding: a budget that left 0 for KV blocks tripped vLLM's "No available
memory for the cache blocks"). Free VRAM comes from nvidia-smi (so the derate works without
`[telemetry]`); resident PIDs come from NVML when present (best-effort attribution in the note).

`apply_contention_guard` adds the opt-in escalations: `--evict` unloads Ollama's resident models
(`/api/ps` -> `keep_alive: 0` per model; never kills a process) then re-reads; `--wait` polls free
VRAM until the requested target fits or a timeout. The runner (`_guard_vllm_contention`) calls it
only for vLLM and only on the real launch path (injected launchers in tests skip it), lowers the
launcher's `gpu_memory_utilization` on a derate, and records the `ContentionReport` in the manifest
(`RunManifest.contention`). Readers, the evict, and sleep are injectable; the math + escalations
are unit-tested without a GPU.

Possible further improvements: live validation on the CUDA host (a real contended vLLM launch);
the guard reads GPU 0 only (single-GPU assumption); the abort's KV headroom is a fixed floor
rather than the arch-derived KV for the served context.

### llama.cpp launcher -- `llb.backends.llamacpp` (M4.5)
The third backend the M3.2 resolver routes to: a model too big for vLLM's no-offload VRAM
resolves to its GGUF, which `llama-server` runs by splitting layers GPU<->CPU. `LlamaCppLauncher`
sits behind the same `BackendLauncher` + OpenAI-compatible `chat_once` seam as Ollama/vLLM, so the
eval/RAG/judge code is unchanged. `build_llamacpp_command` assembles the `llama-server` argv:
`llamacpp_source_args` maps a source to `-m <path.gguf>` (local) or `-hf <repo>[:quant]` (an HF
GGUF repo, incl. the Ollama-style `hf.co/<repo>:<quant>` the resolver's sources carry -- one
string serves on both GGUF backends); `-ngl` is the GPU/CPU offload split and `-c` the served
context. `start()` polls `/health` until 200 (preserving the startup log on failure, mirroring
vLLM), then reads the served `n_ctx` from `/props` (falling back to the requested `ctx_size`).

Telemetry reuses the backend-agnostic `collect_telemetry` (steady tokens/sec + peak VRAM); the
launcher records `n_gpu_layers` + `ctx_size` in its meta, and `TelemetryReport` now carries
`n_gpu_layers` so the served-vs-requested context (`requested_context`/`served_context`) and the
offload split land in the manifest. `llamacpp` is in `GATE_BACKENDS`, so the M3.3 reclaim gate
applies (it owns its VRAM). The runner's `_make_launcher` builds it from `RunConfig.llamacpp_host`
(env `LLAMACPP_HOST`, port parsed from the URL) + `n_gpu_layers`, with the context from
`max_model_len`. The process factory, HTTP probe, and sleep are injectable, so command building,
readiness, chat, telemetry, resolver routing, and the reclaim gate are all unit-tested without
llama.cpp/CUDA.

Possible further improvements: live validation on a CUDA host serving a real GGUF; auto-derive
`n_gpu_layers` from the planner's `gpu_layers` split for an offload model (today it is config-set,
defaulting to -1 = all on GPU); the `/props` served-context parse depends on the llama.cpp build's
response shape (both known shapes are handled, with a fallback).

| Step | What | State |
|------|------|-------|
| M4.1 | embedding-aware weights (`weights_mib_detailed` + `hi_precision_params`, partial-quant-gated) + `config.json` enrichment (`enrich_arch`/`arch_from_config`); fed through `plan_model` to resolver + Optuna; YAML arch fields + measured anchor | DONE (E4B estimate 9.81 vs 9.8 GiB measured) |
| M4.2 | pre-launch VRAM-contention guard (`plan_guard` derate + abort, `--evict`/`--wait`), wired into `run-eval` for vLLM, recorded in the manifest | DONE (unit-tested; live contended-launch validation pending a CUDA host) |
| M4.5 | llama.cpp launcher (`LlamaCppLauncher` `llama-server` subprocess: `-hf`/`-m` source, `-ngl` offload split, `/health`+`/props`), telemetry (`n_gpu_layers` + served ctx), reclaim gate, `_make_launcher` wiring | DONE (unit-tested; live GGUF serve pending a CUDA host) |
