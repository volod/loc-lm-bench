# loc-lm-bench — Implemented (current state)

A snapshot of what exists and runs **today**. Forward work lives in
[`plan.md`](plan.md); the full spec is [`spec.md`](../design/spec.md).

**Status:**
- **Milestone 0 (data prep) complete:** schema, validator, disjoint splits, SQuAD
  ingestion, a real 250-item Ukrainian gold set (HPLT/ua-squad), judge-calibration
  stats, a chunking RAG-store builder.
- **Milestone 1 (CUDA-free eval skeleton) complete:** a canonical `RunConfig` + Typer
  CLI, a pinned-embedding FAISS RAG store + source-span retrieval metrics, a LangGraph
  retrieve -> generate flow over an OpenAI-compatible backend (Ollama), objective
  answer-correctness + a gated judge, a canonical manifest + scores record (MLflow
  mirror), and a minimal sequential runner with a VRAM gate. `llb run-eval` prints one
  ranked row, CUDA-free.

Two host-aware model utilities: `prep-models` prepares candidate models (pulls Ollama
tags, caches vLLM Hugging Face weights once), and `list-models` reports which candidates
can actually run here (GPU VRAM + system RAM, KV-cache-aware, with a GPU/CPU layer split).

114 tests passing, `ruff` clean. CI runs lint + unit tests only (no GPU / network / heavy
extras); every heavy dependency is lazy-imported so the base install stays importable.

## Dev setup

Requires [`uv`](https://docs.astral.sh/uv/). `make venv` creates `.venv` (Python 3.11),
installs the `llb` package editable with ALL extras, and seeds `.env` from `.env.example`,
so a fresh checkout can run every command without a follow-up `uv pip install`.

    make            # list targets
    make venv       # .venv (py3.11) + package + all extras + .env (one-time setup)
    make test       # pytest (114 tests)

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
      gen_rag_items.sh             # thin entrypoint -> llb.prep.gen_rag_items
    src/llb/
      config.py                    # RunConfig (Pydantic) -- the canonical run config
      main.py                      # Typer CLI: build-index, validate-retrieval, run-eval
      goldset/schema.py            # GoldItem + SourceSpan (Pydantic), load/dump
      goldset/splits.py            # deterministic disjoint split assignment
      goldset/validate.py          # corpus-grounded validator + CLI
      prep/gen_rag_items.py        # spec -> seed gold set
      prep/ingest_squad.py         # SQuAD-format (local or HF) -> canonical gold items
      judge/calibration.py         # Spearman rho + CI + trust decision + worksheet
      rag/chunking.py              # fixed/sentence/recursive/markdown/semantic chunking (offset-exact)
      rag/{embedding,index,store}.py  # pinned embedder + FAISS index + store (flat / parent-child)
      rag/retrieval.py             # recall@k / MRR by source-span overlap (pure)
      backends/{base,openai_client,ollama}.py  # launcher iface + chat call + Ollama
      backends/{hardware,prepare,planner}.py  # GPU/RAM detect + pull/cache + feasibility plan
      eval/graph.py                # LangGraph retrieve->generate flow + failure taxonomy
      scoring/{correctness,judge,aggregate}.py  # objective + semantic + gated judge + ranking
      tracking/manifest.py         # canonical manifest + scores (MLflow mirror)
      executor/{vram,runner}.py    # VRAM gate + minimal sequential run-eval
    tests/                         # 114 tests across the above

Runtime output (gitignored) under `$DATA_DIR/llb/` (default `.data/llb/`):
`corpus/`, `goldset/*.jsonl`, `rag/` (chunks + FAISS index), `runs/<run_name>/`
(`manifest.json` + `scores.{parquet,jsonl}`), `calibration_worksheet.csv`.

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
Reads `samples/rag_items_uk.json`, computes spans, writes + validates a seed gold set.

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

The CUDA-free walking skeleton: one model, fixed config, retrieve -> generate -> score ->
ranked row + manifest. Every heavy collaborator (FAISS, sentence-transformers, langgraph,
mlflow, pyarrow, pynvml) is lazy-imported, so the base install imports the whole package;
the real run needs a running Ollama (the `[rag,eval]` deps are installed by `make venv`).

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
`samples/run_config_uk.yaml`); CLI flags override individual fields.

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

### Backends — `llb.backends.{base,openai_client,ollama}`
`BackendLauncher` is the seam (Premise 1): all backends speak OpenAI-compatible HTTP, so
only the launcher + telemetry hook are backend-specific. `openai_client.chat_once` maps
transport failures to normalized tokens (`timeout` / `backend_error`). M1 ships the
`OllamaLauncher` (CUDA-free); vLLM / llama.cpp slot in behind the same interface in M2.

### Eval graph — `llb.eval.graph`
A LangGraph retrieve -> generate flow (the first of the ~3 DRY templates). The node
closures and `classify_response` are pure and unit-tested; only `build_rag_graph` imports
langgraph. Each case ends in exactly one typed status, recorded separately.

### Scoring — `llb.scoring.{correctness,judge,aggregate}`
`correctness` ranks models by reference answer-correctness (exact / token-F1 / contains,
Unicode-normalized for UA morphology); `score` is token-F1. An optional semantic-similarity
signal (cosine via the pinned embedder) is recorded when `--score-semantic` is set -- a
paraphrase signal token overlap misses, not yet blended into `score`. `judge` enforces the
gate (Premise 2): the Ragas judge only enters the blend at calibration rho >= threshold,
else it is demoted and the objective score ranks alone. `aggregate` produces the ranked row
(quality, then tok/s, then VRAM; infeasible models listed without a rank).

### Tracking — `llb.tracking.manifest`
The immutable `manifest.json` + per-case `scores.{parquet,jsonl}` are written FIRST; the
MLflow mirror runs after, best-effort, and a mirror failure never loses a completed run.
Parquet when `pyarrow` ([track]) is present, JSONL otherwise.

### Executor — `llb.executor.{vram,runner}`
`vram` is the basic NVML reclaim gate (injectable reader; raises `VramNotReclaimed` when
freed VRAM stays above tolerance). `runner.run_eval` orchestrates the single-model run;
every heavy collaborator is injectable, so the whole vertical runs end to end in a unit
test with fakes (`tests/test_runner.py`).

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
