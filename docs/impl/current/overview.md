# loc-lm-bench -- Implemented (current state)

A snapshot of what exists and runs **today**. Forward work lives in
[`plan.md`](../plan.md); the full spec is [`spec.md`](../../design/spec.md).

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
  ~112 s, served ctx 8192). See the [vLLM guide](../../guides/vllm-backend.md).
- **Milestone 3 core delivered:** backend resolution, process-isolated resumable sweeps,
  two-stage RAG tuning, public-screen and frontier-prep adapters, N-model ranking, and a
  final-only Streamlit board. The audit fixes prevent tuning/calibration leakage into the
  board, make sweep markers interruption-safe, align drafted document ids with the RAG index,
  and harden external JSON/metric parsing. Full design acceptance gaps remain in `plan.md`.
- **Milestone 4 (robustness + ontology data prep + third backend) complete:** an
  embedding-aware VRAM estimate (prices the high-precision embedding mass; E4B 9.81 vs 9.8 GiB
  measured), a pre-launch VRAM-contention guard (auto-derate + `--evict`/`--wait`), vLLM serving
  knobs as `run-eval` flags + a flashinfer sampler preflight, the ontology-assisted gold-set
  draft pipeline (`prepare-goldset-draft`: 7 grained stages, local/frontier endpoint adapter,
  exact-grounded `verified=false` bundles), and the llama.cpp launcher (the third backend behind
  the same OpenAI-compatible seam). All unit-tested without a GPU; the on-hardware confirmations
  are carried forward in `plan.md` (M5.6). The only M3 residual is human-gated (judge
  calibration ratings).

Two host-aware model utilities: `prep-models` prepares candidate models (pulls Ollama
tags, caches vLLM Hugging Face weights once), and `list-models` reports which candidates
can actually run here (GPU VRAM + system RAM, KV-cache-aware, with a GPU/CPU layer split).

The full pytest suite passes; Ruff format/lint and mypy are clean. CI enforces formatting, linting,
static typing, and unit tests only (no GPU / network / heavy extras); every heavy dependency
is lazy-imported so the base install stays importable.

The suite is split into two groups via the `slow` pytest marker (registered in `pyproject.toml`):
the FULL suite (`make test`, run locally) includes everything, while the LIGHTWEIGHT suite
(`make ci` / `make test-fast`, selected with `-m "not slow"`) drops the intrinsically expensive
tests so GitHub CI stays fast (~5s vs ~19s locally). Ten tests carry `@pytest.mark.slow`: the six
real-Optuna `tune`/`two_stage` sweeps (`tests/test_tuner.py`), the embedder-loading prefilled
worksheet (`tests/test_runner.py`), the two `bash`-subprocess vLLM build helpers
(`tests/test_build_helper.py`), and the real-DeepEval metric path (`tests/test_judge.py`). Mark a
new test `slow` only when its cost is intrinsic (Optuna sweep, model/embedder load, deepeval,
subprocess), not a one-off first-import artifact.

## Dev setup

Requires [`uv`](https://docs.astral.sh/uv/). `make venv` creates `.venv` (Python 3.11),
installs the `llb` package editable with ALL extras, and seeds `.env` from `.env.example`,
so a fresh checkout can run every command without a follow-up `uv pip install`.

    make            # list targets
    make venv # .venv (py3.11) + package + all extras + .env (idempotent;
    RECREATE_VENV=1 to
    rebuild)
    make test       # pytest -- FULL suite (incl. slow tests); run locally
    make test-fast  # pytest -m "not slow" -- lightweight suite (mirrors CI)
    make format     # apply canonical Ruff formatting to src/ and tests/
    make ci         # format check + lint + mypy + LIGHTWEIGHT tests (-m "not slow")
    make demo-eval # idempotent end-to-end: venv -> gold set -> index ->
    validate -> prep-models ->
    run-eval+telemetry
    make mlflow # review all mirrored experiment runs at http://127.0.0.1:5000

`make demo-eval` runs the whole pipeline in order and is **idempotent** -- the venv is reused,
the committed gold set is validated, the index is rebuilt from its matching committed corpus,
and cached model downloads are skipped. It tees per-step output to `.data/llb/logs/pipeline-
<ts>.log` and, on failure, names the failing step + log path. It needs a running Ollama for
the final telemetry run. Every command shares one runtime (`llb.runtime`): Ctrl-C shuts down
cleanly (exit 130, backends killed via their context manager) and an unexpected crash is
logged with a traceback (`LLB_LOG=debug` for more) instead of a raw stack dump.

Extras (`rag, eval, track, board, prep, telemetry, goldset, mcp, dev`) are all installed by
`make venv`; trim with `EXTRAS=` (e.g. `make venv EXTRAS=rag,eval`). GitHub CI installs
only `.[dev]` (it never runs `make venv`), so it stays light. vLLM/torch/flash-attn are
hardware-matched and installed via a separate path per AGENTS.md, never declared here.

Gitignored: `.data/` (runtime output), `.env` (secrets), `.venv/`.

## Repo layout (current)

    pyproject.toml # package "llb": deps + extras, pytest/ruff config
    Makefile # venv, test, ci, build-index, validate-retrieval, run-eval,
      # prep-models, calibration-{run,rate,score}, judge-experiment, ingest-uk-squad, ...
    .env.example                   # DATA_DIR + frontier-API key placeholders
    samples/                       # COMMITTED DATA (kept separate from code)
      rag_items_uk.json            #   sample RAG spec: source docs + item defs
      squad_uk_fixture.json        #   tiny SQuAD-format parser fixture
      goldsets/ua_squad_postedited_v1/
        goldset.jsonl # 250 canonical verified public-development items
        corpus/                    #   250 exact source documents
        source.json # pinned revision, source digest, selection rule
      corpus/ip_regulation_uk.md # substantial UA domain doc (IP regulation)
      for chunking
      text_analysis_bundle_uk/ # committed text-analysis fixture (corpus/ + labels)
      verification/composite_samples/ # committed sample verification refs for composite smoke runs
    scripts/
      shared/common.sh # shared bootstrap + canonical max_jobs() helper
      (AGENTS.md)
      gen_rag_items.sh             # thin entrypoint -> llb.prep.gen_rag_items
      build_vllm.sh # uv-shared prebuilt install or one checkout-built wheel
    src/llb/
      build/vllm.py # uv-shared installer + checkout wheel builder
      config.py # RunConfig (Pydantic) -- the canonical run config
      contracts.py                 # shared TypedDict boundary contracts
      paths.py # project root, .env, and DATA_DIR path resolution
      main.py # thin Typer entry point -> llb.cli.app
      cli/ # Typer app root (app.py) + per-area command modules:
      # eval, models, prep, rag, bench, inference, ui, helpers
      env.py # canonical env-var names (single source of truth)
      fsutil.py # atomic_write_text -- safe whole-file rewrites (temp + os.replace)
      runtime.py # shared CLI runtime: graceful Ctrl-C (exit 130) + crash
      logging
      goldset/schema.py # GoldItem + SourceSpan (Pydantic), load/dump
      goldset/splits.py            # deterministic disjoint split assignment
      goldset/validate.py          # corpus-grounded validator + CLI
      prep/gen_rag_items.py        # spec -> seed gold set
      prep/ingest_squad.py # SQuAD-format (local or HF) -> canonical gold items
      prep/ua_squad_source.py # pinned reviewed-source identity + selection
      policy
      prep/goldset_skeleton.py # timestamped from-scratch authoring template
      prep/published_goldset.py # strict builder for the pinned committed
      fixture
      judge/calibration.py # Spearman rho + CI + trust decision + worksheet
      rag/chunking.py # fixed/sentence/recursive/markdown/semantic chunking
      (offset-exact)
      rag/{embedding,index,store}.py # pinned embedder + FAISS index + store
      (flat / parent-child)
      rag/retrieval.py # recall@k / MRR by source-span overlap (pure)
      backends/{base,openai_client,ollama,vllm}.py # launcher iface + chat call
      + Ollama + vLLM
      backends/{hardware,prepare,planner,telemetry}.py # GPU/RAM detect +
      pull/cache + plan +
      telemetry
      eval/graph.py # LangGraph retrieve->generate flow + failure taxonomy
      scoring/{correctness,judge,aggregate}.py # objective + semantic + gated
      judge + N-model board
      tracking/{manifest,mlflow,server}.py # canonical artifacts + MLflow
      mirror/UI
      executor/{cases,reporting,runner,vram,isolation}.py # per-case work +
      reporting + sweep
      isolation
      backends/resolver.py # M3.2 AvailabilityResolver (discovery + backend
      priority + fit)
      optimize/tuner.py # M3.4 two-stage Optuna (tuning-split search -> stage-2
      entry)
      screen/public.py # M3.1 Tier-1 lm-eval-harness-uk adapter
      (logprob/generation tracks)
      prep/frontier.py # M3.5 prepare-goldset + prepare-synthetic-corpus
      (litellm)
      prep/ontology/ # M4.4 ontology-assisted draft pipeline (7 grained stages
      + endpoint)
      board/{data,app}.py # M3.7 thin Streamlit leaderboard over the run
      bundles
      backends/preflight.py # M4.3 flashinfer JIT-sampler preflight
      inference/generate.py # batch generation helper (cli inference)
      eval/{common,map_reduce,multi_hop}.py # M1.4 eval templates (map-reduce / multi-hop)
      judge/{experiment,rate}.py # M3.8 UA judge smoke + interactive calibration rater
      bench/{security,tooling,agentic,tool_world,structured,summarization,text_analysis,common}.py
      # M5 category benchmarks (objective floor + opt-in gated judge)
      bench/{mcp_server,agentic_tasks}.py # M5.2 MCP transport + M5.3 real-corpus search tasks
      scoring/{security,tooling,structured,reliability,text_analysis,composite}.py
      # M5 per-category scorers + guarded composite headline
      prep/{cross_check,verified_ledger,text_analysis_corpus}.py # 2nd-frontier gate / verified
      ledger / synthetic planter
      prep/{security_sources,security_planter}.py # M5.1 public-set adapters + corpus planter
      prep/{tooling_sources,chat_corpus}.py # M5.2 BFCL adapter + M5.4 chat-period producers
      prep/ontology/spacy_adapter.py # M5.6 opt-in spaCy uk_core_news extraction adapter
    tests/                         # pytest suite (run via make test / make ci)

Shared runtime data is gitignored under `$DATA_DIR/llb/` (default `.data/llb/`):
`corpus/`, `goldset/*.jsonl`, `rag/` (chunks + FAISS index), and generated calibration worksheets
(`$DATA_DIR/llb/calibration/`). The CANONICAL calibration worksheet, by contrast, is TRACKED under
the repo-root `calibration/` dir (committed -- survives a clone; see `calibration/README.md`).
Immutable eval artifacts are isolated per invocation under
`$DATA_DIR/run-eval/<UTC timestamp>-<run id>/` (`manifest.json`,
`scores.{parquet,jsonl}`, and optional `vllm/` logs); M4.4 draft bundles under
`$DATA_DIR/prepare-goldset/<UTC timestamp>/` (`goldset.jsonl`, `corpus/`, `ontology.json`,
`extraction.jsonl`, `provenance.json`).

## Operator workflows (delivered tooling -- run as needed, not plan items)

These are reproducible `make` flows an operator runs repeatedly (e.g. when onboarding a new corpus
or model); they are NOT open plan items. Run the linked guide when a task needs one:

- **Create a new gold set (end-to-end):** [`goldset-from-scratch.md`](../../guides/goldset-from-scratch.md).
- **Second-frontier cross-check + MH.5 human sample-verify:**
  [`verification-tooling.md`](../../guides/verification-tooling.md).
- **Judge calibration (incl. a harder split):** [`calibration-tooling.md`](../../guides/calibration-tooling.md).
- **Graph-vs-FAISS retrieval comparison:** [`graph-vs-faiss-comparison.md`](../../guides/graph-vs-faiss-comparison.md).
- **Composite headline close-out:** [`composite-headline.md`](../../guides/composite-headline.md).

**Data gate (always):** a NEW gold set or corpus runs the
[data-verification workflow](../../guides/goldset-from-scratch.md) before any `verified=true` item
scores a real model (needs a real CUDA host). The objective category boards do not depend on the
gated judge. The committed `ua_squad_postedited_v1` set is already verified + calibrated, so it needs
no further data gate.
