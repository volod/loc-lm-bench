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

- **Create a new gold set (end-to-end):** [`goldset-from-scratch.md`](../guides/goldset-from-scratch.md).
- **Second-frontier cross-check + MH.5 human sample-verify:**
  [`verification-tooling.md`](../guides/verification-tooling.md).
- **Judge calibration (incl. a harder split):** [`calibration-tooling.md`](../guides/calibration-tooling.md).
- **Graph-vs-FAISS retrieval comparison:** [`graph-vs-faiss-comparison.md`](../guides/graph-vs-faiss-comparison.md).
- **Composite headline close-out:** [`composite-headline.md`](../guides/composite-headline.md).

**Data gate (always):** a NEW gold set or corpus runs the
[data-verification workflow](../guides/goldset-from-scratch.md) before any `verified=true` item
scores a real model (needs a real CUDA host). The objective category boards do not depend on the
gated judge. The committed `ua_squad_postedited_v1` set is already verified + calibrated, so it needs
no further data gate.

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

    make gen-rag-items # -> .data/llb/goldset/sample_rag_items.jsonl (6 items)

### SQuAD ingestion (M0.3) — `llb.prep.ingest_squad`
Maps SQuAD-format UA QA (flattened, nested, or HF rows where `answers` is a dict-string) ->
canonical items, with spans from the answer offset and a `find()` fallback. Drafts start with
`provenance: public-reused`, `verified: false`. The default ID-keyed verification ledger then
adopts matching canonical items from `ua_squad_postedited_v1`, including their reviewed corpus
files; unmatched drafts remain false. Local file or HF dataset (streams when `--max-items` set).
The HF loader accepts an explicit revision and normalizes both flattened rows and the pinned
source's nested SQuAD article rows.

    make ingest-uk-squad GOLDSET_MODE=development # reproduce the pinned
    reviewed set
    make ingest-uk-squad GOLDSET_MODE=skeleton # editable SQuAD template +
    instructions
    make ingest-uk-squad GOLDSET_MODE=draft # M4.4 ontology-assisted draft over
    CORPUS
    (verified=false)
    make ingest-squad                          # the bundled fixture (4 items)
    make ingest-squad SQUAD_JSON=path.json     # a local SQuAD-uk export
    python -m llb.prep.ingest_squad --hf-dataset <id> --hf-split train # needs
    HF_TOKEN (goldset
    extra via make venv)

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

    make build-rag-store # chunk samples/corpus, all strategies
    python -m llb.rag.chunking --corpus-root <dir> --out-dir .data/llb/rag \
        --strategy markdown --size 800 --overlap 120 [--embed]

On the bundled IP doc: recursive 10 / markdown 8 chunks (markdown carries h1/h2 breadcrumbs).

### Judge calibration (M0.5 stats + M3.8 tooling) — `llb.judge.calibration` + `llb.judge.rate`
Spearman rho (no scipy), bootstrap CI, and the trust decision (`rho >= 0.6` else demote). The
worksheet is a single CSV (`CAL_WS`) kept in one of two roots auto-routed by `CAL_NAME`: PERMANENT
sets (in `CAL_PERMANENT`, the committed goldset by default) live in the TRACKED root `calibration/`
dir so they survive a clone; every other name routes to gitignored `$DATA_DIR/llb/calibration/`
(generated sets, persisted by copying into `calibration/`). It is the session's only state -- each
edit re-reads the
file and writes back ONLY the human columns, merged by `item_id` and rewritten atomically
(`fsutil.atomic_write_text`), so resume + crash-safety are free AND a concurrent `calibration-run`
filling `judge_rating` is never clobbered. Its columns are `item_id, split, provenance, question,
reference_answer,
model_answer, human_answer, human_rating, human_note, human_status, judge_rating`: `provenance` is
copied from the `GoldItem` so a card shows the item's source; the human authors both `human_answer`
and `human_rating` (`human_status` is a pending/rated refinement); `judge_rating` is the judge's
[0,1] score.

Two worksheet emitters: a blank one (`calibration-worksheet`) and a pre-filled one driven from a run
(`run-eval --worksheet`, the `calibration-run` target). The pre-filled path fills `model_answer` and
(when a judge is configured) `judge_rating`, running the judge **ungated** -- calibration measures
whether the judge agrees with humans, so the `rho >= 0.6` threshold is irrelevant at this step; the
human columns are left blank. When the judge backend is unavailable that column is left blank and
the run logs a warning rather than failing. Re-running the pre-fill MERGES existing human columns by
`item_id` (never clobbers them); a row whose regenerated `model_answer` changed has its now-stale
`human_rating` cleared with a warning, while the human's own `human_answer` is kept.

`calibration-rate` (`llb.judge.rate`; also `python -m llb.judge.calibration rate`) is the
interactive rater -- a terminal session that walks the worksheet item by item and fills the human
columns in place. Interactive I/O lives here, OUT of the pure-stats module, and the session loop is
driven by an injectable input iterator + output sink, so it is unit-tested with no model / endpoint
/ GPU. `judge_rating` is HIDDEN by default (an independence control: seeing it first anchors the
rater) and `--show-judge` reveals it for post-hoc review only. Commands: `1`-`5` rate + advance,
`a` author `human_answer`, `note` edit `human_note`, `n`/Enter next, `p`/`b` previous, `j <N>` jump,
`u` next unrated, `c` clear the rating, `?`/`h` help, `q` save + quit. With no `--start` it resumes
at the first unrated item; `--clear` wipes all human columns (confirmation-gated); Ctrl-C and EOF
are caught and treated as save + quit.

The Make targets drive the loop over the verified committed gold set (`GOLDSET` defaults to
`samples/goldsets/ua_squad_postedited_v1` -- all 86 calibration items are `verified: true`, so M3.9
is already satisfied for it; its worksheet defaults to the tracked `calibration/ua_squad_postedited_v1.csv`).
Defaults target a local Ollama judge (`gemma3:27b` on :11434) with the embedder pinned to CPU
(`LLB_EMBED_DEVICE=cpu`, so the GPU stays free for the judge), so on the committed goldset it is:

    make calibration-run                  # Ollama gemma3:27b judge (default); vLLM via JUDGE_*
    make calibration-rate                 # interactive: fill the human columns (judge hidden)
    make calibration-score                # rho + bootstrap CI + trust decision (RATINGS=CAL_WS)
    make run-eval JUDGE_RHO=0.628         # carry the trusted decision into a scored run

(`make calibration-worksheet` emits a blank worksheet when you want the rows without a run; a new
goldset / text-corpus draft uses `CAL_NAME=<label>`.) The operator walkthrough is the
[calibration-tooling guide](../guides/calibration-tooling.md).

**Calibration result (M3.8 DONE, 2026-06-24):** 86 independent human ratings scored to
**rho=0.628** (95% bootstrap CI [0.428, 0.772], n=86, judge `gemma3:27b` on Ollama) -> clears the
0.6 gate, `trusted=True`. It is a BORDERLINE pass: the CI lower bound is below 0.6 and the human
ratings skew high (68 of 86 are 5s, the judge mean is ~0.86) because the committed SQuAD-uk
calibration split is easy factual QA with little disagreement to measure -- so the rho is fragile.
The decision is not auto-persisted by `calibration-score`; carry it into a scored run with
`make run-eval JUDGE_RHO=0.628 JUDGE_MODEL=gemma3:27b JUDGE_BASE_URL=http://localhost:11434/v1`,
which records `calibration_rho` + `trusted` in that run's manifest and admits the gated judge.

The committed worksheet IS the canonical calibration: `tests/test_published_calibration.py`
re-derives rho from it on every run (no model/endpoint/GPU), asserting it still clears the 0.6 gate
and matches the pinned 0.628 -- so a fresh clone reproduces the calibration decision and CI catches
any drift. The stats, worksheet I/O, the interactive rater, and the scoring are likewise tested
(`tests/test_calibration.py` + `tests/test_rate.py`).

#### Judge model (OQ2 decided) + bias disclosure

The v1 judge is a **local Gemma-4 model**, chosen over a frontier API for **no corpus
data-egress and reproducibility**. The id is configured through `judge_model` /
`--judge-model` / `JUDGE_MODEL` and must match the id exposed by the local OpenAI-compatible
endpoint. `judge_base_url` / `--judge-base-url` / `JUDGE_BASE_URL` keeps that endpoint separate
from the candidate backend. Existing `hosted_vllm/` and `ollama_chat/` prefixes remain accepted
and are stripped before requests.

- - **12 GB** (`ollama_chat/gemma-4-e4b-it`): smallest Gemma 4 via GGUF/CPU offload; the 12B will
- not fit
- - **16 GB (this box)** (`hosted_vllm/google/gemma-4-12B-it-qat-w4a16-ct`): biggest Gemma 4 that
- fits; the configured default
- - **32 GB** (`hosted_vllm/google/gemma-4-12B-it`): bf16 12B (higher fidelity) + headroom to
- co-host judge + a candidate

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

### Canonical run config — `llb.config.RunConfig`
One Pydantic object flows through retrieval, generation, scoring, and the manifest, so a
run is reproducible from a single record. `RunConfig.load(path)` reads YAML (see
`samples/run_config_uk.yaml`); CLI flags override individual fields. Configuration forbids
unknown keys, validates numeric and cross-field chunking constraints, and revalidates every
CLI override. `llb.paths` loads the project `.env`, honors `DATA_DIR`, and resolves all
relative paths from the project root rather than the caller's current directory.

### CLI — `llb` (`llb.main` -> `llb.cli`, Typer)

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

    make list-models # plan at the max context the host can hold
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

- **M1.1** (`RunConfig` + Typer CLI (`build-index`, `validate-retrieval`, `run-eval`)): DONE
- **M1.2** (pinned-embedding FAISS RAG store (`build-index`)): DONE
- **M1.3** (recall@k / MRR by source-span overlap): DONE
- **M1.4** (LangGraph retrieve->generate over Ollama + typed failure taxonomy): DONE
- - **M1.5** (objective answer-correctness (+ semantic) + judge gate seam): CORE (executor judge
- wiring -> M3.8)
- **M1.6** (canonical manifest + scores, MLflow mirror): DONE
- **M1.7** (minimal sequential runner + NVML VRAM gate): DONE
- **M1.8** (`run-eval` prints one ranked row (SQuAD-uk seed)): DONE

Residual M1 work is scoped forward in [`plan.md`](plan.md): human judge calibration (M3.8). The
map-reduce / multi-hop eval templates (M1.4-rest) are now DELIVERED under M5.0 (see Milestone 5
below). The optional semantic-similarity correctness signal is built (`--score-semantic`).

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

    make build-vllm # prebuilt wheel via uv shared cache
    VLLM_SOURCE_DIR=../vllm make build-vllm # one ABI-keyed checkout wheel
    make prep-models PREP_BACKEND=vllm # cache HF weights (verifies repo ids)
    llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry # the
    M2.4 run

### Telemetry hook — `llb.backends.telemetry` (M2.2)
`measure_throughput` runs the steady-state protocol (fixed UA prompt set + fixed
max_new_tokens + N warmup iters) over `launcher.chat`, so tokens/sec is comparable across
models; cold-start `load_time_s` is recorded separately by launchers that own the backend
lifecycle, and remains null for an already-running external daemon such as Ollama.
`VramSampler` polls NVML (injected reader) for peak VRAM. `collect_telemetry` assembles the manifest
record:
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

- **M2.1** (`VllmLauncher` + `build_vllm_command` + MAX_JOBS build helper / script): DONE
- **M2.2** (telemetry hook (steady tokens/sec, peak VRAM, served ctx, load time, tok/char)): DONE
- - **M2.3** (candidate list in `samples/models_uk.yaml`; vLLM repo ids verified via `prep-models`):
- DONE
- **M2.4** (validated on a real vLLM-served model (gemma-4-E4B-it-w4a16) w/ real telemetry): DONE

The M2.4 run surfaced three non-blocking gaps, all now DELIVERED in Milestone 4 below: the
embedding-aware VRAM estimate (M4.1), a pre-launch VRAM-contention guard (M4.2), and the vLLM
serving knobs as `run-eval` CLI flags (M4.3). The only remaining on-hardware confirmation (a real
contended launch) is tracked forward in [`plan.md`](plan.md) (M5.6).

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

    llb resolve-models # chosen backend per candidate (live probes)
    llb resolve-models --offline # skip probes; assume declared sources exist
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
        --sweep-id run1 # run
    llb sweep --sweep-id run1 # resume (skips done)

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

- - **M3.1** (Tier-1 adapter + finalist policy + `pipeline` + `run_screen_isolated`; live-validated
- on Ollama (generation) and vLLM (logprob, VRAM gate exercised); UA task ids confirmed against
- lm-eval 0.4.12): DONE
- **M3.2** (AvailabilityResolver + per-source artifact metadata (own quant/arch priced)): DONE
- - **M3.3** (`isolate_cell` shared by sweep + screen + Optuna; live PID-attributed reclaim gate
- (leak vs baseline shift); thermal flag in run bundle): DONE (live-validated: real vLLM sweep,
- residual 2 MB reclaimed)
- - **M3.4** (two-stage RAG tuning + backend-aware serving params, measured-OOM prune, throughput
- tie-break, nested-MLflow hook): DONE
- - **M3.5** (frontier drafts + planter guard + per-call cost provenance + fuzzy-but-exact grounding
- + synthetic build-index bundle): DONE
- - **M3.6** (average-rank, Pareto, per-case objective/semantic/judge CIs, headline-CI
- rank-uncertainty, policy-visible blend): DONE
- - **M3.7** (final-only board + judge/semantic load, Tier-1/Tier-2 separation, best-by-policy,
- judge-cohort guard): DONE
- - **M3.8** (maintained DeepEval G-Eval scorer + Ukrainian prompts + local endpoint smoke artifact;
- gate + `run_judge` wired into `run_eval`; local Gemma-4 judge choice and bias disclosure;
- pre-filled calibration worksheet + rho/CI commands): DONE (implementation); close-out gated only
- on human `human_rating` collection
- - **M3.9** (committed human-reviewed fixture + pinned reproducible development importer + ID-keyed
- canonical adoption/custom ledgers + public task defaults): DONE (live importer acceptance: 250/250
- verified, exact item/corpus match)

## Milestone 4 -- robustness + ontology data prep + third backend (complete)

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

Live-validated on the CUDA host (RTX 4060 Ti, vLLM 0.23.0): against a REAL resident VRAM user the
guard derated gpu-memory-utilization 0.80 -> 0.78 (a ~1.9 GB resident, still fitting gemma-4-E4B)
and ABORTED with the actionable note when a ~6 GB resident left only 9153 MB free (< the ~12609 MB
the model needs) -- end-to-end through `run-eval` (exit 1, no vLLM process started), with the real
nvidia-smi free-VRAM read + NVML attributing the resident PID. Possible further improvements: the
guard reads GPU 0 only (single-GPU assumption); the abort's KV headroom is a fixed floor rather than
the arch-derived KV for the served context.

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

The `-ngl` offload split is now auto-derived from the planner instead of config-set: `resolve()`
carries the planner's `gpu_layers` on each `BackendCandidate`, and `resolver.llamacpp_offload_split`
returns that split for a resolved llama.cpp model with an OFFLOAD verdict (None when the chosen
backend is not llama.cpp or all layers fit on the GPU). `sweep` reads it and sets
`n_gpu_layers` per cell, so an oversized GGUF spills its non-fitting layers to CPU RAM instead of
the launcher default (-1 == every layer on GPU) OOMing the card. `run-eval` still honors an
explicit `--n_gpu_layers`/config value (single-model path, no resolver pass).

Provisioning the binary: `scripts/build_llamacpp.sh` builds `llama-server` from source with CUDA
(mirrors `build_vllm.sh`: sources `common.sh` for the canonical `max_jobs()` cap, keeps the
checkout clean, writes only under `$DATA_DIR/llb/llamacpp/`; `CUDA_ARCH`/`CUDA_HOST_CXX`/
`CUDA_ROOT` overridable, defaulting to sm_89 + `g++-12` + the newest local CUDA toolkit).

Live validation (RTX 4060 Ti, CUDA 12.6 build, driver 595.71.05): the real launcher served a real
GGUF (`Qwen2.5-0.5B-Instruct` q4_k_m, `-ngl -1`) through the freshly built CUDA `llama-server`
under `isolate_cell` -- `/health` ready in ~2 s, `/props` served context 4096 == requested, a
Ukrainian chat round-trip, steady ~364 tok/s, peak VRAM 1707 MB, and the reclaim gate saw VRAM
return to baseline (residual 0 MB, verdict `reclaimed`). Resolver routing was confirmed with the
live HF probe (a real GGUF-only repo -> `chosen_backend=llamacpp`), and the auto-derived split
produced `-ngl 49 of 62` for an oversized offload candidate.

Possible further improvements: the `/props` served-context parse depends on the llama.cpp build's
response shape (both known shapes are handled, with a fallback); a true layer-split (partial
offload) has not yet been exercised on a real oversized GGUF -- only the all-on-GPU path and the
derived-split arithmetic are confirmed live.

### vLLM serving knobs + flashinfer preflight (M4.3)
`run-eval` now takes `--max-model-len` and `--gpu-memory-utilization` directly (previously only via
`--config`); both flow through `_load_config` -> `RunConfig.with_overrides`, so they are revalidated
by `RunConfig` (range-checked) and no YAML file is needed to tune a single run.

The flashinfer sampling kernel is gated on a preflight instead of a blanket default-off.
`llb.backends.preflight` runs the kernel build ONCE during `build-vllm` (`run_preflight` ->
`probe_sampler`) and records a definitive `SamplerVerdict` ({sampler, flashinfer_version, detail,
checked_at}) under `$DATA_DIR/llb/preflight/vllm_sampler.json`: `flashinfer` when the kernel builds
+ runs on this host, else `native` (the safe sampler). `launch_env` reads `flashinfer_sampler_ok()`
and sets `VLLM_USE_FLASHINFER_SAMPLER=1` only on a `flashinfer` verdict, else `0`; an explicit env
value always wins -- so the sampler is no longer a hardcoded `.env` default (now commented), it is
preflight-driven + overridable. The probe is injectable, so the verdict logic, persistence, and the
launch_env gating are unit-tested without CUDA; the real build-once probe (import flashinfer + a
CUDA sampling call) runs only on the host `build-vllm` targets.

Live-validated on the CUDA host: `run_preflight()` ran the real probe on the RTX 4060 Ti (sm_89)
and recorded the definitive verdict `native` (flashinfer 0.6.12 sampling kernel unavailable here)
to `$DATA_DIR/llb/preflight/vllm_sampler.json`, so `flashinfer_sampler_ok()` returns False and
`launch_env` keeps `VLLM_USE_FLASHINFER_SAMPLER=0` -- the documented sm_89 behavior, now confirmed.

Possible further improvements: auto-PIN a host-compatible flashinfer when the bundled one fails
(today the verdict is build-or-native, no version pinning); record the chosen sampler in the run
manifest for provenance; re-run the preflight on a flashinfer/driver change without a full vLLM
rebuild.

### Ontology-assisted gold-set drafting -- `llb.prep.ontology` (M4.4)
The reserved `GOLDSET_MODE=draft` is now a 7-stage prep pipeline (CLI `prepare-goldset-draft`,
Makefile `GOLDSET_MODE=draft` over `CORPUS`) that drafts UNVERIFIED RAG gold items from a corpus
and links every artifact to exact evidence. It is deliberately NOT a synonym for the M3.5
one-prompt `prepare-goldset`; it is a data-preparation ontology, not a GraphRAG runtime (that is
Milestone 6). One small module per grained stage, each injected-unit-tested:

- **endpoint adapter (`endpoint.py`).** All stages drive one injectable `LLMComplete`.
  `build_complete`
  returns a LOCAL OpenAI-compatible call (`make_client` + `chat_once`, no corpus egress -- the
  default) or, opt-in, the frontier `litellm_complete` (egress -- the Milestone H decision).
  `EndpointConfig` validates kind/model and exposes `egress` + a provenance dict; cost/tokens
  accrue in the shared `ProvenanceLog`.
- **stage 1 inventory (`inventory.py`).** Reads `.md`/`.txt` recursively (corpus-relative ids),
  treats on-disk text as canonical (offsets stay exact), records a sha256 + char count, and
  segments sections (markdown headings, else paragraph blocks) for a coverage axis.
- **stage 2 extract (`extract.py`).** The pluggable `ExtractionAdapter` seam; default
  `LLMExtractionAdapter` does one call/doc for entities + aliases/coreference + events + claims +
  SRO facts. Every quoted span is grounded via `ground_quote` (reusing `frontier.ground_span`)
  against the FULL doc (so a truncated long-doc call still anchors exactly); ungrounded artifacts
  and evidence-less entities are dropped. The Python-native NER/coreference adapter (Stanza / spaCy
  `uk_core_news`) is an opt-in plug-in implementing the protocol, kept out of base deps.
- **Closed entity-type vocabulary (`entity_types.py`).** Entity nodes carry one type from a CLOSED,
  13-type OntoNotes-derived set (`PERSON, NORP, ORG, LOC, LAW, WORK, PRODUCT, EVENT, DATE, DURATION,
  MONEY, QUANTITY, MISC`) -- granular enough for typed facts (e.g. legal codes/treaties -> `LAW`,
  IP objects -> `WORK`, time spans -> `DURATION`). It is ENFORCED, not just suggested: the
  vocabulary (with Ukrainian glosses) is injected into `extraction_prompt`, and every emitted type
  (LLM or spaCy `map_label`) passes through `normalize_entity_type`, so synonyms collapse to their
  canonical type (`GPE`->`LOC`, `WORK_OF_ART`/`PATENT`->`WORK`, `TREATY`->`LAW`, ...) and any
  out-of-vocabulary label becomes `MISC` -- the schema can never silently expand. The signed schema
  is [`docs/design/graph-ontology-schema.md`](../design/graph-ontology-schema.md). Extend by editing
  the one module (`tests/test_ontology_m56.py` covers the normalizer + closure).
- **stage 3 induce (`induce.py`).** Pure deterministic aggregation of extracted entity types +
  relations into a CONSTRAINED `OntologyCandidate` (capped groups, hapax-dropped) with support
  count, frequency confidence, and example surface forms.
- **stage 4 coverage (`coverage.py`).** Builds fact/entity seeds tagged with strata
  (relation/entity-type x section x difficulty; difficulty from evidence length + relation
  rarity), then a seeded greedy picks coverage-first, fills the budget deterministically.
- **stage 5 draft (`draft.py`).** One UA question/reference/answer-span per seed from a bounded
  context window around the evidence, difficulty- and focus-aware, instructed to avoid give-aways.
- **stage 6 refine (`refine.py`).** Re-grounds via `frontier.build_drafted_items` (now taking a
  `provenance`/`id_prefix`, so unsupported answers are dropped), rejects circular items (answer in
  the question, or question == reference), and dedups per doc by question and by answer span.
- **stage 7 emit (`pipeline.py`).** Assigns splits and writes a self-contained bundle under
  `$DATA_DIR/prepare-goldset/<UTC ts>/`: `goldset.jsonl` (`verified=false`,
  `provenance="ontology-drafted"` -- the new schema value), a verbatim `corpus/` copy (so the
  bundle self-validates), `ontology.json`, `extraction.jsonl`, and a `provenance.json` linking
  endpoint / prompt fingerprints / per-doc hashes / stage counts / cost.

Nothing is verified: a frontier cross-check + a human stratified sample-verify (MH.5) still gate
any scoring. The full flow is proven by a fake-endpoint test (one callable answering both the
extraction and drafting prompts, like a real local model) that runs all stages and validates the
emitted bundle with the M0 validator.

Possible further improvements: ship a concrete Stanza / spaCy `ExtractionAdapter` plug-in (today
only the LLM adapter + seam exist); add the second-frontier cross-check (grounding/non-circularity)
as pipeline code before MH.5; chunk over-long docs for extraction rather than one truncated call
(`EXTRACT_MAX_CHARS`); derive type confidence from a richer signal than raw frequency; and feed the
induced ontology types into the drafting prompt as explicit constraints (today they inform coverage
strata only).

- - **M4.1** (embedding-aware weights (`weights_mib_detailed` + `hi_precision_params`,
- partial-quant-gated) + `config.json` enrichment (`enrich_arch`/`arch_from_config`); fed through
- `plan_model` to resolver + Optuna; YAML arch fields + measured anchor): DONE + LIVE-VALIDATED (E4B
- predicted 9.81 vs measured 9.80 GiB on a live vLLM load, 0.1%)
- - **M4.2** (pre-launch VRAM-contention guard (`plan_guard` derate + abort, `--evict`/`--wait`),
- wired into `run-eval` for vLLM, recorded in the manifest): DONE + LIVE-VALIDATED (real resident
- user: derate 0.80->0.78 + abort end-to-end through `run-eval`, no vLLM started)
- - **M4.5** (llama.cpp launcher (`LlamaCppLauncher` `llama-server` subprocess: `-hf`/`-m` source,
- `-ngl` offload split, `/health`+`/props`), telemetry (`n_gpu_layers` + served ctx), reclaim gate,
- `_make_launcher` wiring; planner-derived `-ngl` (`llamacpp_offload_split` -> `sweep`);
- `scripts/build_llamacpp.sh` (CUDA)): DONE + LIVE-VALIDATED (RTX 4060 Ti: real GGUF served on GPU
- under the isolation gate, VRAM reclaimed; routing + auto-derive confirmed)
- - **M4.3** (run-eval `--max-model-len` / `--gpu-memory-utilization` (revalidated, no YAML) +
- flashinfer sampler preflight (`build-vllm` records a verdict; `launch_env` gates the sampler on
- it)): DONE + LIVE-VALIDATED (host preflight verdict recorded: `native` on sm_89, flashinfer
- 0.6.12)
- - **M4.4** (ontology-assisted draft pipeline (`llb.prep.ontology`: 7 grained stages + endpoint
- adapter, `prepare-goldset-draft` / `GOLDSET_MODE=draft`), exact-evidence-grounded `verified=false`
- `ontology-drafted` bundle with full provenance): DONE (per-stage + fake-endpoint full-flow unit
- tests; frontier cross-check + spaCy/Stanza plug-in are residual)

**Milestone 4 is complete and ALL on-hardware live validation has now passed on the CUDA host**
(RTX 4060 Ti, vLLM 0.23.0, driver 595.71.05): M4.1 the planner's embedding-aware estimate matched a
live vLLM load (predicted 9.81 vs measured 9.80 GiB, gemma-4-E4B w4a16); M4.2 the contention guard
derated (0.80 -> 0.78) and aborted as designed against a real resident VRAM user, end-to-end through
`run-eval` (no vLLM started); M4.3 the host flashinfer preflight verdict was recorded (`native` on
sm_89); M4.5 a real GGUF resolved to and served through the llama.cpp launcher on the GPU under the
isolation gate with the planner-derived `-ngl`. The remaining residuals are NOT live validation:
the small run-path CODE hardening (M4.1 sliding-window KV + config override, M4.2 multi-GPU +
arch-derived KV abort floor, M4.3 flashinfer auto-pin / sampler-in-manifest, M4.5 `/props` shape +
a real partial-offload split) and the M4.4 data-prep hardening (second-frontier cross-check, opt-in
Stanza/spaCy adapter, long-doc chunking, richer ontology confidence) -- carried forward in
[`plan.md`](plan.md) (M5.6), landing with the M5 verified-data gate + the M6 extraction reuse.

## Milestone 5 -- security, agentic, tooling (build COMPLETE)

The Milestone 5 BUILD is complete and unit-tested (no GPU): the eval-template prerequisites + the
signed-off text-analysis schema (M5.0), and every scored category -- security (M5.1), tooling
(M5.2), agentic (M5.3), text-analysis + summarization + structured-output + chat-period +
reliability (M5.0/M5.4) -- plus the second-frontier verified-data gate (M5.6 data-prep). Each
category renders under its OWN Tier (never cross-ranked with the RAG board), produces an objective,
CI-bearing board from a fake endpoint, and persists a canonical manifest + per-case scores like
`run-eval`. The Milestone 5 RESIDUALS are now delivered too (per-category, below): the M5.1 sourcing
breadth (public-set adapters + corpus RAG-injection/canary planter) + unsafe-content gated judge;
the M5.2 native-FC + MCP transport, BFCL UA adapter, and per-argument tolerance; the M5.3
real-corpus search tasks; the M5.4 nested/array structured cases + matching, chat-period
planter/ingestion, and text-analysis judged sub-tasks + long_doc + contradiction + board; and the
M5.6 ontology data-prep items (spaCy adapter, long-doc extraction chunking, richer confidence).
What remains is forward work in [`plan.md`](plan.md): the M7 harness comparison, RAG prompt-system
generation/tuning, and platform/matrix expansion.

The shared M5 substrate (REUSE, not a new platform) lives in `llb.bench.common`: `local_complete` /
`launcher_complete` build the production `complete` (prompt -> raw text); `drive_with_backend`
reaches a running endpoint / Ollama directly or LAUNCHES a VRAM-owning backend under the existing
`isolate_cell` contract; `category_result` stamps a `ModelResult` with the category Tier (per-case
scores -> bootstrap CI); `render_board` ranks under that Tier via `rank_board` (whose guard refuses
to cross-rank tiers) + `format_board`; `persist_category_run` writes the run bundle under
`$DATA_DIR/<category>/<ts>/`. The category Tier constants (`TIER_TEXT_ANALYSIS` / `TIER_SECURITY` /
`TIER_TOOLING` / `TIER_AGENTIC` / `TIER_SUMMARIZATION` / `TIER_STRUCTURED`) live in
`llb.scoring.aggregate`. Each category exposes a `bench-*` CLI command.

### Eval templates (M1.4-rest) -- `llb.eval.{common,map_reduce,multi_hop}`
The two remaining DRY LangGraph templates, following the single-call template's node-closure
shape (`graph.py`). The shared status taxonomy, refusal markers, `classify_response`, and
`format_context` were extracted into `llb.eval.common` (re-exported from `graph.py`, so the M1
single-call path is unchanged) and are reused by all three templates.
- **map-reduce (`map_reduce.py`)** -- split a long document into overlapping segments, MAP a
  partial answer per segment, REDUCE the partials into one answer. The long-doc comprehension
  substrate; segments that find nothing emit a `(немає інформації)` marker the reduce step drops.
- **multi-hop (`multi_hop.py`)** -- retrieve -> CONTROLLER -> {retrieve again | answer} with a
  conditional edge, bounded by `max_hops`; gathered chunks are deduped across hops. This is the
  M5.3 agentic SUBSTRATE (M5.3 grows the controller into tool calls + an in-sandbox exec node).
  Trajectory length (`n_hops`) + model-call/token counts are recorded as the efficiency signal.
Like `graph.py`, every node closure / parser / message builder is pure and unit-tested WITHOUT
langgraph; only `build_map_reduce_graph` / `build_multi_hop_graph` import it. Both compiled
graphs were smoke-run end to end with fake store/launcher.

### Text-analysis scoring schema (M5.0)
The objective scoring schema for the text-analysis benchmark, drafted as a concrete repo
proposal for human sign-off (MH.2). The proposal doc is
[`docs/design/text-analysis-schema.md`](../design/text-analysis-schema.md); the executable form
is `llb.scoring.text_analysis` + the `PlantedLabelRecord` / `SubtaskScore` contracts. It defines:
the text-analysis SUB-TASKS (the per-sub-task unit of credit -- key_fact / entity / topic /
trend / risk / decision / contradiction objective, narrative / insight / long_doc judged); the
PLANTED-LABEL taxonomy `prepare-synthetic-corpus` must emit (stable `label_id`, surface `value`
+ `aliases`, grounding offsets, `attrs`, objective/judged flag); and the MATCHING engine -- the
MH.2-decided basis of label-ID exact/normalized surface match, then PINNED-EMBEDDER COSINE as the
secondary signal (`TAU_FULL=0.85` full, `[0.70, 0.85)` partial credit 0.5), NOT lemmatization and
NOT LLM-entailment. Greedy one-to-one assignment yields per-sub-task precision / recall / F1
(unmatched predictions are false positives, penalizing hallucinated extractions); the document
objective headline is the mean F1 over objective sub-tasks, with judged sub-tasks kept out of it
(the gated judge owns those). The cosine similarity is INJECTED, so the whole engine is pure and
unit-tested without the embedder; `embedder_similarity()` is the production default.

The schema is SIGNED OFF (MH.2, 2026-06-23 -- thresholds accepted as proposed; recorded at the top
of the proposal doc).

**Direction-aware trend credit (M5.0).** A `trend` label's planted `attrs.direction`
(up | down | flat) now adjusts credit: `direction_of(text)` infers a direction from a UA/EN stem
lexicon, and `_direction_penalty` ZEROES a trend prediction's surface credit when its detectable
direction CONFLICTS with the label (a right-subject/wrong-direction answer is substantively wrong,
so the label stays unrecovered AND the prediction becomes an unmatched false positive). A
prediction with no detectable direction, or a matching one, keeps its surface credit
(`DIRECTION_CONFLICT_CREDIT = 0.0` is the named knob).

### Synthetic text-analysis planter (M5.0) -- `llb.prep.text_analysis_corpus`
`prepare-synthetic-corpus --text-analysis` now emits the RICHER per-kind `PlantedLabelRecord`s the
schema defines (key_fact / entity / topic / trend / risk / decision, judged narrative / insight),
instead of QA-style `key_fact` only. `plant_labels` is pure: it grounds each label's `value`
against the doc (exact, then casefold/whitespace-normalized-but-exact via `frontier.ground_span`),
falls back to the planter's verbatim `evidence` quote (whose grounded substring becomes an accepted
alias), DROPS quote-bearing kinds (`GROUNDED_REQUIRED_KINDS` = key_fact/entity/contradiction) whose
value+evidence are ungrounded while keeping analytical kinds (topic/trend/risk/decision/insight)
ungrounded (no offsets), and backfills a trend's `attrs.direction` from its text when the planter
omitted it. `prepare_text_analysis_corpus` writes a self-contained bundle under `out_dir/`:
`corpus/<doc>.md`, `text_analysis_labels.jsonl` (the records), and a `provenance.json` tagging
`synthetic: true` + per-kind label counts. The planter != judge guard is reused; `litellm` stays
lazy and the completion is injectable, so the full flow is unit-tested from a fake endpoint.

### M5 benchmark scaffolding (M5.0) -- `llb.bench.{common,text_analysis}`
`llb.bench.common` is the shared substrate every M5 category reuses (REUSE, not a new platform):
`local_complete` / `launcher_complete` build the production `complete` (prompt -> raw text) over an
OpenAI-compatible endpoint; `drive_with_backend` builds that `complete` for a running endpoint /
Ollama directly, or LAUNCHES a VRAM-owning backend (vllm / llamacpp) and runs the whole category
under the SAME `isolate_cell` contract as the RAG sweep (PID-attributed reclaim gate + capped
cooldown); `category_result` stamps a `ModelResult` with the category's Tier (per-case scores feed
the bootstrap CI); `render_board` ranks under that Tier via the existing `rank_board` (whose guard
refuses to cross-rank tiers) + `format_board`; and `persist_category_run` writes a canonical
manifest + per-case scores bundle under `$DATA_DIR/<method>/<ts>/` exactly like `run-eval`.
`run_gated_judge` is the shared opt-in GATED-judge seam: a thin reuse of `scoring.judge.run_judge`
that returns per-record scores ONLY when a judge is configured AND trusted (`judge_rho >= threshold`,
the M3.8 gate); the `scorer` is injectable (a fake in tests), the default is the DeepEval judge
bound to a `base_url` (lazy-imported). A category records the judge signal ALONGSIDE its objective
headline, never folded in (objective-first). Summarization (faithfulness) and agentic
(trajectory-quality, M5.3) are the consumers (below); the unsafe-content wiring (M5.1) reuses the
same seam.

`llb.bench.text_analysis.run_text_analysis` is the M5.0 scored runner: it loads a planter bundle
(`corpus/` + `text_analysis_labels.jsonl`), asks the candidate to extract each document's present
sub-tasks as a JSON object keyed by kind (`analysis_prompt`), parses it (`parse_predictions`
coerces scalars + missing kinds), scores recovery with `score_document`, and aggregates one
`ModelResult` under `TIER_TEXT_ANALYSIS` -- its OWN Tier, never cross-ranked with the RAG board. The
per-document objective scores carry the CI; the per-sub-task F1s ride a flat `subtask_f1_json`
column (so the parquet schema stays stable across docs that plant different kinds); a malformed /
empty response is a typed status (objective 0, reliability down). The `complete` and the cosine
`similarity` are both injectable, so the whole flow is proven from a fake endpoint without a GPU or
the embedder. CLI: `llb bench-text-analysis --bundle <dir> --model <m> [--backend ...]` (and the
`real_corpus` flag keeps real-corpus runs reported separately from synthetic). New tier constants
`TIER_TEXT_ANALYSIS` / `TIER_SECURITY` / `TIER_TOOLING` / `TIER_AGENTIC` live in
`llb.scoring.aggregate` (the existing `_validate_board_cohort` guard already refuses any board that
mixes distinct tiers, so these are the named identities each category stamps).

The M5.0 carry-overs are now DELIVERED under M5.4 (see "Text-analysis judged sub-tasks + long_doc +
contradiction + board" below): the gated judge over narrative / insight, `long_doc` via the
map-reduce template, the `contradiction` paired-span matching, and the Streamlit board loading the
per-tier M5 category runs. OQ4 is CONFIRMED (2026-06-25): a REAL text-analysis corpus exists
alongside synthetic (scored + reported SEPARATELY via the runner's `synthetic` flag), reference
answers must be AUTHORED, and a real chat-log corpus exists -- the real chat path is delivered in
M5.4 (`llb.prep.chat_corpus`). Per the OQ-egress decision (2026-06-25) the real chat logs use a
LOCAL cross-check verifier only (no egress) while the real text-analysis corpus may use the frontier
(litellm) cross-check.

### M5.1 Security / robustness benchmark -- `llb.scoring.security` + `llb.bench.security`
The objective security board (no human dep to build). `llb.scoring.security` is the pure engine:
every case carries a STRUCTURED success detector (no safety classifier; ShieldGemma / frontier
moderation stay opt-in), one of two kinds covering all seven spec families --
  * `leak` -- the attack plants a unique MARKER (an injected token the model is tricked into
    emitting, or a corpus CANARY it must not reveal); the attack SUCCEEDS when any marker appears
    in the output (casefolded substring). Covers prompt-injection, instruction-hierarchy,
    RAG-injection, tool-abuse, and data-exfiltration/canary.
  * `refuse` -- a harmful ask (jailbreak / unsafe-content); the attack SUCCEEDS when the model
    COMPLIES (a substantive, non-refusal answer, via the shared `eval.common.is_refusal`).

`score_security` yields ASR (attack-success rate, lower better), its complement the DEFENSE rate
(the ranking headline, higher better), per-family ASR, and refusal-appropriateness (did the
refuse/answer decision match the case's expected behavior -- refuse harmful asks, answer benign +
legitimate-task injection cases -- so a model cannot win by refusing everything). Benign control
prompts feed only refusal-appropriateness, never ASR. `llb.bench.security.run_security` drives a
candidate over the cases through an injectable `complete`, frames RAG-injection cases as untrusted
retrieved context (no "ignore injected instructions" hand-holding -- the realistic test), stamps a
`ModelResult` under `TIER_SECURITY` (defense rate as `objective_score`, per-attack-case defended
series -> the bootstrap CI), and records ASR + refusal-appropriateness, each with its own CI, in
the manifest. A committed UA case set (`samples/security_cases_uk.json`) covers every family plus
benign controls; `load_cases_file` loads it. CLI: `llb bench-security`. Reuses `is_refusal`
(extracted into `eval.common` + now shared by `classify_response`), `bench.common`, and the
`isolate_cell` contract via `drive_with_backend`. Detectors + scoring + the runner are unit-tested
with planted fixtures + a scripted fake endpoint (vulnerable vs robust model), no GPU.

**Sourcing breadth (M5.1, delivered).** Two pure, injectable adapters extend the committed seed:
- `llb.prep.security_sources` adapts the public adversarial sets (AdvBench / HarmBench /
  JailbreakBench) into `SecurityCase` records, UA-adapted: a per-source normalizer extracts one
  `Behavior` (id + text + category), and `behavior_to_case` frames it as a UA request (`refuse`
  detector, `expect_refusal=True`). UA adaptation is INJECTABLE (`translate`); a `jailbreak_wrap`
  (UA DAN-style templates) turns a bare `unsafe_content` ask into a `jailbreak`-family case. Raw
  rows are passed IN (operator's local CSV/JSON under each set's license) -- nothing vendored or
  fetched at import. CLI: `llb adapt-security-set --source advbench --rows-file ... [--jailbreak]
  [--merge-seed]`.
- `llb.prep.security_planter` is the M3.5-style corpus-specific planter: over a real corpus it
  splices a marker-bearing instruction INTO a document (`rag_injection` leak cases, the attack in
  the retrieved context, a benign user turn) and embeds a unique CANARY in a confidential doc
  (`data_exfiltration` leak cases). Markers/canaries are DETERMINISTIC per (doc, index). CLI:
  `llb plant-security-cases --corpus-root ... [--merge-seed]`.

**Unsafe-content gated judge (M5.1, delivered).** `run_security` now takes the opt-in gated judge
(`--judge-model` / `--judge-rho` / `--judge-base-url`): over the harmful-ask (`refuse`-detector,
non-benign) cases it scores a `refusal_quality` scalar (answer-relevancy to a SAFE-REFUSAL intent --
a clean refusal scores high, compliance low), recorded ALONGSIDE the objective defense rate (per
harmful-ask case + mean + CI, a `JudgeStatus` in the manifest), never folded into the headline, and
only when trusted (`judge_rho >= 0.6`, the M3.8 gate); else demoted. No new safety classifier
(ShieldGemma / frontier moderation stay opt-in). `judge_scorer` is injectable, so it is proven with
a fake judge.

Both producers ship `verified=false`-equivalent: the MH.5 human sample-verify still gates any
headline use of the attack set.

### M5.2 Tooling / function-calling benchmark -- `llb.scoring.tooling` + `llb.bench.tooling`
The objective, CALL-ONLY function-calling board (tools are NOT executed -- that is M5.3).
`llb.scoring.tooling` has two pure layers: the PARSE layer (`parse_tool_call`) normalizes a backend
response into a `ToolCall` whether it is a NATIVE OpenAI `tool_calls` object, a pre-parsed dict, or
a text-only backend's JSON call in `content` (name/arguments aliases tolerated) -- so tool-capable
and text-only backends share ONE scorer and are never cross-ranked; the SCORE layer
(`score_tooling`) reports the four plan metrics -- tool-selection accuracy, argument-exactness
(`validate_arguments` is a lightweight required/type/no-unknown check, no `jsonschema` dep; plus
`arguments_match` exact value match, casefold/strip-insensitive for strings), no-hallucinated-tool
rate, and well-formed-call rate -- with the headline `call_accuracy` requiring the right tool AND
exact arguments. A no-tool case (the model should NOT call) scores correct only on no-call, so
over-calling is penalized.

`llb.bench.tooling.run_tooling` drives a candidate over a catalog + cases through an injectable
`complete` using a universal TEXT tool-calling protocol (`text_tool_prompt` embeds the catalog as
JSON; the model returns a JSON call), so every backend is exercised uniformly and a FAKE endpoint
proves the flow; it stamps a `ModelResult` under `TIER_TOOLING` (call accuracy as `objective_score`,
per-case correctness -> CI) and records all four rates + the tool-call protocol/capability in the
manifest. A committed BFCL-style UA bundle (`samples/tooling_cases_uk.json`: 5 tools, 12 cases incl.
no-tool controls + per-argument-tolerance cases) ships; `load_catalog_file` loads it. CLI:
`llb bench-tooling`. Parse, validation,
scoring, and the runner are unit-tested (native + text + malformed responses, perfect vs text-only
model), no GPU.

**Transports + breadth (M5.2, delivered).**
- **Selectable native FC + MCP from one catalog.** `run_tooling` takes an injectable `ToolCaller`
  `(instruction, catalog) -> ToolCall | None`; `text_tool_caller` is the default universal text
  protocol, `native_tool_caller` calls a tool-capable endpoint with native OpenAI `tools=`
  (`openai_tools` converts the catalog; the existing `parse_tool_call` normalizes the native
  response, so the SAME scorer runs). CLI: `llb bench-tooling --tool-protocol native|text` (native
  needs a running endpoint via `--base-url` / Ollama). `llb.bench.mcp_server` serves the SAME
  catalog over the official `mcp` Python SDK (`mcp_tool_specs` is the pure ToolDef->MCP mapping;
  `build_mcp_server` / `serve_stdio` lazily build a low-level server). CLI: `llb serve-tools-mcp`;
  the `mcp` SDK is an opt-in `[mcp]` extra.
- **BFCL UA adaptation.** `llb.prep.tooling_sources.from_bfcl` maps BFCL function-doc entries
  (+ optional possible-answers) into a `{tools, cases}` bundle; BFCL's several-acceptable-values-
  per-arg map onto the scorer's `oneof` tolerance. UA adaptation is injectable (`translate`); the
  schemas are kept verbatim. CLI: `llb adapt-bfcl --functions-file ... [--answers-file ...]`.
- **Per-argument tolerance.** `arguments_match` takes an optional `arg_match` spec relaxing a single
  argument to `contains` / `fuzzy` (difflib, stdlib) / `numeric` (abs tol) / `oneof` (default stays
  exact). The committed `samples/tooling_cases_uk.json` grew from 8 to 12 cases exercising each
  mode. The cases still need the MH.5 human sample-verify before headline use.

### M5.3 Agentic workflows benchmark -- `llb.bench.{tool_world,agentic}`
The agentic loop EXTENDS the M5.0 multi-hop controller pattern with tool calls + an in-sandbox
execution step. `llb.bench.tool_world` is the deterministic sandbox (no tau-bench / AgentBench): a
mock filesystem, a mock key-value DB, substring `search` over a small UA corpus, and a `calculator`
backed by a SAFE restricted-AST evaluator (`safe_eval` allows only numbers + arithmetic operators
+ parentheses -- no names/calls/imports). Each tool is a pure `(world, args) -> observation`
mutating only the in-memory `ToolWorld`, so a task's success is checkable from the final env-state.

`llb.bench.agentic.run_episode` is the harness loop: each step the model emits one tool call
(reusing the M5.2 `parse_tool_call`), the world EXECUTES it, the observation is fed back, and the
loop runs until the model calls `finish` (or answers in prose) or the step budget is exhausted.
`check_success` is an OBJECTIVE assertion over the final env-state / answer (`file_equals` /
`file_contains` / `db_equals` / `answer_contains`; ALL must hold; an empty assertion list never
passes). `run_agentic` aggregates completion-rate as the headline `objective_score` under
`TIER_AGENTIC` (per-task success -> the bootstrap CI), records trajectory length + tool-call count
as efficiency, and persists the manifest. A committed UA task set (`samples/agentic_tasks_uk.json`,
4 tasks) ships; `load_tasks_file` loads it. CLI: `llb bench-agentic`. The loop is the pure,
langgraph-free form of the controller->execute->controller cycle, unit-tested from a scripted fake
tool-calling endpoint (good agent solves tasks, failing agent does not; budget-exhaustion -> typed
`incomplete`), no GPU.

An OPT-IN gated TRAJECTORY-QUALITY signal (a check the env-state assertions cannot make) is wired
exactly like the M5.4 summarization faithfulness signal, through the same `run_gated_judge` seam:
each episode becomes one judge record whose retrieval context is the trajectory's TOOL OBSERVATIONS
and whose answer is the agent's final answer, so the judge's faithfulness (answer grounded in what
the tools returned) and answer-relevancy (answer addresses the goal) average into one
`trajectory_quality` scalar (`trajectory_quality` / `_trajectory_records`). It is recorded
ALONGSIDE completion-rate -- per-case + mean + CI, with a `JudgeStatus` in the manifest -- and ONLY
when the judge is configured AND trusted (`--judge-rho >= 0.6`, the M3.8 gate); otherwise the judge
is demoted and completion-rate ranks alone. The `judge_scorer` is injectable, so the wiring is
proven with a fake judge (no DeepEval / endpoint / GPU). CLI: `--judge-model` / `--judge-rho` /
`--judge-base-url`.

**Task-set breadth (M5.3, delivered).** `llb.bench.agentic_tasks` GROWS the committed seed with
real-UA-corpus SEARCH tasks whose success assertion is computed PURELY from the corpus -- no human
gold authoring to BUILD: `count` ("how many docs mention X?" -> document frequency of X) and
`locate` ("which doc mentions X?" -> the single doc id, only for terms in EXACTLY ONE doc, so the
answer is unambiguous). Query terms are DERIVED from the corpus by document frequency (UA-stopword
filtered) or supplied explicitly; each task drops straight into the `bench-agentic` loop +
`check_success` via the sandbox `search` tool. CLI: `llb prepare-agentic-search --corpus-root ...
[--merge-seed]`. Pure + unit-tested; tasks still need the MH.5 sample-verify.

The trajectory-quality judge stays gated by the existing M3.8 calibration; corpus/task-specific
judge tuning is part of future corpus onboarding and prompt-system benchmark runs, while objective
completion-rate remains the ranking floor. The `build_agentic_graph` LangGraph wrapper and the
LangGraph-vs-CrewAI harness comparison are scoped to Milestone 7 (the remaining frameworks --
LangChain / LlamaIndex / Haystack / AutoGen -- stay out of scope; see `plan.md`).

### M5.4 Remaining taxonomy -- summarization / structured output / chat-period / reliability
The remaining spec categories, each on the shared `bench.common` substrate:
- **summarization (`llb.bench.summarization`, `TIER_SUMMARIZATION`)** -- reference coverage via the
  PINNED-embedder cosine (NOT ROUGE): for each reference-summary sentence, the best cosine to any
  candidate sentence, averaged (`reference_coverage`; `similarity` injected, same basis as retrieval
  + the text-analysis matcher). Headline is mean coverage with a CI. An OPT-IN gated-judge
  FAITHFULNESS signal (summary vs source) is wired via `run_gated_judge` (the M3.8-calibrated
  faithfulness metric): when configured AND trusted (`--judge-rho >= 0.6`) it records per-case +
  mean + CI faithfulness ALONGSIDE coverage (never folded into the headline; the manifest carries a
  `JudgeStatus`), else the judge is demoted and coverage ranks alone. Committed cases
  `samples/summarization_cases_uk.json`; CLI `bench-summarization` (`--judge-model` / `--judge-rho`
  / `--judge-base-url`).
- **structured output (`llb.scoring.structured` + `llb.bench.structured`, `TIER_STRUCTURED`)** --
  objective JSON-schema conformance via PYDANTIC (`build_model` from a field schema; no new
  `jsonschema` dep) + field accuracy. Schemas may be NESTED: `_field_type` recurses so a
  `type: object` field with `fields` builds a nested model and a `type: array` field with `items`
  builds a typed `list[...]`, so conformance validates the whole shape. Field accuracy recurses too
  (`_compare`): it counts matching expected LEAF values across nested objects + array items, with
  strings casefold/strip-insensitive and numbers exact unless the field spec relaxes it. Per-field
  tolerance (M5.4): numeric `tolerance` (abs) / `rel_tolerance` (relative); string `string_match`
  `fuzzy` (difflib ratio >= `threshold`) / `contains`; and array `unordered: true` for order-
  insensitive (greedy best-assignment) set matching. A non-conformant output scores 0 field
  accuracy; the headline is field accuracy, conformance rate recorded alongside, both with CIs. The
  committed `samples/structured_cases_uk.json` grew from 3 flat to 6 cases incl. nested objects,
  arrays of objects, an unordered tag set, and fuzzy/relative tolerance. CLI `bench-structured`.
- **chat-period analysis** -- DELIVERED BY REUSE + the chat-specific producers in
  `llb.prep.chat_corpus`: a chat-log-shaped SYNTHETIC planter (`prepare_synthetic_chat_corpus`,
  reusing the text-analysis flow with a chat prompt) and REAL chat-corpus ingestion
  (`ingest_chat_corpus`: parse an export -- array / Telegram / JSONL -- render a chat-shaped doc,
  then DRAFT grounded labels with a LOCAL completion, NO egress per OQ-egress). Both write a bundle
  the `bench.text_analysis` runner scores; the runner's `synthetic` flag keeps real (`synthetic:
  false`) and synthetic results reported SEPARATELY. CLI: `llb ingest-chat-corpus` (real, local-only)
  and `llb prepare-synthetic-corpus --text-analysis --chat` (synthetic).
- **reliability (`llb.scoring.reliability`)** -- rolls the existing TYPED failure taxonomy
  (ok/empty/malformed/refusal/timeout/backend_error/retrieval_miss/...) from ANY run's per-case
  scores into a first-class reliability score + per-failure-type breakdown (`reliability_report`);
  `read_case_statuses` reads a run bundle's `scores.parquet`/`scores.jsonl`. CLI `bench-reliability
  --run-dir`. Pure + unit-tested.

All four score on a fixed seeded set with CIs and are pure/fake-endpoint unit-tested, no GPU.

**Guarded M5 composite headline (M7 support).** `llb.scoring.composite` builds a separate composite
row only when every required M5 tier for a model is present, has a reloadable per-case objective
series for bootstrap CIs, and is stamped with `data_verified=true`. The weights use the spec's
M5-category proportions and renormalize over the M5 subset: text-analysis 20, summarization 10,
structured 10, security 10, agentic 10, tooling 5. Category runners now persist a standardized
per-case `objective_score`, and the `bench-*` commands accept `--data-verified` plus
`--verification-ref` so verified runs can be made composite-eligible. `llb.goldset.verify` validates
those references mechanically before model calls and before a verified manifest can be persisted:
accepted forms are a fully decided `verify_sample.csv` within tolerance, a `sample_manifest.json`
that points to one, or an accepted-ledger bundle whose items are all `verified=true`. Invalid
references render a detailed operator diagnostic (`format_verification_status`) with path, kind,
reason, worksheet or ledger statistics, failing strata or unverified ids, and the exact
`verify-review` / `verify-accept` / rerun instruction needed to make the reference usable.
`llb.board.data` preserves category run metadata via `CategoryRunRecord` and re-checks verification
refs when building the composite; `load_m5_composite` returns either ranked composite rows or
concrete blockers. CLI: `llb bench-composite` (diagnostic escape hatches: `--allow-unverified`,
`--allow-missing-ci`). The Makefile target `make composite-headline` chains all six required
category benches with `--data-verified --verification-ref ...`, then runs `llb bench-composite` as
the clean preflight. Streamlit shows the composite section only when the verified, CI-capable suite
is complete. Operator flow: [`composite-headline.md`](../guides/composite-headline.md).

**Text-analysis judged sub-tasks + long_doc + contradiction + board (M5.4, delivered).**
`run_text_analysis` now takes the opt-in gated judge (`--judge-model` / `--judge-rho` /
`--judge-base-url`): the free-form sub-tasks (narrative / insight) become judge records (intent +
extracted answer + the doc as context), and a `long_doc` label is answered through the MAP-REDUCE
template (`eval.map_reduce.run_map_reduce_text` -- a `complete: str->str` driver over split -> map ->
reduce). The judge's faithfulness+relevancy collapse to one `judged_quality` per record, aggregated
per doc + overall (mean + CI, a `JudgeStatus`), recorded ALONGSIDE the objective recovery headline,
never folded in, and only when trusted; else demoted. The matcher now uses a `contradiction`'s
paired-span `attrs` (`spans` / `span_a` + `span_b`): credit requires the prediction to cover BOTH
contradicting sides (min of the two side credits). The Streamlit board (`llb.board`) loads the M5
category run bundles via `load_category_records` (grouped BY TIER, best run per model) and renders
each under its OWN Tier, never cross-ranked (the `aggregate` guard refuses a mixed-tier board).

Summarization's gated-judge faithfulness stays gated by the existing M3.8 calibration; corpus/task
judge tuning belongs with future corpus onboarding and prompt-system benchmark runs. All M5.4 cases
need the MH.5 human sample-verify before headline use.

### M5.6 second-frontier cross-check (verified-data gate) -- `llb.prep.cross_check`
The M4.4 data-prep residual the plan says "lands with M5's first scored category": the in-pipeline
verified-data gate. Every AI-DRAFTED item is re-confirmed by a SECOND, independent endpoint
(different from the drafter) layered on cheap deterministic pre-checks: GROUNDED (a labeled span
still resolves via `ground_span`) + NON-CIRCULAR (the answer is not leaked in the question), then
the second frontier's SUPPORTED (the cited span supports the answer) + ANSWERABLE (the question is
sensible/answerable). The pre-checks run FIRST so a clearly-broken item never spends a frontier
call. The verifier is injectable (`SecondFrontierVerify`); `second_frontier_verify` builds the
litellm-backed default; `cross_check_goldset` produces a `CrossCheckReport` (per-item verdicts +
pass count). Passing does NOT set `verified=true` -- only the human MH.5 sample-verify does; the
cross-check gates which drafted items are even eligible and is the report a human samples. CLI:
`llb cross-check-goldset --goldset --corpus --model` (`make cross-check-goldset BUNDLE= CROSS_CHECK_MODEL=`).
Pure + unit-tested (no key).

### MH.5 human sample-verify tooling -- `llb.goldset.verify` + `llb.goldset.verify_session`
The codeable half of the MH.5 gate: the operator tooling that turns a drafted + cross-checked bundle
into an accepted ledger, the verification-side twin of the `calibration-*` trio (mirrors how
`judge/calibration.py` pairs with `judge/rate.py`). `llb.goldset.verify` is the pure half --
stratification (`provenance|split|source_doc_id`; the `synthetic` flag is a BUNDLE-level fact read
from `provenance.json`, since the canonical `GoldItem` carries no per-item synthetic tag),
deterministic proportional sampling with a floor of one per stratum, the acceptance-sampling
arithmetic (per-stratum + overall reject rate vs tolerance, plus `undecided_with_failures`),
atomic CSV worksheet I/O, and `emit_accepted_ledger` (accepted items with `verified=true` + their
copied corpus, so the flip is an ADOPTION by replacement, never a boolean edit). It detects the gold
file (`goldset.jsonl` or the planter's `planted_labels.jsonl`) and surfaces any `*.cross_check.json`
verdict as read-only `cc_*` context. `llb.goldset.verify_session` is the interactive reviewer
(`run_session` + pure `parse_command` / `format_card` / `first_undecided_index`): a per-item card
showing the cited span inside its corpus window, the four checks (`g/a/r/p` pass, uppercase fail),
`y`/`x` accept/reject, resumable CSV-as-state; the cross-check is hidden by default (anti-anchoring,
`--show-crosscheck` reveals it). Subcommands `sample` / `review` / `accept`
(`python -m llb.goldset.verify`); make shortcuts `verify-sample` / `verify-review` / `verify-accept`
(`BUNDLE=`, `VERIFY_WS=`, `VERIFY_N=`, `VERIFY_TOLERANCE=`). Fully unit-tested with injected
inputs/output -- no model/endpoint/GPU (`tests/test_goldset_verify.py`). The committed
`ua_squad_postedited_v1` set is the verified side of the ledger (`DEFAULT_VERIFIED_GOLDSET`), already
250/250 `verified=true`, so MH.5 has no open work against it; the gate fires per NEW drafted bundle.
Operator workflow: [`docs/guides/goldset-from-scratch.md`](../guides/goldset-from-scratch.md) +
[`verification-tooling.md`](../guides/verification-tooling.md).

### M5.6 run-path hardening (M4 carry-overs) -- delivered

The host-dependent M4 run-path residuals are delivered and host-validated (see the Real-host
verification below):
- **M4.1 sliding-window KV + config override (`llb.backends.planner`).** KV is now SLIDING-WINDOW
  -aware: `attention_layer_split` / `kv_mib_at_context` / `max_context_for_kv` cap the KV of Gemma's
  sliding layers at `sliding_window` while the periodic full-attention layers grow with context, so
  a long context costs far less KV (piecewise, not linear). `arch_from_config` also reads
  `sliding_window` / `sliding_window_pattern` (and derives the period from a `layer_types` list), and
  `enrich_arch(spec, override=True)` lets a cached `config.json` OVERRIDE curated arch fields (the
  real served architecture wins), exposed as `list-models --trust-config`.
- **M4.2 multi-GPU read + arch-derived KV headroom (`llb.backends.hardware` + `llb.executor.contention`).**
  `select_target_gpu` reads ALL GPUs and targets the `CUDA_VISIBLE_DEVICES` device (or the most-free
  one) instead of hard-coding GPU 0; `default_gpu_reader` uses it. `model_kv_headroom_mb` derives the
  abort headroom from the served arch (the KV at a minimal serving context, sliding-window-aware)
  rather than a fixed 512 MB floor, so a heavy-KV model is judged un-launchable at the right
  threshold.
- **M4.3 flashinfer auto-pin + sampler-in-manifest + driver re-probe (`llb.backends.preflight` +
  `vllm` + `telemetry`).** The verdict records the GPU `driver`; `verdict_is_current` invalidates a
  cached verdict when the driver changes, so `llb preflight-vllm` re-probes WITHOUT a full
  `build-vllm` (`--force` to re-run regardless). `auto_pin_flashinfer` installs + re-probes candidate
  flashinfer versions (`LLB_FLASHINFER_CANDIDATES`) when the bundled one fails -- OPT-IN behind
  `preflight-vllm --auto-pin`, since it changes the environment. The vLLM launcher records the
  sampler actually used (`flashinfer` | `native`) + the flashinfer version into the manifest
  telemetry.
- **M4.5 further `/props` shapes + drivable partial offload (`llb.backends.llamacpp` + `cli.eval`).**
  `parse_served_context` checks the known `n_ctx` locations across llama.cpp versions (top level,
  `default_generation_settings[.params|.context]`, `generation_settings`, `model`, `props`) and never
  mistakes `n_ctx_train`; `run-eval --gpu-layers N` drives a partial GPU/CPU split without a YAML.

**M5.6 ontology data-prep residuals (delivered).** The three data-prep items feeding the M6
extraction reuse:
- **spaCy / Stanza adapter.** `llb.prep.ontology.spacy_adapter.SpacyExtractionAdapter` implements
  the `ExtractionAdapter` seam over spaCy `uk_core_news` NER -- entities only (spaCy's strength),
  with exact spans from the ent's own offsets (a prefix of the doc -> exact into full text) and
  repeated surfaces grouped into one entity (lightweight coreference). spaCy is lazy/opt-in; `nlp`
  is injectable so the mapping is unit-tested with a fake. CLI: `llb prepare-goldset-draft
  --extractor spacy [--spacy-model ...]`.
- **Long-doc chunking for extraction.** `LLMExtractionAdapter` now CHUNKS a doc longer than
  `EXTRACT_MAX_CHARS` into overlapping windows (reusing `split_document`), extracts per window, and
  MERGES (`merge_extractions` dedups entities by (name,type) merging mentions+aliases; events/claims/
  facts by evidence span + payload) -- so a long doc's later content is no longer truncated away.
  Grounding still runs against the FULL text, so offsets stay exact.
- **Richer ontology-type confidence.** `induce_ontology` now blends normalized count with normalized
  DOCUMENT frequency (`CONFIDENCE_COUNT_WEIGHT` / `CONFIDENCE_DOCFREQ_WEIGHT`), so a type spread
  across docs outranks one of equal count concentrated in one doc; types sort by confidence.
  `ontology_constraints` carries the high-confidence types into every drafting prompt
  (`draft_prompt` / `draft_items` `ontology_hint`) as an explicit constraint.

## Milestone 6 -- GraphRAG (knowledge-graph + narrative retrieval) (build COMPLETE)

A graph retrieval backend behind the RAG-store seam, selected with `--retrieval-backend graph`.
FAISS stays the default and is untouched. The BUILD is delivered + unit-tested without a GPU.
**MH.2 is signed off (2026-06-26):** the 13-type closed node vocabulary + relationship caps + the
GraphRAG scope are accepted, so the ontology/scope is trusted for HEADLINE use -- the signed schema
is [`docs/design/graph-ontology-schema.md`](../design/graph-ontology-schema.md) (`APPROVED`; node
vocab + caps + scope + a worked example over `samples/corpus/ip_regulation_uk.md`). Real-model graph
HEADLINE numbers ride only the standing MH.5 data gate now, exactly like the M5 categories; the
objective graph boards rank regardless.

**Store choice -- DuckDB (the abandoned Kuzu pick was dropped).** `duckdb` is already a dependency
(now also its own `[graph]` extra) so no new/abandoned dep is added. The graph is persisted as
node/edge JSONL (inspectable, diffable -- the same shape as the FAISS store's chunks) and loaded
into an in-memory DuckDB engine that carries the two graph queries: local k-hop via a recursive CTE
over the (undirected) edge table, and community grouping via `WHERE community_id IN (...)`. The
community ids are detected ONCE offline (so query time needs no graph-analytics dep) -- the
condition under which "DuckDB covers narratives". `duckdb` is lazy-imported, so the base install
still imports.

### Modules -- `llb.graph.*`
- `model.py` -- the RAM-resident `KnowledgeGraph` (`GraphNode` / `GraphEdge` / `GraphMention`).
  Every mention + edge evidence keeps `doc_id` + char offsets + exact text, and carries the
  induced ontology `type`/`confidence`, the containing `section_title`, and the `community_id`.
- `build.py` -- `build_graph(extractions, docs, ontology=None)` REUSES the M4.4 `DocExtraction`
  (no second extraction framework): entity mentions -> nodes, SRO facts -> directed edges; a fact
  endpoint that is not a known entity becomes a lightweight fact-only node (no grounded fact is
  dropped). Pure + deterministic; the ontology is induced from the extractions when not supplied,
  so each node carries its type confidence.
- `community.py` -- deterministic, seeded asynchronous label propagation (`detect_communities` /
  `assign_communities`): no graph-analytics dependency, the same corpus always partitions
  identically; an isolated node stays its own community.
- `retrieval.py` -- pure question linking + span-preserving serialization. Lexical entity linking
  keys on entity NAME + aliases (not mention text, which would link unrelated entities sharing a
  relation verb); `serialize_subgraph` renders member node mentions + intra-member edge evidence to
  ranked, span-deduplicated, offset-bearing `ChunkRecord`s. Linking is **morphology-aware**
  (`morph_key`): an exact token match scores full weight, and a shared leading-stem match (first
  `MIN_STEM_LEN` chars -- so the genitive "Франка" links "Франко") scores `STEM_MATCH_WEIGHT`
  below it, so inflected Ukrainian question forms still link while exact hits rank first. Pure +
  deterministic -- no lemmatizer, no embedder (constants in `graph/constants.py`).
- `store.py` -- `GraphStore`: `.build` / `.save` / `.load` + the `.retrieve(question, k)` seam (so
  the eval graph, scoring incl. the gated judge, isolation, and the board are UNCHANGED). The two
  strategies, recorded per run as `retrieval_strategy`:
  - **local_khop** -- entity-link the question to seed nodes, expand `graph_khop_depth` hops with
    the recursive CTE, serialize the subgraph (the multi-hop "connect these facts" path).
  - **global_community** -- map the question to its communities, serialize each community's member
    nodes/edges WITH their offsets (the narrative layer for corpus-level theme/trend questions).
- `summary.py` -- OPTIONAL `summarize_communities(graph, complete)`: an LLM one-paragraph summary
  per sizable community, recorded as a TAGGED DIAGNOSTIC (`community_summaries`, its own file) and
  NEVER returned by `.retrieve` -- the un-grounded abstraction never enters the span metric (the
  same recorded-but-not-ranked discipline as `--score-semantic`). Off by default; reachable from the
  operator CLI via `build-graph --summarize` (see below).
- `rag/compare.py` -- `compare_retrieval(stores, items, k)` scores several backends' `recall@k` /
  `MRR` on the SAME goldset by the one source-span metric, with `format_comparison` (ASCII table)
  and `load_compare_stores(cfg)` (loads `{faiss, graph/local_khop, graph/global_community}`, skipping
  any whose store is not built). Pure -- driven by the `.retrieve` seam, so it is unit-tested with
  fake stores (no FAISS / DuckDB / GPU). Backs the `compare-retrieval` command (below).
- `ingest.py` -- load the M4.4 extraction back: `load_bundle` reads a `prepare-goldset` draft
  bundle's `extraction.jsonl` + `corpus/` + `ontology.json`; `load_extractions` reads an explicit
  `extraction.jsonl`.

### Config + CLI + manifest
`RunConfig` gained `retrieval_backend` (`faiss` | `graph`), `retrieval_strategy` (`local_khop` |
`global_community`), and `graph_khop_depth`, plus `graph_dir()` (`$DATA_DIR/llb/graph/`). They ride
in the config fingerprint, so the manifest records backend + strategy and graph-vs-FAISS /
local-vs-global runs are comparable. `run-eval` and `validate-retrieval` take `--retrieval-backend`
/ `--retrieval-strategy`; the runner's `_load_store` picks `GraphStore` vs `RagStore` by backend.

    llb build-graph --bundle <prepare-goldset dir>     # reads extraction.jsonl + corpus/
    llb build-graph --extraction <e.jsonl> --corpus-root <dir>   # explicit M4.4 extraction
    llb build-graph --corpus-root <dir> --extract-model llama3.2:3b   # extract fresh (M4.4)
    llb build-graph --bundle <dir> --summarize --summarize-model llama3.2:3b  # + diagnostic summaries
    llb validate-retrieval --retrieval-backend graph --retrieval-strategy local_khop
    llb compare-retrieval --k 10 [--out report.json]   # faiss vs both graph strategies on one goldset
    llb run-eval --retrieval-backend graph --retrieval-strategy global_community ...

`build-graph --summarize` attaches the tagged-diagnostic community summaries via the M4.4 local
endpoint adapter (`--summarize-model`, defaulting to `--extract-model`); they persist to
`community_summaries.json` and are NEVER span-scored. `compare-retrieval` reuses the runner's
`_load_store` (varying backend/strategy via `with_overrides`) to rank `{faiss, graph/local_khop,
graph/global_community}` on `recall@k` / `MRR` over the same goldset, skipping any backend whose
store is not built (answer-quality comparison rides `run-eval --retrieval-backend ...`, which needs a
model). Or via make: `make build-graph BUNDLE=<dir>`. The `graph` extra (`duckdb`) is in the default
`make venv` EXTRAS.

### Tests + acceptance
`tests/test_graph.py` (25 tests): construction (offsets/section/confidence carry-through,
fact-endpoint linking, fact-only nodes), deterministic community detection, linking +
serialization, **morphology-aware linking** (`morph_key` collapses inflected forms, an inflected
question links the node, exact match outranks a stem match), both strategies returning
offset-bearing context that scores recall@k=1.0 on the existing span metric, k-hop depth bounding,
save/load round-trip, the tagged-diagnostic summaries (asserting they never appear as retrieved
chunks) + the `build-graph --summarize` no-model guard, ingest round-trip, and the full vertical
through `run_eval` (manifest records backend + strategy). `tests/test_compare_retrieval.py` (5
tests) covers the comparison core from fake stores (recall ranking, MRR tie-break, empty backends,
ASCII output). The DuckDB-engine tests `importorskip("duckdb")` so the lightweight CI install skips
them while the pure-graph logic runs everywhere; `make ci` is green (ruff + mypy strict + `-m "not
slow"`, 671 passed). Smoke-validated end to end: `build-graph` from a bundle ->
`validate-retrieval --retrieval-backend graph` recall@5=1.000, and `compare-retrieval` over a saved
GraphStore renders the ranked table (FAISS skipped when not built).

### M6 residuals (engineering)
The three optional/forward M6 residuals are delivered + CI-proven from fakes:
1. **Morphology-aware entity linking** -- `morph_key` stem matching in `graph/retrieval.py` (above);
   lifts graph recall on inflected Ukrainian questions without an embedder, behind the pure linking
   seam.
2. **`build-graph --summarize`** -- exposes `summarize_communities` to operators via the M4.4 local
   endpoint adapter (above); summaries stay tagged-diagnostic.
3. **`compare-retrieval`** -- the graph-vs-FAISS comparison tool (`rag/compare.py` + the CLI
   command, above), RUN on the real CUDA host over the committed corpus (result below).

### Reasoning-model extraction (build-graph fresh extraction)
Fresh `build-graph` extraction can drive a calibrated reasoning model. `EndpointConfig` gained
`think` (None | True | False) and `chat_once` an `extra_body` passthrough; when `think` is set the
local adapter routes to Ollama's **native `/api/chat`** (`_ollama_native_complete` /
`_native_chat_url`) because the OpenAI `/v1` layer ignores `think` and the model then burns the whole
token budget on hidden reasoning and returns empty JSON. `build-graph` exposes `--extract-base-url`
(point at a vLLM `/v1`), `--extract-max-tokens`, and `--extract-no-think`. Tests:
`tests/test_ontology_draft.py` (`_native_chat_url` mapping, think routes to native + records tokens +
provenance). See [`graph-vs-faiss-comparison.md`](../guides/graph-vs-faiss-comparison.md) for the
operator flow incl. the Ollama-vs-vLLM throughput finding on the 16 GB host.

### M6 verification -- real-host graph-vs-FAISS (2026-06-26, RTX 4060 Ti 16 GB)
Graph built from a fresh `gemma4:26b` extraction (Ollama, reasoning disabled, 3 concurrent workers,
~2.6 h) over the committed `ua_squad_postedited_v1` corpus: 250 docs -> **2396 nodes, 558 edges,
1839 communities** (21 docs yielded no extraction). `compare-retrieval --k 10` over the 250-item
gold set:

| backend | recall@10 | MRR |
| --- | --- | --- |
| **faiss** | **0.980** | **0.847** |
| graph/local_khop | 0.292 | 0.091 |
| graph/global_community | 0.284 | 0.229 |

**Flat vector retrieval strongly beats GraphRAG on this factoid corpus.** SQuAD paragraphs are
short and weakly connected, so the LLM extraction is sparse (0.23 edges/node) and the answer span is
usually not an extracted entity mention / edge evidence -- the graph has nothing to return for most
questions; `local_khop`'s low MRR (0.091 vs recall 0.292) shows the answer ranks behind many node
mentions when it is covered. GraphRAG is expected to pay off on multi-hop / narrative corpora, which
the committed factoid set is not -- this run quantifies that. Throughput finding: on the 16 GB card
Ollama's llama.cpp offload (~32 s/doc) out-throughputs vLLM `--cpu-offload-gb` (MoE 26B ~49 s/doc;
dense 31B w4a16 <5 tok/s) -- a model that fits fully in VRAM is the only way to get native speed.

## Real-host verification (2026-06-25, RTX 4060 Ti 16 GB)

All three backends + every M5 category benchmark were validated on the real CUDA host (driver
595.71.05), complementing the fake-endpoint CI boards:
- **Core RAG eval:** `validate-goldset` PASS (250 items); `build-index` 311 chunks (dim 768);
  `validate-retrieval` recall@10=0.980, MRR=0.847 (> the 0.8 gate); `run-eval` llama3.2:3b on the
  final split with telemetry -> ~101 tok/s, peak VRAM 3994 MB, reliability 1.0, manifest + MLflow
  mirror written. The uncalibrated judge demoted correctly (objective ranks alone).
- **Three-backend run-path (same `run-eval` interface, source-built backends):**
  - **Ollama** (prebuilt daemon): llama3.2:3b -- ~101 tok/s, peak VRAM 3994 MB (above).
  - **vLLM 0.23.0** (`backend=vllm`, `samples/run_config_vllm_uk.yaml`): `google/gemma-4-E4B-it-qat-w4a16-ct`
    (w4a16, the M2.4 16 GB fit) -- the pre-launch VRAM guard fired (`gpu-memory-utilization 0.80`
    fits, 14990 MB free), then served at 63.5 tok/s, peak VRAM 14496 MB, cold load 114.0s, served
    ctx 8192; reliability 1.0. Ollama released the GPU cleanly first.
  - **llama.cpp** (`backend=llamacpp`, source-built `llama-server` under `.data/llb/llamacpp/build/bin`):
    `Qwen2.5-0.5B-Instruct` Q4_K_M GGUF (loaded via `-m`, all layers on GPU) -- 401.6 tok/s, peak
    VRAM 2318 MB, load 2.0s, served ctx 32768; reliability 1.0.
  All three stamped a ranked row + telemetry + manifest + MLflow mirror under the SAME executor.
- **M5.6 M4 run-path hardening (host-validated):**
  - **M4.1:** `list-models --trust-config` over the cached `config.json` of `gemma-4-E4B-it-w4a16`
    resolved it to `42/42 gpu` at ctx 131072 -- the sliding-window KV fits the full window where the
    old full-attention estimate forced an offload.
  - **M4.2:** the vLLM pre-launch guard read the free VRAM and the arch-derived KV headroom
    (`gpu-memory-utilization 0.80 fits, 15309 MB free`) before launching.
  - **M4.3:** `preflight-vllm --force` found the cached verdict STALE and re-probed -- the venv's
    flashinfer 0.2.5 builds + runs here (sampler=flashinfer, driver 595.71.05 recorded). A vLLM
    `--telemetry` run then recorded `sampler` + `flashinfer_version=0.2.5` in the manifest (the value
    is the sampler ACTUALLY used: `native`, because an explicit env flag won over the verdict).
  - **M4.5:** a real PARTIAL-offload split of the oversized `deepseek-r1:32b` Q4 GGUF (19.9 GB > 16
    GB) at `--gpu-layers 20` served end to end -- peak VRAM 7710 MB (only ~20 of ~64 layers on GPU,
    the rest in CPU RAM) at 3.36 tok/s, with `n_gpu_layers=20` recorded in the manifest. (Ollama's
    `gemma3:27b` GGUF would not load -- a llama.cpp-build arch-key mismatch, not a bug here.)
- **M5 category benches (llama3.2:3b, committed UA case sets):** agentic completion 0.500;
  security ASR 0.600 / defense 0.400 with a per-family breakdown; tooling call-accuracy 0.750;
  summarization reference-coverage 0.941 -- each stamped under its OWN Tier with a manifest + CI.
- **M5.3 gated trajectory-quality judge end-to-end:** with a real judge endpoint (gemma4:e4b on
  Ollama as the wiring smoke -- the calibrated production judge is gemma3:27b -- `--judge-rho
  0.628`) the judge was TRUSTED and recorded `trajectory_quality=0.5875` (CI [0.375, 0.85])
  ALONGSIDE the objective completion (0.500, unchanged), with a `JudgeStatus`
  (`metrics=["trajectory_quality"]`) persisted in the manifest -- confirming the objective-first
  "diagnostic alongside, never folded into the headline/ranking" contract on a real model.
- `make ci` (Ruff format + lint, mypy strict, lightweight pytest `-m "not slow"`) is green with
  zero warnings; after the M5 residuals it selects 615 tests (10 slow deselected; `make test` runs
  all 625 locally).

(Still host-pending: only the M5.6 host-dependent M4 hardening items the basic run did not exercise
-- Gemma sliding-window KV + cached-`config.json` arch override (M4.1), multi-GPU read + arch-derived
KV abort headroom (M4.2), flashinfer auto-pin + sampler-in-manifest (M4.3, the sampler stays off by
default), and a real PARTIAL-offload llama.cpp split (M4.5; only the all-on-GPU path is now confirmed)
-- see `plan.md`.)

## Resolved questions and scope boundaries

The design spec ([`spec.md`](../design/spec.md)) is the source of truth for decisions; this
records the settled ones that affect WHAT is and is not built, so the forward plan
([`plan.md`](plan.md)) stays forward-only.

Resolved open questions:
- **OQ2 -- judge locality (M3.8):** a LOCAL Gemma-4 judge, tiered by GPU class (12/16/32 GB),
  chosen for no corpus egress + reproducibility; the Gemma-family self-preference bias is
  disclosed (see "Judge model (OQ2 decided) + bias disclosure" above). The only residual is the
  human calibration ratings (M3.8 in `plan.md`), not the scorer or the model choice.
- **OQ3 -- first candidate-model list (M2):** seeded in `samples/models_uk.yaml`; the vLLM repo
  ids are verified via `prep-models`.
- **OQ6 -- MAX_JOBS build helper (M2):** the canonical `max_jobs()` lives in
  `scripts/shared/common.sh` (AGENTS.md) and caps every CUDA source build.
- **OQ4 -- text-analysis + chat corpus facts (confirmed 2026-06-25):** text-analysis reference
  answers must be AUTHORED (AI-draft -> MH.5-verify, the current pipeline -- they do NOT pre-exist);
  the text-analysis corpus is BOTH real + synthetic, scored + reported SEPARATELY via the runner's
  `synthetic` flag (never merged); and a REAL chat-log corpus exists for chat-period (run via the
  real path, reported separately). So the M5.4 text-analysis + chat-period residuals must wire the
  REAL path, not only the synthetic planter.
- **OQ-egress -- cross-check egress for the real corpus (resolved 2026-06-25):** the second-frontier
  cross-check verifier is injectable (`SecondFrontierVerify`), so egress is per-corpus: the real
  CHAT-LOG corpus uses a LOCAL verifier only (no egress -- inject a local `SecondFrontierVerify`);
  the real TEXT-ANALYSIS corpus has frontier (litellm) cross-check egress APPROVED; synthetic
  bundles keep the litellm default. MH.5 remains the human gate for all of them.

Rejected pushbacks (ruled the other way; do NOT revisit -- see spec.md "Outside-voice
resolutions"): defer-Optuna-to-finalists, LangGraph-only-where-needed, drop-MLflow,
drop-thermal-gate, defer-vLLM.

Genuinely out of scope (v-next): a FULL six-framework comparison axis -- M5.3 ranks the model under
ONE fixed harness (spec Appendix D), and only a TWO-harness comparison (LangGraph vs CrewAI) is
taken forward, as Milestone 7 in `plan.md`; the remaining frameworks (LangChain / LlamaIndex /
Haystack / AutoGen) stay deferred. Also out of scope: loc-lm-bench as a public leaderboard (it
consumes lang-uk / INSAIT results as a prior, never duplicates them).

No longer deferred (now forward work in `plan.md`, not "out of scope"): the security / agentic /
MCP-tooling categories and the remaining taxonomy (Milestone 5), GraphRAG (Milestone 6, GO
decided), the LangGraph-vs-CrewAI harness comparison and RAG prompt-system generation/tuning
(Milestone 7), and the multi-backend / multi-vector-store / GPU-matrix / quality-per-watt
expansions (Milestone 7 / M7.4, built only with a committed consumer or more hardware).
