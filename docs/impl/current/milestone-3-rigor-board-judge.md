# Milestone 3 Current State

## Milestone 3 -- two-tier + scale + rigor (core + depth hardening delivered)

The M3 core components are built and unit-tested. The CLI grew `resolve-models`, `sweep`,
`tune`, `prepare-goldset`, `prepare-synthetic-corpus`, `screen-public`, and `board`. A
post-implementation audit confirmed the component boundaries and found that the full design
acceptance is not yet closed: screen-to-finalist orchestration, process-isolated Optuna trials
with backend-parameter search, judge integration/calibration, and human gold-set verification
remain forward work. The delivered behavior and those boundaries are stated below; residual
work stays in [`plan.md`](../plan.md).

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
[local judge guide](../../guides/judge-experiments.md).

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
