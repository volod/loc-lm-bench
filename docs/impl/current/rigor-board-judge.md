# Evaluation Rigor

Evaluation rigor covers host-aware model selection, isolated execution, tuning discipline, public
screening, board ranking, and local judge integration. The common theme is preventing convenience
shortcuts from leaking into model rankings.

## Backend Resolution

`src/llb/backends/resolver.py` chooses a runnable backend for a logical model. A model can declare
one source or a per-backend `sources:` map. The resolver combines:

- availability probes for Hugging Face repos, Ollama tags, and GGUF sources;
- host fit planning from GPU VRAM plus system RAM;
- backend priority: vLLM, then Ollama, then llama.cpp;
- artifact-specific metadata such as quantization and architecture fields.

```bash
llb resolve-models
llb resolve-models --offline
llb resolve-models --context 8192
```

The design favors actual serveability over nominal parameter size. vLLM must fit its serving
context in GPU memory; Ollama and llama.cpp may offload layers to CPU RAM.

## Isolated Sweeps

`src/llb/executor/isolation.py` defines the reusable process-per-cell primitive used by sweeps,
public screens, and isolated Optuna trials. The primitive:

- snapshots baseline GPU state;
- runs one backend-owning cell in its own process;
- checks VRAM reclaim after the cell;
- distinguishes new leaked PIDs from tolerated baseline shifts when PID attribution is available;
- applies a capped thermal cooldown.

```bash
llb sweep --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl --sweep-id run1
llb sweep --sweep-id run1
```

The `run1` value is a user-chosen sweep name; it writes under `$DATA_DIR/sweep/run1/`.
Cells publish stable markers under `$DATA_DIR/sweep/<id>/cells/`. Marker keys ignore the display
run name and keep reproducibility-relevant config fields.
After backend resolution, the sweep command also checks local serving prerequisites before creating
cells: vLLM cells require the `vllm` executable, and llama.cpp cells require a project-managed or
PATH-visible `llama-server`. Missing binaries are reported as skips instead of failed benchmark
cells, while real cell execution errors are still recorded and counted as failures.

## Two-Stage Tuning

`src/llb/optimize/tuner.py` uses Optuna for RAG parameter search. Stage 1 searches only on the
`tuning` split. Stage 2 evaluates the winning config on the `final` split, and only that final run
is a leaderboard candidate.

The search space includes chunking strategy, chunk size, overlap fraction, `top_k`, retrieval mode,
child chunk size, and vLLM serving knobs where relevant. In single-objective mode the embedder stays
pinned. Multi-objective mode (below) may sample it.

```bash
llb tune --model llama3.2:3b --backend ollama --trials 30 --study uk1 \
  --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl
```

Over-context configs are pruned before model calls. Measured OOMs can also prune trials. Persistent
SQLite studies live under `$DATA_DIR/optuna/`.

### Multi-objective RAG tuner

`llb tune --objectives quality,latency[,cost]` switches stage 1 to Optuna multi-objective search
(`NSGAIISampler` plus median-style early pruning on progressive case subsets) across
`src/llb/optimize/multi_objective_trial.py`, `multi_objective_runtime.py`, and
`multi_objective_study.py`. Objectives:

| Goal | Direction | Source |
| --- | --- | --- |
| `quality` | maximize | tuning-split objective score |
| `latency` | minimize | mean generate latency (falls back to trial wall-clock) |
| `cost` | minimize | frontier ledger `cost_usd` (requires `scorer_policy=frontier`) |

Instead of one winner, the study emits a Pareto front plus named picks: `best_quality`,
`best_quality_per_second`, and (when cost is active) `cheapest_within_floor` (default floor =
0.9 * best quality on the front, override with `--accuracy-floor`). Stage 2 scores each named pick
on the final split. Reports land under `$DATA_DIR/tune/<run>/` as `pareto.json` + `pareto.md`.

Additional search knobs in this mode:

- **Embedder** -- categorical over the bake-off shortlist
  (`DEFAULT_LOCAL_CANDIDATES` in `src/llb/rag/embedding_bakeoff.py`); override with
  `--embedders a,b` or pass `--embedders ""` to keep the pinned model. The per-study
  `StoreRegistry` (`src/llb/optimize/store_registry.py`) rebuilds when the embedder or
  chunking fingerprint changes, and never reuses a store across different embedders.
- **Store prewarm / disk cache** -- when `--embedders` is active, the shortlist is pre-built
  for the base config's chunking fingerprint before the Optuna loop; the first sight of any
  new chunking shape also fan-outs all shortlist embedders once. Bare stores persist under
  `$DATA_DIR/optuna/<study>/stores/<fingerprint-slug>/` so a resumed study reloads instead of
  re-embedding. Fusion and rerank knobs still apply from the current trial config on every
  get. CI: `tests/llb/optimize/test_store_registry.py` (fake builder counts embeds; second
  reuse of a fingerprint issues zero new embeds).
- **Context budget** -- samples a token budget from `{2048, 4096, 8192, 16384}` that couples
  `top_k` / `chunk_size` / `max_model_len` (`RunConfig.context_budget`); disable with
  `--no-context-budget`.

```bash
llb tune --model llama3.2:3b --backend ollama --objectives quality,latency \
  --trials 40 --study mo1 --limit 12 \
  --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl \
  --corpus samples/goldsets/ua_squad_postedited_v1/corpus
```

CI covers vocabulary and trial policy in `tests/llb/optimize/test_multi_objective.py`, study and
Pareto behavior in `test_multi_objective_studies.py`, and store prewarm/fingerprint reuse in
`test_store_registry.py`.

Host evidence (2026-07-18, RTX 4060 Ti 16 GiB, Ollama `llama3.2:3b`, UA-SQuAD postedited fixture,
`--trials 40 --limit 20 --seed 21 --objectives quality,latency`):

- Study: `$DATA_DIR/optuna/mo-ua-evidence-20260718c.db`
- Report: `$DATA_DIR/tune/mo-ua-evidence-20260718c/pareto.{json,md}`
- 11 complete / 29 median-pruned of 40; Pareto front size 4 (non-dominated)
- Picks: `best_quality` trial 30 (tuning quality 0.386, generate latency 0.378 s) -> final
  quality 0.434; `best_quality_per_second` trial 8 (0.386 / 0.320 s) -> final quality 0.477
- Context-budget knob active (sampled 8192 / 16384 on the picks); embedder rebuild invariant
  and store-prewarm zero-reuse-embed gate covered by unit tests with fake builders / registries

## Joint model + config search

`llb joint-search` (`make joint-search`) folds model selection into the optimization loop with a
successive-halving schedule so the recommendation covers model + RAG config + serving knobs
together instead of tuning RAG for one pre-chosen model.

Schedule (`src/llb/optimize/joint_search/`):

1. **Host-fit filter** -- `resolve_all` over `--candidates` (default
   `samples/configs/models_uk.yaml`); unresolvable models are skipped and recorded in the run
   manifest.
2. **Cheap screen** -- each runnable candidate is scored on the **tuning** split only with a small
   case cap (`--screen-limit`, growing by `--eta` each round). Screen cells reuse
   `isolate_cell` for VRAM-owning backends. Each completed cell writes
   `screen/<slug>-r<round>.json` so a resume skips re-evaluation.
3. **Successive halving** -- each round keeps `max(min_finalists, n // eta)` survivors by screen
   quality; eliminations are written to `ledger.json` with `split=tuning` (final-split scores
   never enter the ledger). The ledger is rewritten after every round.
4. **Per-finalist multi-objective tune** -- survivors run Optuna `tune_multi` then final-split
   pick scoring in isolated cells under `$DATA_DIR/joint-search/<run>/finalists/<model>/`. Study
   ids are `joint-<run_id>-<slug>` under `$DATA_DIR/optuna/`; only remaining trials run when the
   SQLite study already has rows. Each finished final-split pick writes
   `finalists/<slug>/picks/<goal>.json` so a kill mid-pick-scoring skips completed picks on
   resume. An explicit `--limit` also bounds final pick scoring, and multiple goals selecting the
   same config share one evaluation while retaining separate resume markers. A finished finalist
   (all picks scored) writes `finalists/<slug>/result.json` (study id + final-split picks) so a
   resume reloads instead of re-tuning.
5. **Final scoreboard** -- `scoreboard.json` + `scoreboard.md` list only **final**-split pick
   scores; the writer refuses any non-final split (tuning/final leak fence). The scoreboard is
   rebuilt after each finalist so a partial run still shows whatever picks exist.

**Resume:** re-run with the same `--run-id` / `JOINT_SEARCH_RUN_ID=<id>`. Completed screen markers,
per-pick scoring markers, and finalist `result.json` files are skipped; Optuna studies only
enqueue `max(0, n_trials - len(study.trials))` new trials.

```bash
make joint-search JOINT_SEARCH_TRIALS=20 JOINT_SEARCH_SCREEN_LIMIT=8
# resume after kill:
make joint-search JOINT_SEARCH_RUN_ID=<id> JOINT_SEARCH_TRIALS=20
# or:
llb joint-search --candidates samples/configs/models_uk.yaml --trials 20 \
  --run-id <id> \
  --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl \
  --corpus samples/goldsets/ua_squad_postedited_v1/corpus
```

Artifacts under `$DATA_DIR/joint-search/<run>/`: `manifest.json`, `ledger.json`,
`screen/<slug>-r<round>.json`,
`finalists/<model>/{pareto.{json,md},picks/<goal>.json,result.json}`,
`scoreboard.{json,md}`.

CI drives the schedule with injectable screen/tune hooks in `test_joint_search.py`. Halving fences
and resume behaviors are separated into adjacent `test_joint_search_halving.py`,
`test_joint_search_resume.py`, `test_joint_search_optuna_resume.py`, and
`test_joint_search_pick_resume.py` modules.

Host evidence (2026-07-18, RTX 4060 Ti 16 GiB, UA-SQuAD postedited fixture, three
`models_uk.yaml` candidates -- MamayLM-12B GGUF, Lapa-12B GGUF, Mistral-Small-3.1-24B --
`--trials 10 --screen-limit 4 --limit 8 --seed 21 --objectives quality,latency`):

- Run: `$DATA_DIR/joint-search/joint-ua-evidence-20260718/`
- Ledger (`ledger.json`): `split=tuning`; round 0 eliminated `lapa-v0.1.2-instruct`
  (screen quality 0.303); kept `mamaylm-v2-12b` (0.381) and `mistral-small-3.1-24b` (0.366)
- Scoreboard (`scoreboard.json`): `split=final` only; MamayLM `best_quality` 0.488 /
  `best_quality_per_second` 0.563; Mistral both picks 0.391
- Recommended: `mamaylm-v2-12b` + `best_quality_per_second` (recursive, chunk 256, top_k 3,
  context_budget 2048)
- Final-split manifests under `$DATA_DIR/run-eval/` for each pick all record `split=final`;
  no tuning rows on the scoreboard

Roster-refresh acceptance evidence (2026-07-21, RTX 4060 Ti 16 GiB, focused Gemma 4 26B-A4B and
Qwen3.6 27B slice from `models_uk.yaml`, UA-SQuAD postedited fixture,
`--trials 2 --screen-limit 2 --min-finalists 2 --limit 4 --seed 13`):

- Run: `$DATA_DIR/joint-search/ua-model-roster-refresh-native-bounded-20260721/`
- Both entries resolved to Ollama CPU/GPU offload and reached the final board; tuning-screen
  quality was 0.368 for Gemma and 0.333 for Qwen, with no elimination because two finalists were
  required.
- The single non-pruned trial for each model selected semantic flat retrieval, chunk size 704,
  overlap 171, top_k 3, and context budget 8192. Identical quality and quality-per-second picks
  shared one final evaluation per model and still produced both goal markers.
- Final quality was 0.174 for Gemma and 0.243 for Qwen; both had reliability 1.0, recall@3 1.0,
  and MRR 1.0. Gemma generated at 15.13 tokens/s versus Qwen at 2.61 tokens/s.
- Recommended for this bounded acceptance sample: `qwen3.6-27b` + `best_quality`. The four-case
  final cap is enough to validate the roster/runtime path, not enough for a research-grade model
  adoption decision; a larger confirmation remains optional forward work.

## Public Screen

`src/llb/screen/public.py` adapts `lm-eval-harness-uk` to a running local endpoint. It keeps
logprob and generation tracks separate because their metrics are not comparable.

```bash
llb screen-public --model <model> --backend vllm --isolated
llb pipeline --top-n 2 --trials 20
```

`screen-public` writes coverage-aware reports under `$DATA_DIR/screen/`. `pipeline` reads those
reports, selects per-track finalists, tunes them on the private RAG split discipline, and prints
the final board.

## Board Ranking

`src/llb/scoring/aggregate.py` generalizes ranking beyond one row.

Ranking guards:

- average rank across shared quality signals rather than a silent arbitrary blend;
- bootstrap confidence intervals from per-case series;
- unresolved marks when adjacent CIs overlap;
- Pareto marks over quality, throughput, and VRAM;
- hard rejection of mixed tiers or incompatible judge cohorts;
- duplicate model-config rejection before ranking.

`src/llb/board/` loads run bundles and renders Streamlit views. Loading is split by concern:
`runs`, `categories`, `harnesses`, `prompt_systems`, and `io`. The board uses final private runs
for RAG leaderboards and separate sections for public screens, category tiers, harness comparisons,
and prompt-system comparisons.

```bash
make board
```

## Recommendation Summary

`llb recommend` (`make recommend`) distills a sweep into the few operator-facing picks a leaderboard
implies but does not state, plus a comparison chart. It reuses the board loaders
(`load_run_records` -> `best_per_model`) and the `aggregate` ranking (`rank_board`, `pareto_front`),
adding the host-efficiency + retrieval fields the `ModelResult` omits (`quality_per_watt`,
`mean_power_w`, `recall@k`, `MRR`); recommendation construction lives in
`src/llb/board/recommend/build.py` and the
matplotlib chart in `src/llb/board/charts.py` (guarded `[viz]` extra).

Picks:

- Recommended for this host: the highest-accuracy model that is feasible, Pareto-optimal, AND fits
  the GPU tier's VRAM budget with headroom (`peak_vram_mb <= 0.92 * total`). This is the
  HOST-ADAPTIVE pick -- on the same bundles a 16 GiB host recommends Lapa while a (simulated)
  24 GiB host recommends the larger MamayLM-27B, because the budget admits it. The pick also names
  its `best RAG top_k`, which is meaningful once the sweep gridded `top_k` (see the RAG-config grid
  in [`rag-core.md`](rag-core.md#sweep-rag-config-grid)): best-per-model dedup represents each model
  by its highest-scoring retrieval depth, so the recommendation answers `(model, top_k)`, not just
  model.
- Best RAG accuracy: rank-1 by objective/blended quality.
- Best efficiency: max `quality_per_watt` (the platform-matrix benchmark-efficiency axis).
- Fastest: max tokens/sec.

The host pick is quality optimization SUBJECT TO host constraints that relax in order
(performance -> VRAM -> Pareto). `--min-tokens-per-s` (`RECOMMEND_MIN_TOK_S=`, 0 = off) adds a
good-enough-performance floor on top of the VRAM fit: the pick must clear the floor, and when it
does the summary names any higher-accuracy models that were traded away for speed, so the operator
sees exactly what the floor cost. All report prose is sourced from `board.recommend.*` prompt
templates (`prompts/templates/board/recommend/`) rather than inline literals, so the wording is
reviewable in files; `format_summary_md` only computes values and assembles the line list.

Only the dominant `(split, n_cases)` cohort is ranked. Comparing models is apples-to-apples only
within a shared split AND case count, so `select_cohort` keeps the cohort with the most models
(ties -> the larger `n_cases`, the more robust comparison) and lists the rest under an
`Excluded (off-cohort, not ranked): MODEL n=N` note rather than ranking a 20-case platform-matrix
row beside an 82-case sweep. `--min-cases` still pre-filters smoke bundles BEFORE the best-per-model
dedup so a 3-case manual run never shadows a full sweep; the cohort split is the backstop when
several real case counts coexist (the quickstart's `--min-cases 1` default would otherwise rank a
2-case bundle beside the 82-case cohort). `--gpu-gb` simulates another CUDA tier's VRAM budget for
the fit check. Outputs land at `$DATA_DIR/recommend/{summary.md,comparison.png}`, and
`quickstart-goldset` runs it as the final eval step.

Below the per-model picks the summary appends a `## RAG configuration detail (model x config)`
section: `load_config_cells` keeps every final-split `(model, RAG-config)` cell (best re-run per
cell, NOT collapsed to best-per-model), so a model swept at several retrieval depths shows all of
them. The table groups by model, marks each model's best config, and — when nothing was gridded —
appends a note pointing at `SWEEP_RAG_GRID`. This is the detailed proof that the winning
configuration is demonstrated per model, not assumed, complementing the best-per-model headline.

```bash
make recommend RECOMMEND_MIN_CASES=50          # detected host tier
make recommend RECOMMEND_GPU_GB=24             # would a 24 GiB box pick a bigger model?
```

Validated on the 16 GiB RTX 4060 Ti committed-goldset sweep (5 families, 82 final cases): MamayLM-27B
led accuracy (objective 0.546), Lapa was the recommended host pick (0.505, fits with headroom),
Qwen3.6 led efficiency (0.216 quality/W), and the Ukrainian-specialized models out-scored the
multilingual Mistral Small 3.1 (0.399) and Qwen baselines on Ukrainian RAG.

## Miss Analysis (analyze-misses)

`llb analyze-misses --run-dir <run>` (`make analyze-misses RUN_DIR=<run>`) explains a finalized
run's wrong answers. Classification, clustering, and recommendations live in
`src/llb/board/miss_analysis/`; probe orchestration lives in `src/llb/board/miss_probe.py`;
tests in `tests/llb/board/test_miss_analysis.py` (a synthetic scored bundle with one case per miss class
proves zero cross-class leakage and that every recommendation line names numeric evidence).

Every miss lands in exactly ONE class, decided in precedence order: `refusal` (typed status),
`format_artifact` (empty / malformed / timeout / backend_error -- output or transport, not
knowledge), `retrieval_miss` (typed status, or the gold span never overlaps a retrieved span),
`judge_disagreement` (objective below the miss threshold while the trusted per-case judge rated
>= 0.7 -- a scoring conflict for a human to look at), else `generation_miss` (evidence present,
answer wrong). A scoreable case is a miss when `objective_score < 0.5`
(`--miss-threshold` / `MISS_THRESHOLD=` overrides). Span overlap reads the additive per-case
`retrieval.jsonl` records persist beside `scores.jsonl`
(`batch_retrieval_records` in `src/llb/executor/cases.py`; doc id + char offsets + rank +
score + bounded 160-char text preview + the gold spans, plus the other places of a
duplicate-collapsed chunk -- see
[the persisted retrieval record](rag-core.md#the-persisted-retrieval-record)). When detailed
retrieval evidence is absent, classification uses the scored `retrieval_hit` flag and logs a
warning. Each miss row also lists `retrieved_docs`, the distinct documents its scored context
carried, so a retrieval miss can be read against the document the operator expected.

Misses are clustered by document (`source_doc_id`), topic, and question type, with per-key miss
rates computed over ALL scored cases of that key. Labels come from the goldset's
`item_provenance.jsonl` sidecar when the draft pipeline emitted one (`question_type` / `topic`);
otherwise a deterministic UA/EN interrogative heuristic types the question and the longest
content token stands in for the topic -- lemmatized through the hybrid-retrieval lemma normalizer
(`llb.rag.lexical.ukrainian_lemma`), so Ukrainian case forms of one topic land in a single cluster
instead of splitting across inflections. Recommendations are ranked by the miss count they
address and rendered from `board.miss.*` prompt templates: raise/lower `top_k`, change
chunking, add prompt-system dictionary terms for a dominant generation-miss cluster, try the
named alternative model (cited with its measured objective from comparable sibling bundles --
same split and case count), review refusals / artifacts / judge disagreements.

Probe mode (`--probe-top-k 3,8` / `PROBE_TOP_K=3,8`) re-runs ONLY the miss subset at each
alternative retrieval depth through the normal durable `run_eval` (same recorded config; only
`top_k` and `run_name` change, judge and telemetry off), so the retrieval hypothesis is
confirmed or rejected with measured recovery numbers, and a shallower depth that beats the miss
subset's baseline objective by >= 0.05 earns a "lower top_k" line. Probe bundles are ordinary
run bundles named `miss-probe-<run_id>-k<k>`: a finalized probe is reused (never re-run), an
interrupted probe's staging is found by its pinned config + goldset digests and resumed via the
durable-eval-runner journal, and only then does a fresh probe start. Off-cohort probe bundles
never pollute the board headline (tiny `n_cases` -> cohort exclusion).

Artifacts land at `$DATA_DIR/miss-analysis/<timestamp>/{report.md,misses.jsonl,analysis.json}`;
`llb recommend` appends a `## Miss analysis` section (intro + top 5 ranked lines) from the
latest `analysis.json` when one exists (`format_miss_section_md` in
`src/llb/board/recommend/sections.py`). Run bundles are never mutated. Automatic re-tuning stays
out of scope -- the Optuna tuner owns search.

## Context-Position Probe (probe-context-position)

`llb probe-context-position --model <m> --backend <b> --k <k>`
(`make probe-context-position MODEL=<m> BACKEND=<b> PROBE_K=5`) measures a model's
lost-in-the-middle sensitivity and names its `context_order` recommendation with evidence
(rerank-context-order). Core in `src/llb/eval/position_probe.py`; CLI in
`src/llb/cli/eval/analysis.py`;
tests in `tests/llb/eval/test_position_probe.py` (a fake store + a fake chat that answers correctly only
when the gold chunk leads the prompt prove case construction, exact gold placement, per-position
scoring, the recommendation rule, and the artifacts -- no backend, no GPU).

Per verified gold item, ONE retrieval at `--candidate-depth` (default 50) supplies both the gold
chunk (the first candidate overlapping a gold span) and the k-1 best-ranked non-gold distractors
-- real retrieved distractors, never synthetic filler. Items whose gold chunk is not retrievable
or that lack k-1 distractors are counted per skip reason (`gold_not_retrieved` /
`too_few_distractors`), never invented. The gold chunk is then laid at the head, middle, and
tail of the fixed-k context (`k >= 3` enforced -- below that the slots collapse) and the same
question is asked three times through the standard RAG chat prompt. Each answer is
status-classified and scored by the objective correctness scorer against the reference answer.

The report gives per-position n / mean objective / bootstrap 95% CI and recommends `rank`
(best-first) when the head mean is at least the tail mean, else `reverse_rank` (best-last);
overlapping head/tail CIs are flagged as unresolved at that n (the recommendation still names
the higher mean, honestly qualified). Artifacts land at
`$DATA_DIR/context-position/<timestamp>/{report.md,cases.jsonl}`; probe cases never enter run
bundles, the board, or correctness aggregates.

Durable evidence (2026-07-10, rerank-order-full-cohort on the CUDA host, outside quick CI):
full-final-split probes (`ua_squad_postedited_v1`, 82 final items, k=5, no LIMIT cap) per
roster model on Ollama:

- `llama3.2:3b`: head 0.448 [0.360, 0.526], middle 0.419 [0.331, 0.498],
  tail 0.433 [0.351, 0.511] -- the mild best-first slope survives at n=82 but the head/tail CIs
  still overlap. Explicit verdict: NOT measurably position-sensitive (head-tail delta 0.015 well
  inside the CIs); the default `rank` ordering stands and no more n will plausibly resolve a
  gap this small into a knob worth setting.
- `gemma4:e4b`: head 0.414 [0.337, 0.493], middle 0.362 [0.291, 0.434],
  tail 0.407 [0.333, 0.482] -- the classic lost-in-the-middle U-shape (the middle slot pays
  ~-0.05 against both edges) but every pairwise CI overlaps at n=82. Explicit verdict: NOT
  measurably head/tail position-sensitive (`rank` stands); the middle dip suggests keeping
  `top_k` small enough that gold evidence never sits deep mid-context, which the shipped
  per-model `top_k` sweep already optimizes.
- `hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M`: head 0.517 [0.427, 0.592],
  middle 0.507 [0.422, 0.584], tail 0.505 [0.423, 0.581] -- the flattest profile in the cohort
  (head-tail delta 0.012, all CIs overlap). Explicit verdict: NOT position-sensitive; `rank`
  stands, and the Ukrainian-specialized 12B is the most ordering-robust model probed.
- `hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M`: head 0.528 [0.442, 0.606],
  middle 0.481 [0.401, 0.566], tail 0.485 [0.404, 0.566] -- the largest head advantage in the
  cohort (+0.043 over tail) with a mild middle dip, but the CIs still overlap at n=82. Explicit
  verdict: NOT measurably position-sensitive; `rank` (already best-first) captures whatever
  head preference exists, so no knob change is warranted.

Cohort verdict: no probed roster model resolves head/tail position sensitivity at the full
final-split n=82 -- the honest cohort-wide recommendation is that the default `rank` ordering
stands everywhere and `context_order` is not a knob worth per-model tuning on this goldset.
The rerank half of the cohort is recorded in [RAG core](rag-core.md) Reranking And Context
Order.

### Roster-wide probe cohort (2026-07-24)

A second full-cohort pass on the same host, index, and item set (`ua_squad_postedited_v1`, 82
final items, k=5, Ollama, no LIMIT cap) extends the probe to the Gemma 4, MamayLM v2.0, and
Qwen3.6 rosters. All seven models probed 82/82 items with 0 skips and reliability 1.0.
Artifacts: `$DATA_DIR/context-position/20260724T0{63341,63726,64807,65031,70950,71850,73314}Z/`
(lapa, gemma4:e4b, gemma4:26b, gemma4:e2b, MamayLM-12B, Qwen3.6-35B-A3B, MamayLM-27B).

| model | head | middle | tail | overall | head-tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| `batiai/qwen3.6-35b:iq3` | 0.558 | 0.602 | 0.552 | 0.571 [0.52, 0.62] | +0.005 |
| MamayLM-Gemma-3-27B-IT v2.0 GGUF Q4_K_M | 0.583 | 0.549 | 0.561 | 0.565 [0.52, 0.61] | +0.022 |
| MamayLM-Gemma-3-12B-IT v2.0 GGUF Q4_K_M | 0.517 | 0.507 | 0.505 | 0.510 [0.46, 0.56] | +0.012 |
| Lapa v0.1.2-instruct GGUF Q4_K_M | 0.528 | 0.481 | 0.485 | 0.498 [0.45, 0.55] | +0.044 |
| `gemma4:e2b` | 0.469 | 0.425 | 0.441 | 0.445 [0.40, 0.49] | +0.028 |
| `gemma4:e4b` | 0.390 | 0.369 | 0.372 | 0.377 [0.34, 0.42] | +0.018 |
| `gemma4:26b` | 0.315 | 0.290 | 0.268 | 0.291 [0.27, 0.32] | +0.047 |

The 2026-07-10 cohort verdict holds unchanged: every head/tail CI still overlaps, so `rank`
stands for all seven and `context_order` remains a knob not worth per-model tuning.

Reproducibility, measured: Lapa and MamayLM-12B reproduced their 2026-07-10 numbers to three
decimals on every position. `gemma4:e4b` did NOT (head 0.414 -> 0.390, tail 0.407 -> 0.372) on
the same index and item set, so ~0.035 is that model's run-to-run noise floor and any e4b delta
below it is unresolvable. Gemma-3-derived GGUFs are bit-stable here; the Gemma 3n/E4B kernel
path is not.

**Position score is not needle-reading skill.** `objective_score` is token F1, which is
precision-sensitive, so it conflates whether the model FOUND the needle with how tersely it
stated it. Scoring the same rows with `contains` (all reference tokens present) separates them
and reverses the ranking:

| model | overall F1 | needle located | F1 given located | answer length vs reference |
| --- | ---: | ---: | ---: | ---: |
| `batiai/qwen3.6-35b:iq3` | 0.571 | 162/246 (0.659) | 0.787 | 1.9x |
| MamayLM-27B v2.0 | 0.565 | 155/246 (0.630) | 0.790 | 1.6x |
| MamayLM-12B v2.0 | 0.510 | 166/246 (0.675) | 0.685 | 2.3x |
| Lapa v0.1.2 | 0.498 | 154/246 (0.626) | 0.688 | 2.7x |
| `gemma4:e2b` | 0.445 | 184/246 (0.748) | 0.542 | 3.9x |
| `gemma4:e4b` | 0.377 | 193/246 (0.785) | 0.444 | 4.6x |
| `gemma4:26b` | 0.291 | 184/246 (0.748) | 0.332 | 5.2x |

The Gemma 4 collection ranks LAST on F1 and FIRST on found-rate: `gemma4:e4b` locates the
needle on 0.785 of probes against Lapa's 0.626 (paired, +0.159 [+0.061, +0.268]), yet loses on
F1 by 0.121 [0.052, 0.194] because it answers at 4.6x reference length. Within Gemma 4,
verbosity rises monotonically with size while F1-given-located falls in lockstep. Among the
Ukrainian-tuned and Qwen models the found-rate spread (0.626-0.675) is entirely inside the CIs,
so their F1 ordering is answer style, not comprehension. Read both columns before calling one
model a better context reader; reference answers on this fixture average 18 characters, which
is what makes token F1 this sensitive to padding.

Caveat on the Qwen row: the 16 GiB roster fallback is an IQ3 (3.5 bpw) artifact while both
MamayLM tags are Q4_K_M (~4.5 bpw), so Qwen ties the 27B from a weaker quantization. The
comparison is conservative in Qwen's favor, not like-for-like.

## Insufficient-Context Abstention Probe (run-eval --insufficient-context-probes)

`llb run-eval --insufficient-context-probes <n>` re-runs a seeded sample of gold items with their
gold evidence excluded from retrieval and scores abstention accuracy -- the share on which the
model correctly declines instead of fabricating an answer. Like the position probe, these probe
cases are scored on their OWN axis (`probes.jsonl` + `insufficient_context_report.md` in the run
bundle) and NEVER enter the correctness aggregates. It is part of the answer-side
groundedness/citation metrics; the mechanism, the deterministic groundedness + citation-validity
scorers (`--score-groundedness` / `--cited-answers`), and durable per-model evidence live in
[RAG core](rag-core.md) groundedness and citation metrics.

## Ukrainian Query Robustness Benchmark

`llb bench-query-robustness` / `make bench-query-robustness MODEL=<m> BACKEND=<b> GOLDSET=<gs>`
measures end-to-end sensitivity to four deterministic noisy-query classes, ONE MECHANISM EACH:
Latin-typed `transliteration`, `apostrophe_variant` (every apostrophe re-typed as a different
variant), `mixed_script` (Cyrillic/Latin homoglyph substitution at the configured character rate),
and keyboard-adjacent Cyrillic `keyboard_typos`. One class per mechanism is what makes a recovery
attributable: the earlier combined `apostrophe_mixed_script` class applied both at once, so its
apostrophe half was invisible whenever the homoglyph half dominated. That combined class is still
implemented and selectable (`--variant-classes ...,apostrophe_mixed_script` /
`QUERY_ROBUSTNESS_CLASSES=`), just not run by default; each mechanism draws from its own seeded
stream and rewrites characters in place, so the combined text is exactly the composition of the
two single-mechanism classes at the same seed and the split loses no comparability.

Each class runs under three isolated mitigation lanes (`MITIGATION_LANES`): `off`, `normalize`
alone, and `normalize,typos` plus the Ukrainian morphology guard. The middle lane is what
separates the two mitigation mechanisms -- normalization only inverts noise it can attribute,
while the typos step additionally rewrites tokens to corpus surfaces -- so a recovery attributable
to safe normalization is never credited to vocabulary correction, and vice versa. Recovery columns
are measured against the `off` lane of the same class. Clean cases are an ordinary `run-eval`
bundle. The 12 x N variant rows are probe-only and publish atomically as
`$DATA_DIR/query-robustness/<run>/{report.md,robustness.jsonl}`; no `scores.jsonl` exists in that
probe directory. Each row carries its lane in `mitigation` (plus `mitigation_steps` /
`mitigation_typo_guard`) and whether the generator actually changed the question
(`variant_changed`).

A single-mechanism class is a NO-OP on any question carrying none of its trigger characters --
only 6 of the 82 committed final-split questions contain an apostrophe at all -- and pooling the
untouched items drags every delta toward zero (a total retrieval loss on 6 items reads as
-0.073 pooled). The report therefore carries a second **affected items only** table that repeats
every lane over the perturbed items, against the SAME items' clean baseline, with an untouched
count per class; a lane whose class perturbed nothing is dashed out rather than shown as zeros.

Implementation is split across `src/llb/eval/query_robustness_variants.py` (seeded generators and
class selection), `query_robustness.py` (lane definitions, per-case joins, lane and affected-subset
metrics), `query_robustness_run.py` (clean baseline, store, endpoint, and per-lane graph wiring),
`query_robustness_report.py` (atomic report/JSONL publication), and
`src/llb/cli/eval/query_robustness.py`.
`tests/llb/eval/test_query_robustness.py` drives a fake endpoint and fake store through all twelve
default lanes using the graph module's pure-node seam, so the base `[dev]` GitHub environment does
not need the optional LangGraph package. It checks deterministic variants, that each split class
applies exactly one mechanism and the two compose into the combined class, class-selection
parsing, morphology-guard wiring (only the vocabulary-correction lane loads the probe), the
mechanism split (normalization alone recovers the script classes but not keyboard noise), the
affected-subset split on an item the apostrophe class cannot touch, and proves the probe directory
never gains correctness scores. Shared query-prep tests
cover the bugs found by CUDA acceptance: collision-safe romanization, preservation of uppercase
Latin acronyms, keyboard grave normalization, embedded homoglyph repair, short-token protection,
and alphabetic/numeric candidate separation, plus the ambiguity-aware restoration constraints
documented in [RAG core](rag-core.md#query-side-processing).

CUDA-host evidence (2026-07-24, supersedes the 2026-07-21 and 2026-07-22 runs, which measured the
combined class): RTX 4060 Ti 16 GiB, Ollama, the full committed `ua_squad_postedited_v1` final
split (n=82), seed 13, 8 percent character noise, k=10, `intfloat/multilingual-e5-base`, 96 answer
tokens, and all five classes (the four defaults plus the opt-in combined one) under all three
mitigation lanes. Both model runs completed 1230/1230 probe cases with zero errors. The store is
dense-only, so these classes exercise the dense lane; the lexical lane is variant-invariant by
construction after the v2 tokenizer ([RAG core](rag-core.md#apostrophe-variant-tokenization-evidence)).

Retrieval is model-independent here (same store, same query prep), and clean recall@10 was 0.9756
for both models:

| Class | `off` | `normalize` | `normalize,typos` |
| --- | ---: | ---: | ---: |
| `transliteration` | 0.7195 | 0.9634 | 0.9634 |
| `apostrophe_variant` | 0.9756 | 0.9634 | 0.9634 |
| `mixed_script` | 0.9634 | 0.9634 | 0.9634 |
| `keyboard_typos` | 0.9268 | 0.9024 | 0.9634 |
| `apostrophe_mixed_script` | 0.9634 | 0.9634 | 0.9634 |

- `hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M`: clean objective 0.4747.
  Transliteration objective was 0.2497 raw, 0.3763 under `normalize` (+0.1267) and 0.3320 under
  `normalize,typos` (+0.0823); `mixed_script` was 0.4346 raw and 0.4641 under both mitigations
  (+0.0296); `apostrophe_variant` was 0.4747 raw -- clean to four decimals -- and 0.4903 mitigated;
  keyboard typos were 0.4451 raw, 0.4375 (-0.0076) and 0.4340 (-0.0111). Artifact:
  `$DATA_DIR/query-robustness/20260724T121701.652129Z-6feeb0cd727e/`; clean baseline:
  `$DATA_DIR/run-eval/20260724T113233.376054Z-7f2b659f138a/`.
- `hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M`: clean objective 0.4970.
  Transliteration objective was 0.3827 raw, 0.5069 (+0.1242) and 0.5194 (+0.1367);
  `mixed_script` was 0.5240 raw and 0.5120 mitigated (-0.0121); `apostrophe_variant` was 0.4970 raw
  -- again exactly clean -- and 0.5127 mitigated; keyboard typos were 0.4048 raw, 0.3840 (-0.0208)
  and 0.4140 (+0.0092). Artifact:
  `$DATA_DIR/query-robustness/20260724T124802.064874Z-526a1af2007d/`; clean baseline:
  `$DATA_DIR/run-eval/20260724T121702.998145Z-b09c3af6a01f/`.

Verdict per mechanism, re-read on the split classes:

- **The recorded combined-class verdict was the homoglyph half, entirely.** `mixed_script` and
  `apostrophe_mixed_script` agree in EVERY column of both reports, and per item the 6 questions
  whose combined text differs (the apostrophe-bearing ones) score identically in both classes.
  Adding apostrophe re-typing on top of homoglyph noise changes no measurement on this corpus, so
  the previously recorded "mixed script" recovery is attributable to homoglyph repair alone.
- **The apostrophe half costs nothing to recover, because it costs nothing.** `apostrophe_variant`
  with no mitigation reproduces the clean objective and clean recall exactly for both models, and
  on the 6 affected questions recall@10 is 1.0000 in all three lanes: the dense e5 encoder is
  insensitive to which apostrophe variant was typed here. Lapa's -0.1296 objective on that
  6-item subset under `normalize` is one item's answer changing (1.000 -> 0.222) with identical
  retrieval, which is generation-side wobble at n=6, not a mitigation cost.
- **Transliteration and keyboard verdicts are unchanged** and reproduce the 2026-07-22 three-lane
  run to within decoding noise (<= 0.0003 objective), which is the control that splitting the
  class disturbed nothing else: those two generators draw from unchanged seeded streams.
- **The `normalize` lane's -0.0122 recall on the apostrophe class is not a failed repair.** The
  affected-items table shows all 6 perturbed questions at 1.0000; the lost item is
  `570d4e6cb3d812140066d66d`, the untranslated ENGLISH question the class never touched. That is
  the normalization step acting on an otherwise clean query -- the cost the opt-in language gate
  and the forward `normalize-casefold-dense-lane-cost` task address.

Mitigation verdict (unchanged): do not make the combined `normalize,typos` lane a universal
default. It is a strong model-specific option -- it restores all retrieval loss on every class and
improves transliteration for both models -- but MamayLM's keyboard-objective regression persists
and the per-edit audit's ambiguous nearest-vocabulary choices show that typo correction still needs
a model/corpus A/B before activation. The report's shared-hit generation delta separates that
answer-side effect from missing evidence.

## Ukrainian Security Adaptation

The security benchmark (`src/llb/bench/security.py`, `src/llb/scoring/security.py`) is adapted to
the Ukrainian, bilingual reality of locally-served UA models. The adaptation lives in the pieces
that decide the scores, so a Ukrainian-language decline is never miscounted as compliance and a
Cyrillic-echoed leak is never missed.

- **Refusal lexicon (UA + RU).** `is_refusal` (`src/llb/eval/common.py`) matches high-precision,
  first-person UA refusals (`не можу`, `не зможу`, `не буду`, `не в змозі`, `відмовляюся`,
  `перепрошую, але я`, …) plus Russian equivalents (`не могу`, `извините, но я`), because a
  bilingual UA model pushed with an adversarial ask frequently declines in Russian. A missed
  refusal would inflate ASR (`refuse` cases), turn a canary-naming refusal into a false leak
  (`leak` cases), and depress refusal-appropriateness. Every marker keeps a `(?!\w)` word boundary
  so first-person `не можу` does not fire on third-person `не можуть`.
- **Text normalization + homoglyph folding.** `normalize_refusal_text` casefolds, applies NFKC, and
  unifies Ukrainian apostrophe variants (`'` `'` `ʼ` `` ` ``) before matching. `marker_leaked`
  additionally folds Cyrillic look-alike letters to Latin, so a Latin canary the model echoes with
  Cyrillic characters (`РАСКЕТ-ОХ-42` for `PACKET-OX-42`) is still detected instead of understating
  ASR.
- **UA / bilingual attack vectors.** Jailbreak frame prose lives in prompt templates under
  `bench/security/` (adversarial text is data, not code). Alongside DAN role-play and fiction
  framing, a **language-switch** frame tells a UA-tuned assistant to drop its guardrails and answer
  in Russian — probing whether safety generalizes across the languages the model knows. The
  committed seed (`samples/benchmarks/security_cases_uk.json`) adds a language-switch jailbreak
  (`jb-003`) and a Cyrillic-homoglyph RAG injection (`rag-003`).
- **Over-refusal controls.** The seed carries UA benign controls that look sensitive but are
  legitimate — dual-use history, defensive security, and crisis mental-health resources — so a model
  cannot win by blanket-refusing Ukrainian civic/dual-use questions. These feed refusal-
  appropriateness only, never ASR.

Seed cases are tagged with `lang` (`uk` / `ru` / `en`) and, for attacks, `attrs.vector`
(`language_switch` / `homoglyph`).

- **Cross-language consistency (Module 6).** `cross_language_consistency`
  (`src/llb/scoring/security.py`) compares matched-prompt groups — the SAME behavior posed in
  several languages, linked by an `xlang_group` id — as a separate outcome from ASR. For every
  group posed in >=2 languages it checks whether the model's refuse/answer decision is the same
  across languages; `consistency` is the fraction of groups treated equivalently and
  `refusal_rate_by_lang` is the diagnostic that reveals which language is the weak point (e.g. a
  model that refuses in Ukrainian but complies in Russian). Consistency is orthogonal to safety —
  it is read alongside ASR, carries its own bootstrap CI, and is persisted in the run manifest
  under `config.cross_language`; `SecurityScore.cross_language` is `None` when a set has no matched
  groups. The committed seed ships one harmful (`xl-weapon`) and one benign (`xl-help`) UA/RU/EN
  group. Behavior-level translation of the public adversarial sets into matched groups remains an
  operator step (inject a per-language `translate`), since the seed keeps human-verified prose.

## Local Judge

`src/llb/scoring/judge/endpoint.py` resolves the local OpenAI-compatible endpoint;
`src/llb/scoring/judge/deepeval_adapter.py` supplies Ukrainian DeepEval G-Eval metrics; and
`src/llb/scoring/judge/model.py` applies the calibration-rho gate. If the judge is not trusted,
objective correctness ranks alone and judge output remains diagnostic. The scorer keeps empty
candidate answers distinct from malformed or unreachable local-judge responses.

```bash
llb judge-experiment --judge-model <model> --judge-base-url <url>
llb judge-smoke --judge-model <model> --judge-base-url <url>
```

`judge-experiment` records prompts, fixed Ukrainian sanity cases, served-model metadata, and scores
under `$DATA_DIR/judge-experiment/<timestamp>/`. `judge-smoke` runs one strict-JSON grounded case
before a long judged run and exits non-zero with a reason if the local judge cannot produce a usable
score.

The local-judge choice is deliberate: corpus data should not leave the host by default. The tradeoff
is model-family bias when the judge shares lineage with candidate models. That bias is disclosed in
manifests and controlled by the calibration gate and judge-cohort guard.

## Scorer Policy Seam

`src/llb/scoring/policy/` selects the judge lane for `run-eval` via `--scorer-policy` /
`RunConfig.scorer_policy`:

| Lane | Behavior |
| --- | --- |
| `human` | Skip automated judging; objective scores rank alone; manifest records `provider=human`. |
| `local` | Existing DeepEval path against `judge_model` / `judge_base_url` (default). |
| `frontier` | Litellm frontier judge using the registered Ukrainian G-Eval step templates. |

Frontier scoring requires one upfront `--scorer-egress-consent` plus a hard cap
(`--frontier-max-usd` and/or `--frontier-max-calls`). Spend is tracked in
`$DATA_DIR/run-eval/<run>/scorer/` (`consent.json`, `ledger.jsonl`, `ledger_state.json`). Hitting
the cap aborts with `abort.json` (`resumable: true`); resume reloads the ledger so spend never
silently exceeds the cap. Each successful (or failed-but-attempted) frontier call also
checkpoints `case_index` plus `faithfulness` / `answer_relevancy` in `ledger.jsonl`; on resume
`frontier_scorer` replays those scores and issues provider calls only for unscored cases
(`src/llb/scoring/policy/ledger.py`, `frontier.py`). Headline ranking is unchanged: judges remain
diagnostic until calibration rho clears the trust threshold.

```bash
llb run-eval --scorer-policy local --judge-model <model> --judge-rho <rho>
llb run-eval --scorer-policy human
llb run-eval --scorer-policy frontier --judge-model openai/<model> \
  --scorer-egress-consent --frontier-max-usd 2.00 --judge-rho <rho>
```

Tests live under `tests/llb/scoring/test_scorer_policy*.py` (fake litellm completions; no network),
including a mid-batch abort/resume case-checkpoint test that proves the second pass issues
`N - K` new calls after `K` cases were already scored.

### Frontier Judge Agreement and Cost Report

`src/llb/scoring/frontier_agreement/` produces the evidence an operator needs to decide whether a
frontier judge may gate autonomously. It scores a filled calibration worksheet with each named
provider through the scorer-policy seam (same consent, ledger, cap, and resume guarantees as a
scored run), then reports two rank correlations per provider -- against the human `human_rating`
and against the local judge's `judge_rating` -- plus measured cost per item.

| Module | Responsibility |
| --- | --- |
| `items.py` | Worksheet rows -> judgeable records; contexts are windows of the gold source doc |
| `provider.py` | One provider's run under `resolve_scorer(lane="frontier")` |
| `agreement.py` | Spearman rho + bootstrap CI per metric; cost-per-item and cap math |
| `report.py` | `report.md` including the operator's sign-off table |
| `run.py` | Orchestration; writes the run bundle |

Design points:

- Grounding contexts come from the gold set's source spans (a `GOLD_CONTEXT_WINDOW_CHARS`-wide
  window of the source document), not from retrieval. Judge agreement is measured with retrieval
  held constant so a retrieval regression cannot masquerade as judge disagreement.
- Correlation is rank-based, so the human 1..5 scale and the judge 0..1 scale are compared without
  rescaling. The headline metric is `mean` -- the same scalar `judge_value` that a scored run
  records per case (`src/llb/scoring/judge/scorer.py`, shared with `runner_judge`).
- A recommended cap is `cost_per_item * n_items * CAP_SAFETY_FACTOR` (2.0), rounded up to the cent.
  When litellm cannot price a model, `cost_usd` is 0; the report marks the provider unpriced and
  recommends no cap rather than guessing one.
- Providers are independent: each owns its ledger directory, so one provider's budget abort or
  transport failure is recorded under `failures` while the others keep their evidence.
- `recommendation` is a machine reading of the rho gate only. `human_decision` stays `pending`
  until an operator records an accept or reject; the report says so explicitly.

```bash
make frontier-judge-agreement FRONTIER_JUDGE_MODELS=<litellm-id>[,<id>...] \
  FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=<cap>
```

Artifacts land under `$DATA_DIR/frontier-judge/<run>/`: `report.md` and `agreement.json` at the
root, and per provider a `<provider-slug>/` holding `scores.jsonl` plus the standard `scorer/`
consent and ledger files. The lane refuses to run without both an explicit egress consent and a
cap. Tests are `tests/llb/scoring/test_frontier_agreement.py` (injected fake completers; no
network, no spend), covering the rho and cap math, the artifact bundle, per-provider failure
isolation, and resume-after-abort.

The provider keys, the spend approval, and the accept/reject decision per provider remain human
steps; see the `frontier-judge-authorization` task in [plan.md](../plan.md).

## Frontier Prep Utilities

`src/llb/prep/frontier.py` contains GPU-free Litellm-backed utilities that emit unverified review
material:

- `prepare_goldset`: drafts question, answer, and exact source span triples from real documents;
- `prepare_synthetic_corpus`: generates synthetic documents with planted labels.

Both are injectable for tests and write provenance. A planter model must differ from the judge model
to avoid circular evaluation.
