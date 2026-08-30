# Backend Resolution, Sweeps, And Search

Part of the [Evaluation rigor](../rigor-board-judge.md) area of the [current implementation index](../../current.md).

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
  (`DEFAULT_LOCAL_CANDIDATES` in `src/llb/rag/embedding_bakeoff/run.py`); override with
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
- **vLLM serving utilization** -- sampled from `GPU_MEMORY_UTILIZATION_RANGE` (0.70-0.90, step
  0.05) for vLLM backends only, and clamped under the run's DECLARED
  `RunConfig.gpu_memory_utilization`
  (`suggest_gpu_memory_utilization` in `src/llb/optimize/tuning_space.py`). The declared value is a
  host fact, not a search axis: a trial served above it is a launch nobody authorized, and on a
  card whose validated value is 0.80 it is an out-of-memory graph capture rather than a slow trial.
  A ceiling at or below the range floor pins the value instead of sampling it, so a very tight host
  still runs its trials. Because the `RunConfig` default is 0.85, the sampled band is 0.70-0.85
  unless a config file or a flag declares otherwise.

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
   `screen/<slug>-r<round>.json` so a resume skips re-evaluation. A caller whose screen size was
   DERIVED rather than chosen passes `screen_case_cap=` as a ceiling over every round, so the
   growth cannot spend more items than the derivation priced; only the
   [roster confirmation](roster-confirmation.md) passes one, and the run manifest records it.
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

### Serving knobs a search carries

Both `joint-search` and `joint-search-long-run` take the serving inputs `run-eval` takes:
`--config <run config>`, `--gpu-memory-utilization`, and `--max-model-len`
(`src/llb/cli/models/search_serving.py`; `JOINT_SEARCH_CONFIG=`, `JOINT_SEARCH_GPU_UTIL=`,
`JOINT_SEARCH_MAX_MODEL_LEN=` on the make targets). The values land on the base `RunConfig`, which
is what `candidate_config` derives every screen cell, every finalist tune, and the confirmation
run's public screen from -- so one declaration reaches all of them, and the tune's sampled
utilization is clamped under it (see [the serving-utilization knob](#multi-objective-rag-tuner)).

Why the flag and not the guard: the pre-launch VRAM guard
(`guard_vllm_contention`, `src/llb/executor/runner_backend.py`) derates against OTHER processes'
memory, so on a quiet card a too-high request reads as free and is passed through unchanged. What
follows is not a slower run but a LOST candidate -- the vLLM engine exceeds its own budget during
graph capture and the candidate drops out of the screen, which on the board is indistinguishable
from a candidate that scored badly. Declare the value the host class was validated for
([acceptance paths](../host-validation/acceptance-paths.md#backend-paths)).

`--max-model-len` resolves flag, then config file, then `DEFAULT_SEARCH_MAX_MODEL_LEN` (8192): a
search always names a window, because every model on this roster has a 128k+ native one whose KV
cache no 16 GiB card can hold. `manifest.json` records both under a `serving` block, so a board can
be told apart from one taken at a different utilization.

Host evidence (2026-08-28, RTX 4060 Ti 16 GiB CUDA host, vLLM 0.23.0). One vLLM candidate --
Gemma 4 E4B int4 `w4a16` (`google/gemma-4-E4B-it-qat-w4a16-ct`), sliced out of `models_uk.yaml`
with its Ollama/llama.cpp alternates removed so the resolver had to serve it through vLLM or skip
it -- searched over the committed `ua_squad_postedited_v1` gold set (250 items, 82 tuning / 82
held-out) and its corpus, embedder `intfloat/multilingual-e5-base` pinned to CPU, no LLM judge, a
bounded `--trials 2 --screen-limit 3 --min-finalists 1 --limit 3`:

- The run config file declared `gpu_memory_utilization: 0.85` and `max_model_len: 8192`; the run
  was started with `--gpu-memory-utilization 0.80` and no `--max-model-len`. All six vLLM launches
  -- the screen cell, the tuning trial's progressive pruning subsets, and the final pick scoring --
  reported the pre-launch guard line `gpu-memory-utilization 0.80 fits`, and the command echoed
  `serving: gpu_memory_utilization=0.8 max_model_len=8192`. **Reading: precedence holds in both
  directions on a real launch -- the flag beat the file's utilization, and the file supplied a
  window no flag named.**
- The candidate SCREENED, at `backend=vllm`, tuning-split quality `0.827` and `0.940 s` latency on
  three cases. **Reading: this is the whole outcome -- before, the same candidate could only reach
  a board through Ollama, because the search had no way to ask for the 0.80 this tier is validated
  for.**
- The tuning trial's sampled `gpu_memory_utilization` came back at `0.80`, the declared ceiling,
  never the `0.85`/`0.90` the unclamped 0.70-0.90 band allows; both final picks carry
  `gpu_memory_utilization=0.8, max_model_len=8192` in their overrides, and `manifest.json` records
  the same pair under `serving`. **Reading: the declaration reaches the tune and the board, not
  just the screen.**
- Both picks scored `0.777` on three held-out cases. **Reading: three cases is a plumbing check.
  These quality numbers say nothing about the model and must not be quoted as a reading of it.**
- Two host facts the run cost to learn. With the embedder resident on the GPU the pre-launch guard
  ABORTED the first screen cell (12439 MB free against a ~12609 MB need, "no other GPU process");
  `LLB_EMBED_DEVICE=cpu` freed it and the same cell reported 14565 MB free and launched. And the
  FIRST tuning trial died with `vLLM exited (code 1) during startup`, whose log lived in the
  `emit=False` temp run dir and was deleted with it; the identical utilization reproduced clean in a
  standalone `run-eval`, and the retried trial completed, so it is not attributable to the serving
  values. **What would overturn this:** a repeat of that startup failure on a quiet card would make
  it a real defect rather than a flake. That repeat is now readable rather than lost -- a failed
  launch keeps its server log and the raised error names where ([a failed launch names a log that
  still
  exists](../host-validation/acceptance-paths.md#a-failed-launch-names-a-log-that-still-exists)) --
  so the next occurrence says which without re-running the search.

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
  adoption decision. The research-scale confirmation that IS enough -- predeclared effect, derived
  screen size, a ranking-stability stopping rule, the full held-out split, the public tracks, and an
  adopt-or-retain verdict -- is [roster confirmation and the adoption
  verdict](roster-confirmation.md).
