# Extended Workflows

Extended workflows cover comparison axes that sit beside the main RAG leaderboard: agentic
harnesses, judge diagnostics, and prompt-system packages.

## Agentic Harness Comparison

The agentic benchmark can run the same task set through multiple harnesses while keeping the model,
tools, world state, objective checks, and optional judge fixed.

Core locations:

- `src/llb/bench/agentic.py`: `Harness` protocol, harness names, runner integration;
- `src/llb/bench/harness/base.py`: pure loop harness;
- `src/llb/bench/harness/langgraph.py`: LangGraph agent/tool graph;
- `src/llb/bench/harness/crewai.py`: CrewAI adapter;
- `src/llb/board/harnesses.py`: one-model harness comparison board rows.

```bash
llb bench-agentic --harness loop --model <model> --backend <backend>
llb bench-agentic --harness langgraph --model <model> --backend <backend>
llb bench-agentic --harness crewai --model <model> --backend <backend>
llb bench-agentic-compare --model <model>
make agentic-harness-compare MODEL=<model> BACKEND=<backend>
```

The comparison fixes the model and treats harness as the row label. This avoids conflating model
quality with orchestration behavior.

CrewAI is optional and lazy-imported. The adapter wraps the candidate completion function as a
CrewAI LLM, builds tools from the benchmark tool definitions, and disables telemetry/tracing for a
local no-egress run.

The `[crewai]` extra is a standalone install lane in `uv`: upstream CrewAI pins older Chroma,
LanceDB, and `tomli` ranges than the repo's RAG/vector/dev extras. `pyproject.toml` declares those
extra conflicts so `uv lock` stays resolvable while `uv pip install -e ".[crewai]"` still works for
host validation.

## Judge Diagnostics

`src/llb/scoring/judge_diag.py` classifies zero-valued judge outcomes so a diagnostic score can be
read correctly:

- `empty_answer`: candidate produced nothing useful;
- `malformed_judge_json`: judge endpoint failed strict JSON expectations;
- `judge_transport_error`: judge endpoint failed transport;
- `zero_score`: judge returned a valid zero.

`run_gated_judge` attaches diagnostics to category judge outcomes, and category manifests carry the
summary under judge metadata. `bench-agentic` also echoes the diagnostic summary.

Before a long judged run, use:

```bash
llb judge-smoke --judge-model <model> --judge-base-url <url>
```

The smoke check runs one grounded case and exits non-zero with a reason when the judge cannot
return a well-formed non-zero strict-JSON score.

## Prompt-System Packages

`src/llb/prompt_system/` builds reviewable RAG prompt-system candidates from a corpus. The package
is deterministic and manifest-addressable so prompt changes become explicit experiment variables.

Important modules:

- `corpus.py`: reads `.md`/`.txt`, keeps exact spans, selects anthology passages, builds metadata;
- `budget.py`: token-budget planning and section trimming;
- `template.py`: prompt fields and `PromptPackage.apply`;
- `tuning.py`: candidate grid and deduplication;
- `review.py`: approve, pin, reject, and persist candidate review state;
- `manifest.py`: corpus, mapping, template digests, and stable prompt-system ids;
- `selection.py`: resolves a selected package for `run-eval`.

```bash
llb prompt-system-prepare --corpus-root <dir> --out-dir <review-dir>
llb prompt-system-review --run-dir <review-dir> --action summary
llb prompt-system-review --run-dir <review-dir> --action pin --id <prompt-id>
llb run-eval --prompt-system <prompt-id> --prompt-package <review-dir> ...
llb prompt-system-compare --lane rag --model <model>
```

`run-eval` prepends the selected prompt package to the normal RAG generation prompt and records
`prompt_system_provenance` in the manifest. Board loaders can rank one model across prompt-system
ids for RAG or agentic lanes.

## Sample Prompt Assets

The IP regulation samples provide a small checked prompt-system fixture:

- `samples/goldsets/ip_regulation_uk/`;
- `samples/prompt_system/ip_regulation_uk/`;
- `samples/prompt_system/ip_regulation_uk/tuned/`;
- `samples/prompt_system/ip_regulation_uk/graph/`.

These samples are useful for local prompt-system mechanics and board rendering. Treat tuning wins
as provisional until a held-out final split confirms them; the prompt-system lane exists to make
that split discipline visible.

## Local Self-Improvement Loop

The self-improvement workflow closes the loop from a measured local RAG run to an adapter-backed
candidate row. It is file-driven and split-guarded:

- `src/llb/finetune/dataset.py` exports SFT records and optional DPO preference pairs from a
  finalized tuning-split run bundle. The exporter renders `eval.rag.chat` messages through the
  same prompt path as `run-eval`, writes `sft.jsonl`, `dpo.jsonl`, and `dataset_manifest.json`,
  and records the item ids, split counts, source run, and dataset digest.
- `src/llb/finetune/trainer.py` trains LoRA/QLoRA adapters behind a trainer seam. `--trainer fake`
  writes deterministic CI artifacts; the real path lazy-imports PEFT, TRL, Transformers, and
  Datasets from the `[finetune]` extra and saves an adapter plus `adapter_manifest.json`.
- `src/llb/finetune/guard.py` enforces the contamination invariant before `run-eval` launches a
  backend: adapter manifests may contain only tuning-split training ids, may not intersect
  calibration/final eval ids, and a tuned model cannot judge itself.
- `src/llb/finetune/loop.py` orchestrates base final eval, per-round tuning eval, miss analysis,
  dataset export, adapter training, adapter final eval, stop/accept logic, `state.json`, and
  `report.md`.
- `src/llb/finetune/campaign.py` schedules the loop ingredients across a `--models` roster with
  planner skip reasons, a shared campaign SFT export, per-model preference exports, VRAM reclaim
  between roster entries, `campaign.progress.jsonl` resume, and a tunability `report.md`.
- `src/llb/finetune/registry.py`, `lifecycle.py`, and `serving.py` make adapters first-class,
  traceable artifacts (see [Adapter Registry And Lifecycle](#adapter-registry-and-lifecycle)).
- `src/llb/finetune/hparam_search.py` searches the LoRA space per model and feeds the winning
  config back as the trainer's defaults (see
  [Hyperparameter Search](#hyperparameter-search)).
- `src/llb/finetune/naming.py` holds `model_slug`, the one filesystem name a model gets across the
  campaign and hyperparameter artifact trees.

Commands:

```bash
llb export-finetune-set --run-dir <tuning-run> --goldset <goldset> --out <dataset-dir>
llb finetune-adapter --dataset <dataset-dir> --model <model> --seed <seed>
llb self-improve --model <model> --backend vllm --goldset <goldset> --rounds 2
llb finetune-campaign --models <m1,m2> --backend vllm \
  --goldset <goldset> --corpus <corpus-dir> --rounds 1
make self-improve MODEL=<model> BACKEND=vllm GOLDSET=<goldset> ROUNDS=2
make finetune-campaign MODELS=<m1,m2> BACKEND=vllm GOLDSET=<goldset> CORPUS=<corpus-dir>
```

Artifacts live under `$DATA_DIR/self-improve/<timestamp>/round-<n>/` for campaign state and under
`$DATA_DIR/run-eval/` for canonical board bundles. Round directories carry `dataset/`, `adapter/`,
`run` and `run-final` pointers, plus per-round reports.

Multi-model campaign artifacts live under `$DATA_DIR/finetune-campaign/<timestamp>/`. The campaign
root contains `shared-dataset/dataset_manifest.json`, `campaign.progress.jsonl`, `report.md`, and
one directory per roster model. Each model directory records base-final and per-round tuning/final
run pointers, miss analysis, a per-model preference dataset, and the final adapter. Resume replays
`campaign.progress.jsonl` and does not retrain a completed roster entry.

Adapter-backed `run-eval` rows are labeled `<base>+adapter-<digest>` in manifests and board loaders.
`llb recommend` appends a self-improvement section when a campaign `state.json` exists and a
fine-tune campaign section when `$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` exists. The
campaign section ranks completed models by final-split delta, then shorter training wall-clock, then
lower peak VRAM; skipped models remain visible with the planner reason.

Tests:

```bash
uv run pytest tests/test_finetune.py tests/test_adapter_registry.py tests/test_recommend.py
```

The campaign implementation is covered by fake eval/trainer/planner tests for scheduling order,
planner skip reasons, shared dataset digest reuse, JSONL resume, and report ranking.

## Hyperparameter Search

`src/llb/finetune/hparam_search.py` searches the LoRA configuration space for one model with a
bounded budget, so fine-tuning stops guessing rank, alpha, learning rate, epochs, and target
modules.

Dependency contract: the `[finetune]` and `[dev]` extras include Optuna. GitHub CI installs
`.[dev]`, so the fake-trainer hparam tests stay in the lightweight `make ci` suite without pulling
the CUDA training stack.

```bash
llb finetune-hparams --model <m> --dataset <tuning-dataset> --backend vllm \
  --goldset <goldset> --max-trials 8 [--max-hours 2] [--seed 13] [--dev-fraction 0.25]
llb finetune-hparams ... --resume <study-dir>
make finetune-hparams MODEL=<m> DATASET=<dir> GOLDSET=<g> MAX_TRIALS=8
```

Artifacts land under `$DATA_DIR/finetune-hparams/<model-slug>/<timestamp>/`: `study.db` (the
persistent Optuna study), `trials.jsonl` (a live progress log), `trials/trial-<n>/` (the trial's
train-slice dataset and adapter), and `hparams_manifest.json` (best config, study seed, dev slice,
budget, and the full trial table).

### Split discipline

The discipline of `optimize/tuner.py` extends one level down. That tuner searches RAG and serving
knobs on the tuning split while `final` stays held out; here the search space is the LoRA config
itself, and the held-out set is carved from *inside* the tuning split:

- `carve_dev_slice` seeds a deterministic, disjoint train/dev partition of the dataset's item ids.
  Each trial trains only on the train sub-slice and is scored only on the dev sub-slice, so a trial
  never sees its own evaluation items.
- `assert_tuning_only` refuses the search outright when the dataset's `split_counts` name any split
  but `tuning`, and -- when a goldset is available -- when its item ids intersect the real
  calibration/final ids. A dataset manifest is operator-writable, so its split counts alone are not
  proof (the same lesson the registry records for adapter manifests).
- The default objective scores the trial adapter through `run_eval` over the dev items only. It
  refuses a non-vLLM backend and a missing goldset BEFORE the study is created: the first trial
  fine-tunes a model before it ever reaches the objective, so a late refusal would waste a full
  training run.

### Budget and resume

`--max-trials` caps the trial count; `--max-hours` caps wall clock. A trial is atomic (a whole
fine-tune), so the wall-clock budget is checked BETWEEN trials through an Optuna callback -- one
in-flight trial may overrun the deadline and is never killed mid-training. An aborted study records
`budget_exhausted: true` and stays resumable: the SQLite study persists, and `--resume <dir>` runs
only `max_trials - len(study.trials)` further trials, so finished trials are never repeated.

A measured OOM prunes its trial (reusing `optimize.tuner.is_oom`) instead of crashing the study; any
other exception fails loudly -- but only after `hparams_manifest.json` is written, so a study killed
by one bad trial stays inspectable and resumable instead of leaving a bare `study.db`.

### Feeding the trainer

`trainer_defaults(data_dir, model)` reads the newest `hparams_manifest.json` for that model and
returns `{"hyperparameters": <best>, "hparams_manifest": <path>}`. It is the default trainer wiring
for `self-improve`, `finetune-campaign`, and `finetune-adapter` (which accepts `--default-hparams`
to opt out). `train_adapter` records `hparams_manifest` in `adapter_manifest.json` as pure
provenance: it never enters `adapter_digest`, because two adapters with identical hyperparameters
are the same adapter whether or not a search chose them.

Discovery only scans the default tree `$DATA_DIR/finetune-hparams/<model-slug>/<timestamp>/`. A
study written elsewhere with `--out-dir` is a one-off: it is never auto-consumed as a trainer
default.

`dataset.subset_dataset` materializes each trial's train sub-slice as a real dataset directory with
its own recomputed digest. A filtered view would inherit the parent's `dataset_digest`, and since
`adapter_digest` derives from it, two adapters trained on different data would collide on one
registry id.

Tests: `tests/test_finetune_hparams.py` covers dev-slice disjointness and determinism, both guard
refusals, the no-protected-id-in-any-trial invariant, a seeded study reproducing its trial table,
the budget abort plus resume without repeated trials, OOM pruning, the manifest surviving a failed
trial, subset digests, and the trainer consuming a recorded best config through a self-improvement
round.

### CUDA evidence on the 12 GB RTX PRO 3000 host

An 8-trial search for `Qwen/Qwen2.5-0.5B-Instruct` over the `ua_squad_postedited_v1` tuning split
(82 verified items -> 62 train / 20 dev at `dev_fraction=0.25`, `seed=13`).

- Tuning-split base run: `objective 0.2610`, reliability `1.000`, recall@3 `0.915`, `177.7` tok/s;
  the dev slice's base objective is `0.2056`.
- Study: `.data/finetune-hparams-evidence/study/hparams_manifest.json`
  (`finetune-hparams-Qwen-Qwen2.5-0.5B-Instruct-313415c09b62-s13`); 8 complete, 0 pruned; each trial
  fine-tunes the 62 train items and scores the 20 dev items through vLLM LoRA serving in `60` to
  `99` s.

| trial | dev objective | rank | alpha | dropout | learning rate | epochs | target modules |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.3233 | 16 | 64 | 0.05 | 2.96e-05 | 3 | qv |
| 1 | 0.2917 | 8 | 16 | 0.00 | 1.18e-04 | 4 | attn_mlp |
| 7 | 0.2861 | 32 | 64 | 0.00 | 1.88e-04 | 1 | attn |
| 6 | 0.2789 | 64 | 128 | 0.00 | 1.38e-05 | 2 | qv |
| 0 | 0.2674 | 64 | 256 | 0.05 | 1.26e-05 | 4 | attn_mlp |
| 4 | 0.2583 | 4 | 8 | 0.15 | 4.71e-04 | 4 | qv |
| 3 | 0.2059 | 4 | 8 | 0.00 | 2.61e-05 | 1 | attn |
| 5 | 0.2056 | 16 | 16 | 0.10 | 1.66e-05 | 4 | qv |

The best config (trial 2) scores `0.3233` on the dev slice against the base model's `0.2056`, and the
spread across trials is non-saturated, so the search discriminates rather than tying. Rank is not
monotonic: the two rank-4 points bracket the field and the widest module preset (`attn_mlp`) does not
win, which is the whole reason to measure rather than guess.

Two caveats the numbers carry:

- The dev slice is drawn uniformly, and this base model answers only a minority of items. A first
  attempt on a 12-item dataset produced a 3-item dev slice holding ONE answerable item, and all
  trials tied at `0.0000` -- the objective was a constant. The full 82-item tuning split fixed it;
  a stratified slice would fix it properly (see the forward task in `plan.md`).
- Trial 5 lands exactly on the base objective `0.2056`: a tuned adapter is not automatically better
  than no adapter, and the search records that honestly.

## Adapter Registry And Lifecycle

Adapters are first-class artifacts, not loose directories. `$DATA_DIR/adapters/registry.jsonl` is an
append-only event log (`register` / `merge` / `delete`) folded into the current entry set on read, so
a partial write can never lose earlier history. The entry id IS the `adapter_digest`, so it can never
be reassigned to different weights.

Modules:

- `src/llb/finetune/registry.py`: `AdapterEntry`, the event log, `register_adapter` (idempotent --
  an unchanged re-registration appends nothing), `resolve_adapter` (id / unique prefix / label /
  directory), and `staleness`;
- `src/llb/finetune/lifecycle.py`: run-bundle citation scan, supersession, and garbage collection;
- `src/llb/finetune/serving.py`: the serve plan, the cached merge lane, and the backend seam.

```bash
llb register-adapter --adapter-dir <dir> [--goldset <g>] [--corpus <c>] [--source-run <run>]
llb list-adapters [--json]
llb serve-adapter --adapter <id> --backend vllm|ollama|llamacpp [--smoke]
llb gc-adapters [--dry-run] [--force]
llb run-eval --adapter <id> --model <base> --backend vllm
make list-adapters ; make serve-adapter ADAPTER=<id> BACKEND=<b> ; make gc-adapters GC_DRY_RUN=1
```

Entries record the base model, dataset digest, dataset item ids and split counts, the goldset and
corpus digests observed AT TRAINING TIME, the source run, and an eval summary. Self-improvement and
campaign rounds auto-register through `register_round_adapter` after the adapter's own final eval,
so the entry carries the evidence the board later cites. Registration is best-effort: an injected
trainer that writes no `adapter_manifest.json` logs a warning instead of aborting the round. A bare
`llb finetune-adapter` does not register, so `llb register-adapter` exists to adopt a hand-trained
adapter into the registry rather than leave its board row silently dropped.

### Staleness

`staleness()` compares the recorded goldset/corpus digests against the present ones
(`durability.goldset_digest` and `corpus_governance.corpus_fingerprint`, the same functions the
durable-run journal and the stale-store check use). Verdicts are `current`, `stale`, and `unknown`;
a missing digest yields `unknown` and never `current`. Detection reports, it never retrains.

`board/runs.py` resolves every adapter-backed bundle through the registry before it can rank:

- an unregistered adapter's row is DROPPED (a tuned number nobody can trace is not comparable);
- a registered-but-stale adapter's row is stamped `<base>+adapter-<digest> [stale]`.

`recommend.load_run_summaries` reuses `load_run_records`, so both the board and `llb recommend`
inherit the rule from one seam.

### Contamination guard through the registry

`validate_adapter_for_eval` reads training provenance from the registry when the adapter is
registered, falling back to `adapter_manifest.json` only when it is not (a freshly trained adapter
registers after its first eval). The manifest beside the weights is operator-writable, so a
hand-edited one could otherwise launder a final-split adapter past the gate. The refusal message
names the intersecting ids, the offending splits, and which provenance was consulted.

### Serving

vLLM serves the LoRA directly through the existing `--enable-lora --lora-modules` wiring, sized by
`--max-lora-rank`. That flag defaults to 16, so an adapter trained at a higher rank fails
`add_lora` at engine startup (`LoRA rank 64 is greater than max_lora_rank 16`) and vLLM exits before
serving anything. Both adapter launch paths (`executor/runner.py` for `run-eval`, `serving.py` for
`serve-adapter`) therefore read the rank off the adapter they are about to serve --
`trainer.adapter_lora_rank` prefers PEFT's own `adapter_config.json` over our manifest, since it
describes the weights actually on disk -- and `backends/vllm.served_lora_rank` rounds it up to the
nearest value vLLM accepts (`1, 8, 16, 32, 64, 128, 256, 320, 512`). An adapter of unknown rank
leaves the flag off and vLLM keeps its default.

Ollama and llama.cpp serve whole model artifacts, so `serving.py` merges the adapter into its base
weights
(PEFT `merge_and_unload`), converts to GGUF via the llama.cpp checkout's `convert_hf_to_gguf.py`, and
for Ollama registers a `llb-adapter-<short-id>` tag. The merge is expensive and one-way, so it is
cached under `$DATA_DIR/adapters/merged/<short-id>/<backend>/` behind a `merge.json` and recorded as
a registry `merge` event. Both the merge and the launcher are injectable, so CI exercises all three
backends without CUDA, llama.cpp, or a running Ollama daemon. `serve-adapter` probes the endpoint
with one generation and then holds it in the foreground until Ctrl-C; there is no serving daemon.

### Garbage collection

An adapter is superseded once a newer adapter exists for the same base model, ordered by
`(created_at, log sequence)`. `created_at` has second resolution, so two fast rounds tie; the
append-log position breaks the tie exactly. Only superseded adapters are GC candidates, and GC
refuses any that a published run bundle cites (matched by recorded digest or by served
`adapter_path`). `--force` overrides the citation refusal but never the safety rule that GC only
deletes directories inside `$DATA_DIR`. Deletions append a `delete` tombstone.

### Committed fixtures

- `samples/finetune/registry/registry.jsonl`: a stale entry (with a folded `merge` event) and a
  poisoned-digest entry, both pointing at adapter dirs outside `$DATA_DIR`;
- `samples/finetune/stale-adapter/`: recorded digests that no longer match
  `samples/goldsets/ip_regulation_uk/`;
- `samples/finetune/laundered-adapter/`: an `adapter_manifest.json` that CLAIMS a clean tuning-only
  training set while the registry records the `final`-split ids it was really trained on;
- `samples/finetune/poisoned-adapter/`: the simpler case where the manifest itself declares the
  protected split, refused even when unregistered.

`tests/test_adapter_registry.py` covers registry round-trip and idempotence, the staleness flip when
the goldset digest changes, the `unknown` verdict, guard resolution through the registry, serving
smoke over a fake launcher for all three backends, merge-event recording and merge caching, GC
citation refusal plus `--force`, the same-second supersession tie, the outside-`$DATA_DIR` safety
rule, and board drop/stamp behavior.

CUDA evidence on the 12 GB RTX PRO 3000 host:

- Command shape: `LLB_EMBED_DEVICE=cpu llb finetune-campaign --config
  .data/quickstart-leaderboard/llb/serving/gpu-12gb/run_eval_gemma_4_12b_vllm.yaml --models
  Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct --corpus
  samples/goldsets/ua_squad_postedited_v1/corpus --rounds 1 --limit 1 --out-dir
  .data/finetune-campaign/task19-evidence-qwen-small-12gb`.
- Campaign report:
  `.data/finetune-campaign/task19-evidence-qwen-small-12gb/report.md`.
- Recommend summary:
  `.data/recommend/task19-summary.md`.
- Shared dataset digest: `5b99939c91b02500eda6fe3aa7cb27c46012928929f93def380a245b4a6711b0`.
- `Qwen/Qwen2.5-0.5B-Instruct`: base final objective `0.0000`, adapted objective `0.0000`,
  delta `0.0000`, train wall-clock `6.7800` s, adapted peak VRAM `11862` MiB.
- `Qwen/Qwen2.5-1.5B-Instruct`: base final objective `0.0000`, adapted objective `0.0000`,
  delta `0.0000`, train wall-clock `6.4219` s, adapted peak VRAM `11690` MiB.
- `llb recommend --gpu-gb 12 --no-chart` rendered the fine-tune campaign section and selected the
  0.5B base model for this smoke cohort because all one-case objectives were tied at zero and the
  base model was faster than its adapter-backed row.
- `google/gemma-4-12B-it-qat-w4a16-ct` served on the same host at `max_model_len=1024`
  (`41.8` to `42.9` tok/s, peak VRAM about `11523` MiB), but PEFT LoRA injection could not train
  the compressed-tensors QAT checkpoint because its compressed linear modules do not expose the
  normal `weight` attribute.
