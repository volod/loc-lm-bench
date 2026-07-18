# Extended Workflows

Extended workflows cover comparison axes that sit beside the main RAG leaderboard: agentic
harnesses, judge diagnostics, and prompt-system packages.

## Agentic Harness Comparison

The agentic benchmark can run the same task set through multiple harnesses while keeping the model,
tools, world state, objective checks, and optional judge fixed.

Core locations:

- `src/llb/bench/agentic/model.py`: `Harness` protocol and harness names;
- `src/llb/bench/agentic/run.py`: runner integration;
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

## Context-Policy Comparison

`bench-chain-context` ranks context-management POLICIES for ONE fixed model over a verified
chain-of-questions set. It holds the model, the chain set, retrieval, and the scoring fixed and
varies only the policy -- the row label -- exactly as the agentic harness comparison holds
everything fixed and varies the harness. It answers "which harness and context policy improves
scores, and how should system prompts be sequenced" over multi-step questions where each step's
answer depends on the prior steps.

Core locations:

- `src/llb/bench/chain_context.py`: the four policies, the pure per-step context-assembly seam,
  the scored per-policy run, run-bundle persistence, and the recommendation block;
- `src/llb/board/chain_context.py`: one-model policy comparison board rows under
  `TIER_CHAIN_CONTEXT` (mirrors `board/harnesses.py`);
- `src/llb/prompts/templates/bench/chain_context/`: the reviewable role/instruction prompt-system
  templates and the step scaffold.

The four policies (each a fresh retrieval per step PLUS a different memory of the prior steps):

- `fresh` -- fresh retrieval per step, NO prior-step carryover (the naive baseline);
- `history` -- the accumulated full (question, answer) transcript;
- `summary` -- a running model-written summary of the prior steps (one extra model call per step);
- `roles` -- a staged role/system-prompt sequence (librarian -> analyst -> answerer) built from
  the prompt-system role templates, plus the accumulated transcript. First step is the librarian,
  last is the answerer, middle steps are analysts; a 2-step chain is librarian -> answerer.

```bash
llb bench-chain-context --chains <chains.jsonl> --corpus <corpus-dir> \
  --model <model> --backend <backend> --policies fresh,history,summary,roles
make bench-chain-context CHAIN_CONTEXT_MODEL=<model> CHAIN_CONTEXT_BACKEND=<backend>
```

Retrieval reaches the store through the injectable `Retriever` seam (`retrieve(question, k)`), and
the model through the same injectable `complete` (prompt -> raw text) every category uses, so the
exact context assembled per policy per step is unit-tested over a FAKE endpoint with no GPU
(`tests/llb/bench/test_chain_context.py`). Each step's answer is scored objectively against its
reference answer (`scoring.correctness` token-F1); the headline is FINAL-answer correctness per
chain (does the chain end right), with per-step correctness recorded alongside, both with bootstrap
CIs. Each policy persists its OWN run bundle under `$DATA_DIR/chain-context/<timestamp>/` tagged
with the policy (mirroring the per-harness agentic bundles); provenance records the policy, the
`prompt_system_ids` (the role/instruction template ids), and the `chain_set_digest`. Verified-data
stamping (`--data-verified` + `--verification-ref`) follows the same category-suite gate as the
other benchmarks. The board loader ranks all policies together, and `llb recommend` gains a
"Context policy" section naming the best policy per model when bundles exist.

CUDA evidence (2026-07-11, RTX 4060 Ti 16 GB): the committed 20-chain fixture (40 steps) run
through `MamayLM-Gemma-3-12B-IT-v2.0` on Ollama, all four policies in one ~11 min invocation,
reliability 1.000 for each. Final-answer correctness ranked `roles` **0.789** [0.635, 0.915] >
`history` 0.625 > `summary` 0.534 > `fresh` 0.431: the naive no-carryover baseline is worst, any
memory of the prior steps helps, and the staged librarian -> analyst -> answerer sequence wins with
its CI resolved above the rest. Run bundles under `.data/chain-context/20260711T1938*` (one per
policy). The discriminating spread is the whole reason to measure the policy rather than assume one.

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
- `knowledge_tree.py`: ontology/graph loading, deterministic community ordering, and strict
  depth/token-budget rendering;
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

Knowledge-tree generation is opt-in and consumes an existing ontology draft bundle, a persisted
graph store, or both. It adds the corpus vocabulary, size-ranked entity communities, and optional
community summaries as a system-prompt block. Every ordinary prompt candidate remains as a
no-tree control; tree candidates record their exact control id plus the source digest, requested
depth, requested/effective token budget, and rendered token count. Tree tokens consume the same
overall prompt allowance as anthology/metadata/mapping context.

```bash
make prompt-system-prepare \
  PROMPT_SYSTEM_CORPUS=<corpus-dir> \
  PROMPT_SYSTEM_GRAPH_DIR=<graph-store> \
  PROMPT_SYSTEM_TREE_DEPTHS=1,2,3 \
  PROMPT_SYSTEM_TREE_BUDGETS=128,256
make run-eval PROMPT_SYSTEM_ID=<control-id> PROMPT_PACKAGE=<review-dir>
make run-eval PROMPT_SYSTEM_ID=<tree-id> PROMPT_PACKAGE=<review-dir>
make prompt-system-compare MODEL=<model> PROMPT_SYSTEM_LANE=rag
```

`PROMPT_SYSTEM_ONTOLOGY_BUNDLE=<draft-bundle>` is the alternative source knob. When only that
source is supplied, preparation reuses the existing graph builder and deterministic community
detector in memory; it does not run extraction. `prompt-system-compare` ranks all evaluated ids and
prints the best evaluated tree against its matched control, including the paired objective delta,
bootstrap CI when per-case series align, and `helps`, `hurts`, or `inconclusive` conclusion.

Local evidence on 2026-07-18 used the committed IP-regulation final split (n=4) and
`MamayLM-Gemma-3-12B-IT-v2.0` Q4_K_M on the RTX 4060 Ti 16 GB host. Control `2af73c060984`
scored 0.685; depth-2, 256-token tree `e6176121770b` scored 0.671. The paired delta was -0.0145
with CI [-0.0339, +0.0000], so the comparison is inconclusive and does not support pinning the
tree on this tiny fixture. Artifacts are under `.data/knowledge-tree-ab/`: prompt candidates in
`prompt-system/evidence/`, control run
`run-eval/20260718T170430.216852Z-20da112f09c7/`, and tree run
`run-eval/20260718T170519.951424Z-cdfe1dc48c62/`. Retrieval held recall@5=1.000 and MRR=1.000
for both. `make ci` passed with 1,477 tests and 42 slow tests deselected.

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
  `--trainer unsloth` selects the Unsloth-accelerated path (`unsloth_train_adapter`): same SFT
  loop, dataset contract, and manifest, but the base model is loaded and LoRA-wrapped through
  `FastLanguageModel` for roughly 2x faster single-GPU training. Unsloth is intentionally not a
  project extra (it pins a hardware-matched torch/triton stack, same policy as marker); install
  `unsloth` manually in the CUDA training environment. Unknown `--trainer` values exit with the
  accepted list; the manifest records the concrete trainer that ran (`peft-trl`, `unsloth`, or
  `fake`), never `auto`. Covered by dispatch/missing-dependency tests in
  `tests/llb/finetune/test_finetune.py`.
- `src/llb/finetune/guard.py` enforces the contamination invariant before `run-eval` launches a
  backend: adapter manifests may contain only tuning-split training ids, may not intersect
  calibration/final eval ids, and a tuned model cannot judge itself.
- `src/llb/finetune/loop.py` orchestrates base final eval, per-round tuning eval, miss analysis,
  dataset export, adapter training, adapter final eval, stop/accept logic, `state.json`, and
  `report.md`.
- `src/llb/finetune/campaign/run.py` schedules the loop ingredients across a `--models` roster with
  planner skip reasons, a shared campaign SFT export, per-model preference exports, VRAM reclaim
  between roster entries, `campaign.progress.jsonl` resume, and a tunability `report.md`.
- `src/llb/finetune/distill/run.py` runs local text-level distillation: a teacher answers verified
  tuning items through the normal RAG backend seam, deterministic correctness gates decide which
  answers become SFT targets, the same student is trained on teacher targets and reference targets,
  and the report compares the two adapters over the same held-out items.
- `src/llb/finetune/registry/`, `lifecycle.py`, and `serving.py` make adapters first-class,
  traceable artifacts (see [Adapter Registry And Lifecycle](#adapter-registry-and-lifecycle)).
- `src/llb/finetune/hparam_search/search.py` searches the LoRA space per model and feeds the winning
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
llb distill --teacher <teacher> --student <student> --backend vllm \
  --goldset <goldset> --corpus <corpus-dir> --gate 0.8
make self-improve MODEL=<model> BACKEND=vllm GOLDSET=<goldset> ROUNDS=2
make finetune-campaign MODELS=<m1,m2> BACKEND=vllm GOLDSET=<goldset> CORPUS=<corpus-dir>
make distill TEACHER=<teacher> STUDENT=<student> BACKEND=vllm GOLDSET=<goldset>
```

Artifacts live under `$DATA_DIR/self-improve/<timestamp>/round-<n>/` for campaign state and under
`$DATA_DIR/run-eval/` for canonical board bundles. Round directories carry `dataset/`, `adapter/`,
`run` and `run-final` pointers, plus per-round reports.

Multi-model campaign artifacts live under `$DATA_DIR/finetune-campaign/<timestamp>/`. The campaign
root contains `shared-dataset/dataset_manifest.json`, `campaign.progress.jsonl`, `report.md`, and
one directory per roster model. Each model directory records base-final and per-round tuning/final
run pointers, miss analysis, a per-model preference dataset, and the final adapter. Resume replays
`campaign.progress.jsonl` and does not retrain a completed roster entry.

Distillation artifacts live under `$DATA_DIR/distill/<timestamp>/`: `teacher_outputs.jsonl`,
`dataset/` for accepted teacher-answer SFT targets, `reference_dataset/` for the same item ids with
reference-answer targets, `adapter/`, `reference_adapter/`, `comparison/`, `distill_manifest.json`,
and `report.md`. The distillation manifest and accepted dataset manifest record the teacher model,
student model, gate threshold, accepted item ids, and per-item gate scores. The distilled adapter is
registered with its paired comparison delta; the reference adapter stays local comparison evidence.

Adapter-backed `run-eval` rows are labeled `<base>+adapter-<digest>` in manifests and board loaders.
`llb recommend` appends a self-improvement section when a campaign `state.json` exists and a
fine-tune campaign section when `$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` exists. The
campaign section ranks completed models by final-split delta, then shorter training wall-clock, then
lower peak VRAM; skipped models remain visible with the planner reason.

Tests:

```bash
uv run pytest tests/llb/finetune/test_finetune.py \
  tests/llb/finetune/test_distill.py \
  tests/llb/finetune/test_adapter_registry.py \
  tests/llb/board/test_recommend.py
```

The campaign implementation is covered by fake eval/trainer/planner tests for scheduling order,
planner skip reasons, shared dataset digest reuse, JSONL resume, and report ranking.
The distillation implementation is covered by fake teacher/trainer/comparison tests for gate
exclusion, tuning-only teacher generation, identity and judge-teacher refusals, report math,
registry registration, and contamination-guard compatibility.

## Hyperparameter Search

`src/llb/finetune/hparam_search/search.py` searches the LoRA configuration space for one model with a
bounded budget, so fine-tuning stops guessing rank, alpha, learning rate, epochs, target modules,
or batch geometry.

The search space also covers the effective batch axis (finetune-hparams-effective-batch-axis):
`per_device_train_batch_size` x `gradient_accumulation_steps` ride ONE `batch_geometry`
categorical (`1x4` the trainer default, `1x8`, `2x4`, `2x8`) rather than two independent draws --
independent draws would mostly differ only in a VRAM/wall-clock trade at the same effective batch,
wasting budget on gradient-equivalent points -- and `max_length` (512/1024/2048) is sampled beside
it. Effective batch size interacts strongly with the learning rate, so the recorded best config is
internally consistent: `hparams_manifest.json` carries the batch geometry the learning rate was
chosen under, and an operator changing the batch size knows they left the searched optimum. The
sampled record always satisfies `effective_batch_size == per_device * grad_accum` (unit-tested).

Dependency contract: the `[finetune]` and `[dev]` extras include Optuna. GitHub CI installs
`.[dev]`, so pure hparam slice/guard tests plus small fake-trainer manifest integrations stay in
the lightweight `make ci` suite without pulling the CUDA training stack. Multi-trial hparam
resume/prune simulations and multi-entry fine-tune campaign ranking/resume simulations are marked
`slow`; they run in the full local `make test` suite.

```bash
llb finetune-hparams --model <m> --dataset <tuning-dataset> --backend vllm \
  --goldset <goldset> --max-trials 8 [--max-hours 2] [--seed 13] [--dev-fraction 0.25] \
  [--stratify-by-base-score <scored-base-run-dir>]
llb finetune-hparams ... --resume <study-dir>
make finetune-hparams MODEL=<m> DATASET=<dir> GOLDSET=<g> MAX_TRIALS=8 \
  HPARAMS_STRATIFY_RUN=<scored-base-run-dir>
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
- `--stratify-by-base-score <scored-base-run-dir>` (make: `HPARAMS_STRATIFY_RUN=`) replaces the
  uniform draw with a stratified one: `carve_stratified_dev_slice` buckets the tuning items by
  the base model's per-item `objective_score` from the given run bundle's `scores.jsonl`
  (`high` >= 0.5, `low` > 0, `zero`, `unscored`) and draws the dev slice proportionally per
  bucket with a floor of one, answerable buckets first -- so a small dev slice always carries
  items the base model can answer and the trial objective can discriminate (the failure the
  first CUDA search hit: a uniform 3-item slice with one answerable item tied every trial at
  0.0000). A population the base model scores 0.0 everywhere is REFUSED -- no slice can rank
  trials against a constant objective. The same disjointness and seeded determinism hold, and
  `hparams_manifest.json` records an additive `dev_slice.strata` block (the source run plus
  per-bucket population/dev counts and mean base score). The default without the flag stays the
  uniform slice. Committed fixture: `samples/finetune/base-score-run/scores.jsonl` (12 items, 3
  answerable), used by `tests/llb/finetune/test_finetune_hparams.py` to prove the stratified
  slice holds an answerable item at every seed where the uniform slice misses.
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

Pre-run infeasibility prune (finetune-hparams-infeasible-point-prune): with
`--vram-headroom-mib <n>` (make: `HPARAMS_VRAM_HEADROOM=`) -- the VRAM left beside the base model
during training on the host -- a trial whose estimated adapter TRAINING footprint exceeds the
headroom is pruned BEFORE `trainer_fn` runs, so a bounded budget never pays a full fine-tune for
a known-infeasible point. The estimate is `rank x targeted modules x layers x 2 (hidden x r)
matrices x 16 bytes/param` (bf16 weight + grad, fp32 Adam moments + master copy;
`estimated_adapter_train_mib`), with hidden size / layer count read from the model's cached HF
config (`model_arch` overrides it programmatically). Every trial row in `hparams_manifest.json`
and `trials.jsonl` carries the additive `estimated_adapter_mib`, and the prune reason names the
estimated footprint against the headroom. The estimate is deliberately coarse: it complements
the measured-OOM prune (which always stays in place), never replaces it. Without a headroom the
pre-run prune is off.

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

Tests: `tests/llb/finetune/test_finetune_hparams.py` covers dev-slice disjointness and
determinism, both guard refusals, the no-protected-id-in-any-trial invariant, manifest writing,
the manifest surviving a failed trial, subset digests, and the trainer consuming a recorded best
config through a self-improvement round in the lightweight suite. Slow coverage keeps the seeded
full trial table, budget abort plus resume without repeated trials, OOM and infeasible-point
pruning, and effective batch sampling.

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

- The dev slice can use a seeded plain split or base-score stratification. Supplying
  `--stratify-by-base-score <run>` represents every non-empty score bucket and guarantees
  answerable items; an all-zero base run is rejected because it cannot discriminate trials.
- Trial 5 lands exactly on the base objective `0.2056`: a tuned adapter is not automatically better
  than no adapter, and the search records that honestly.

### Effective-batch-axis evidence on the 16 GB RTX 4060 Ti host

The widened-space acceptance run (2026-07-10, finetune-hparams-effective-batch-axis): a 6-trial
search for `google/gemma-3-1b-it` over the `ua_squad_postedited_v1` tuning split (82 items ->
62 train / 20 dev, `seed=13`; full-split base tuning objective `0.3050`), study
`.data/finetune-hparams/google-gemma-3-1b-it/20260710T121020*/hparams_manifest.json`, ~2 min per
trial end to end (QLoRA fine-tune + vLLM LoRA dev eval):

| trial | dev objective | geometry | eff. batch | max_length | rank | lr | preset |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | **0.3262** | 2x8 | 16 | 2048 | 16 | 2.63e-05 | attn_mlp |
| 2 | 0.3151 | 2x8 | 16 | 2048 | 4 | 2.53e-05 | attn |
| 4 | 0.2986 | 2x8 | 16 | 2048 | 64 | 1.53e-04 | qv |
| 0 | 0.2865 | 2x4 | 8 | 2048 | 64 | 2.73e-05 | attn_mlp |
| 5 | 0.2692 | 2x4 | 8 | 2048 | 8 | 3.55e-04 | attn_mlp |
| 3 | 0.2427 | 1x8 | 8 | 2048 | 64 | 9.08e-05 | attn_mlp |

What the run demonstrates: the learning-rate x effective-batch interaction is measurable, not
theoretical -- trials 0 and 1 sample a near-identical learning rate (2.7e-05 vs 2.6e-05) and the
effective-batch-16 point beats the effective-batch-8 point by **+0.040** dev objective; the three
top trials all ride the largest geometry (`2x8`). The honest caveats: the trainer-default `1x4`
geometry was never drawn in this 6-trial budget (TPE explored the wider geometries), so the
comparison to the pinned default is indirect (via `2x4`/`1x8` at effective batch 8, both of which
lose), and a 20-item dev slice carries wide uncertainty per point. The operational win stands
regardless of ranking noise: `hparams_manifest.json` records the batch geometry every
learning rate was chosen under, so the recorded best config
(`2x8`, lr 2.63e-05, rank 16, `attn_mlp`, `max_length` 2048) is self-consistent and
`trainer_defaults` feeds all of it -- geometry included -- to later rounds.

## Compressed-QAT Trainability (finetune-compat)

`src/llb/finetune/compat.py` (compressed-qat-adapter-support) answers "can this checkpoint take a
LoRA adapter on this host?" BEFORE a campaign pays for a base eval or a training run. Compressed
QAT checkpoints (`*-qat-w4a16-ct` and friends) serve well on vLLM, but PEFT can only inject LoRA
into layer types it has a dispatch for (full-precision `Linear`, bitsandbytes 4/8-bit, GPTQ, AWQ,
EETQ, HQQ) -- a `compressed-tensors` checkpoint's `CompressedLinear` layers cannot take adapters.

Two stages, both pure over injectable seams (`tests/llb/finetune/test_finetune_compat.py` runs with fake
modules and configs, no torch):

- Config introspection (`inspect_quantization` + `assess_quantization`): classifies the
  checkpoint's native `quantization_config.quant_method` against PEFT's dispatch table -- no
  weights, no CUDA. `compressed-tensors` is a deterministic not-trainable verdict with the exact
  blocker plus the documented fallback (train on the uncompressed base and serve merged/quantized,
  or take the bitsandbytes path); a PEFT-dispatched scheme names its injection strategy; an
  unrecognized scheme stays `unknown` so the heavy probe decides.
- The heavy probe (`probe_trainability`, `llb finetune-compat --model <m>`): loads the model,
  scans its ACTUAL linear module classes, selects per-architecture target modules from the modules
  that exist (`select_target_modules` grounds the choice in the model's own names -- llama-style
  `q_proj`, falcon `query_key_value`, gpt2 `c_attn`, with a most-frequent-suffix fallback --
  instead of assuming llama naming), attaches a rank-4 LoRA, and runs one forward/backward
  micro-step. Any failure becomes the recorded blocker, never a crash. Reports land under
  `$DATA_DIR/finetune-compat/<model>/<timestamp>/compat_report.json`; `--config-only` stops after
  stage 1.

Campaign integration: `run_finetune_campaign` runs a config-only compat probe (injectable
`compat_fn`; the default reads only locally-cached configs, so Ollama tags and never-downloaded
models return `unknown` without touching the network) after the memory planner and BEFORE the
base eval -- a positive not-trainable verdict skips the entry into `campaign.progress.jsonl` and
`report.md` with the exact blocker; an unknown verdict never false-skips.

CUDA evidence (2026-07-10, RTX 4060 Ti 16 GB):

- `google/gemma-4-E4B-it-qat-w4a16-ct` -> **not-trainable** at the config stage
  (`quant_method 'compressed-tensors' has no PEFT LoRA dispatch`); the skip fires before any
  weights load. `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` hits the same verdict -- its "AWQ"
  is AWQ-inside-compressed-tensors, which the config stage classifies correctly.
- `Qwen/Qwen3-4B-FP8` -> config stage says `unknown` (`quant_method 'fp8'`), the heavy probe
  loads it and the module scan finds `FP8Linear` (no PEFT dispatch) -> **not-trainable** with
  that exact blocker -- the load-time detection path proven on a real checkpoint.
- Reports: `.data/finetune-compat/google-gemma-4-E4B-it-qat-w4a16-ct/*/compat_report.json`,
  `.data/finetune-compat/Qwen-Qwen3-4B-FP8/*/compat_report.json`.

## Adapter Registry And Lifecycle

Adapters are first-class artifacts, not loose directories. `$DATA_DIR/adapters/registry.jsonl` is an
append-only event log (`register` / `merge` / `delete`) folded into the current entry set on read, so
a partial write can never lose earlier history. The entry id IS the `adapter_digest`, so it can never
be reassigned to different weights.

Modules:

- `src/llb/finetune/registry/`: `model.py` owns `AdapterEntry`; `io.py` folds the event log;
  `register.py` performs idempotent registration and lifecycle writes; `resolve.py` handles id /
  unique prefix / label / directory lookup; `staleness.py` compares benchmark fingerprints; and
  `rows.py` renders CLI/board rows;
- `src/llb/finetune/lifecycle.py`: run-bundle citation scan, supersession, and garbage collection;
- `src/llb/finetune/serving/model.py`: immutable serve and merge contracts;
- `src/llb/finetune/serving/run.py`: serve-plan construction and launcher lifecycle;
- `src/llb/finetune/serving/merge.py`: cached merges, GGUF conversion, and Ollama Modelfiles;
- `src/llb/finetune/serving/launcher.py`: backend-specific launcher construction.

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

A third axis covers the RAG store (adapter-staleness-retrieval-fingerprint): an adapter is
trained on retrieved CONTEXT, so re-embedding or rechunking the same corpus invalidates its
training contexts while `corpus_fingerprint` stays unchanged. Registration records a
`retrieval_fingerprint` (embedder, chunk strategy/size/overlap, retrieval mode) read from the
store's `store_meta.json` (`register_adapter --index-dir` on the CLI; `self-improve` /
`finetune-campaign` rounds record the config's index dir automatically), and `staleness()`
compares it per knob against the store's present meta -- a rebuilt store flips the entry `stale`
with the changed knob named in the reason (for example
`retrieval embedding_model changed since training (a -> b)`). An adapter registered without an
index directory reads `unknown` on the retrieval axis (reason `retrieval fingerprint unavailable`),
never `current`.

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
with one generation -- an empty completion FAILS the probe (a served-but-mute endpoint is not
serving) -- and then holds it in the foreground until Ctrl-C; there is no serving daemon.

Chat-template preservation is required because llama.cpp's server applies
the `tokenizer.chat_template` GGUF metadata natively, but **Ollama ignores it** when a model is
created from a bare `FROM <gguf>` Modelfile -- the tag serves raw completions and a merged
instruct model degrades to gibberish or empty chat answers. `modelfile_text` therefore reads the
merged tokenizer's `chat_template.jinja`, detects the template family by its unambiguous marker
(ChatML
`<|im_start|>`, Gemma `<start_of_turn>`, Llama 3 `<|start_header_id|>`), and writes the
equivalent Go `TEMPLATE` plus its `PARAMETER stop` tokens into the Modelfile; an unrecognized
template stays a bare FROM with a loud warning naming the fix. Family detection, the bare-FROM
fallback, and the empty-probe failure are unit-tested with fixtures.

Pristine tokenizer files are copied from the base model because a LoRA never changes the tokenizer,
while `AutoTokenizer.save_pretrained` can be lossy for GGUF conversion: it drops the sentencepiece
`tokenizer.model` (the converter's GPT-2-style fallback then asserts on vocabularies whose added
tokens sit past `config.vocab_size`) and rewrites `tokenizer_config.json` so the control-token
markings are lost: `<start_of_turn>`/`<end_of_turn>` exported as NORMAL instead of CONTROL token
types, Ollama then never matched the template's turn markers as specials, and the merged Gemma
answered every non-trivial prompt with an immediate `<end_of_turn>` (final-split objective 0.199
vs 0.410 served properly -- while the SAME safetensors answered correctly in transformers).
`copy_base_tokenizer_assets` overwrites the resaved files with the base repo's originals
(`tokenizer.model`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`),
best-effort per file so repos without a given file (Qwen has no sentencepiece model) keep the
resaved copy that already converts fine. Unit-tested with an injected downloader.

### Garbage collection

An adapter is superseded once a newer adapter exists for the same base model, ordered by
`(created_at, log sequence)`. `created_at` has second resolution, so two fast rounds tie; the
append-log position breaks the tie exactly. Only superseded adapters are GC candidates, and GC
refuses any that a durable artifact still cites. The citation scan covers published run bundles
(`$DATA_DIR/run-eval/*/manifest.json`, matched by recorded digest or by served `adapter_path`)
AND the orchestrator journals that also link adapter directories: self-improvement
`$DATA_DIR/self-improve/*/state.json` (`rounds[].adapter_dir`) and campaign
`$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` (`entry.adapter_dir`), both resolved
through the registry's adapter-dir index the way the served-path match is. Every citation
carries its artifact kind (`run-bundle` / `self-improve-state` / `campaign-journal`) in
`GcDecision.cited_by`, the refusal reason names the citing artifact(s), and `gc_rows` exposes
the kinds in a `cited_kinds` column. `--force` overrides the citation refusal but never the
safety rule that GC only deletes directories inside `$DATA_DIR`. Deletions append a `delete`
tombstone.

### Committed fixtures

- `samples/finetune/registry/registry.jsonl`: a stale entry (with a folded `merge` event) and a
  poisoned-digest entry, both pointing at adapter dirs outside `$DATA_DIR`;
- `samples/finetune/gc-journals/`: a data-dir-shaped fixture whose campaign journal cites the
  committed stale adapter, proving a journal-only citation blocks an unforced GC;
- `samples/finetune/stale-adapter/`: recorded digests that no longer match
  `samples/goldsets/ip_regulation_uk/`;
- `samples/finetune/laundered-adapter/`: an `adapter_manifest.json` that CLAIMS a clean tuning-only
  training set while the registry records the `final`-split ids it was really trained on;
- `samples/finetune/poisoned-adapter/`: the simpler case where the manifest itself declares the
  protected split, refused even when unregistered.

`tests/llb/finetune/test_adapter_registry.py` covers registry round-trip and idempotence, the
staleness flip when the goldset digest changes, the `unknown` verdict, guard resolution through
the registry, serving smoke over a fake launcher for all three backends, merge-event recording and
merge caching, GC citation refusal plus `--force` (run-bundle, self-improve-state, and
campaign-journal citations, including the committed journal fixture), the same-second supersession
tie, the outside-`$DATA_DIR` safety rule, and board drop/stamp behavior.

Merge-serving CUDA evidence (2026-07-10, RTX 4060 Ti 16 GB, adapter-merge-serving-cuda-evidence;
the first time the real merge lane ran end to end):

- Adapter: `ea848f7e160e` (`Qwen/Qwen2.5-0.5B-Instruct`, one `self-improve` round over the
  `ua_squad_postedited_v1` tuning split, registered; campaign
  `.data/self-improve/merge-evidence-qwen05b/`).
- Both GGUF backends merged and answered the smoke probe: PEFT merge + `convert_hf_to_gguf.py`
  (f16) + launch + probe in **~15 s wall-clock per backend** for the 0.5B model, GGUF size
  **949 MB** (vs ~1 GB safetensors); converter accepted the Qwen2 architecture without complaint.
- Three-way final-split objective (n=82, same goldset/store/seed):
  base (vLLM) **0.2880** [0.204, 0.370]; vLLM LoRA row **0.3272** [0.239, 0.422]; merged tag on
  ollama **0.3119** [0.218, 0.402] -- inside the LoRA row's CI and above the base point estimate,
  so the merged artifact answers as the ADAPTER, not the base model. Run bundles:
  `.data/run-eval/20260710T075222*` (base), `...075718*` (LoRA), `...081359*` (merged, fixed
  template).
- The Ollama Modelfile carries the explicit chat template described above, while the smoke probe
  rejects an empty completion. The `finetune` extra includes both the converter's `gguf` import and
  the trainer's `bitsandbytes` dependency so failures occur during dependency validation.

Second cohort model, `google/gemma-3-1b-it` (2026-07-10, same host; adapter `db80e8440b7d` from
one `self-improve` round trained with the effective-batch search's best config, campaign
`.data/self-improve/merge-evidence-gemma-3-1b/`):

- Merge cost: ~24 s (Ollama) / ~18 s (llama.cpp) wall-clock per backend, 1.9 GB f16 GGUF. The
  converter uses the base repository's pristine tokenizer files to preserve the sentencepiece
  vocabulary and control-token types.
- Three-way final-split objective (n=82): base (vLLM) **0.3872** [0.299, 0.480]; vLLM LoRA row
  **0.4103** [0.326, 0.498]; merged tag on ollama **0.3427** [0.260, 0.428] -- inside the LoRA
  row's CI, so the merge passes the fidelity gate, with the honest caveat that the point
  estimate sits 0.068 below the LoRA row (unresolved at n=82, and partly a cross-backend
  comparison: the merged row is f16-GGUF-on-ollama while both reference rows are
  safetensors-on-vLLM). Run bundles: `.data/run-eval/20260710T122520*` (base),
  `...122821*` (LoRA), `...125503*` (merged).

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
