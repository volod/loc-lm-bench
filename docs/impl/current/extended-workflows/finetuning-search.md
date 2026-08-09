# Fine-Tuning Search And Trainability

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

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
