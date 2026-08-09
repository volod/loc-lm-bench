# Prompt Systems And The Self-Improvement Loop

Part of the [Extended workflows](../extended-workflows.md) area of the
[current implementation index](../../current.md).

## Prompt-System Packages

`src/llb/prompt_system/` builds reviewable RAG prompt-system candidates from a corpus. The package
is deterministic and manifest-addressable so prompt changes become explicit experiment variables.

Important modules:

- `corpus.py`: reads `.md`/`.txt`, keeps exact spans, selects anthology passages, builds metadata;
- `budget.py`: token-budget planning and section trimming;
- `template.py`: prompt fields and `PromptPackage.apply`;
- `tuning.py`: candidate grid and deduplication;
- `knowledge_tree_source.py`: ontology/graph loading and source identity;
- `knowledge_tree_render.py`: deterministic community ordering and strict depth/token-budget
  rendering;
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
- `src/llb/finetune/trainer.py` selects and orchestrates LoRA/QLoRA trainer backends, while
  `training_runtime.py` owns dataset/tokenizer/model preparation and the shared TRL loop.
  `--trainer fake` writes deterministic CI artifacts; the real path lazy-imports PEFT, TRL,
  Transformers, and Datasets from the `[finetune]` extra and saves an adapter plus
  `adapter_manifest.json`.
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
  traceable artifacts (see [Adapter Registry And Lifecycle](adapter-registry.md#adapter-registry-and-lifecycle)).
- `src/llb/finetune/hparam_search/search.py` searches the LoRA space per model and feeds the winning
  config back as the trainer's defaults (see
  [Hyperparameter Search](finetuning-search.md#hyperparameter-search)).
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
