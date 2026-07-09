# Self-Improvement Loop

Use this workflow to adapt a local model to the benchmark corpus while preserving the final split
as held-out evidence.

## Install

Control-plane smoke runs need only the normal development environment. Real adapter training on a
CUDA host needs the fine-tuning extra:

```bash
uv pip install -e ".[finetune]"
```

## Export A Dataset

Start from a finalized tuning-split run bundle. If miss analysis exists, pass its `misses.jsonl` so
the export targets the measured miss set and builds preference pairs from wrong answers.

```bash
llb export-finetune-set \
  --run-dir <tuning-run> \
  --goldset <goldset> \
  --misses <misses-jsonl> \
  --out <dataset-dir>
```

The output contains:

- `sft.jsonl`: chat messages in the same shape as `run-eval`, with reference answers;
- `dpo.jsonl`: optional chosen/reference versus rejected/model-answer pairs;
- `dataset_manifest.json`: source run, item ids, split counts, and dataset digest.

Only verified tuning-split items are exported.

## Train An Adapter

For CI or a dry run:

```bash
llb finetune-adapter --dataset <dataset-dir> --model <model> --trainer fake
```

For CUDA training:

```bash
llb finetune-adapter --dataset <dataset-dir> --model <model>
```

The adapter directory contains `adapter_manifest.json` with base model, dataset digest, item ids,
hyperparameters, seed, loss curve, and adapter digest.

## Run The Full Loop

```bash
llb self-improve --model <model> --backend vllm --goldset <goldset> --rounds 2
make self-improve MODEL=<model> BACKEND=vllm GOLDSET=<goldset> ROUNDS=2
```

The orchestrator runs:

1. base final eval;
2. tuning eval;
3. miss analysis;
4. dataset export;
5. adapter training;
6. adapter final eval;
7. accept/reject reporting.

Campaign state lands under `$DATA_DIR/self-improve/<timestamp>/`; canonical eval bundles remain
under `$DATA_DIR/run-eval/` so the board and `llb recommend` see the tuned row.

## Run A Multi-Model Campaign

Use the roster campaign when the question is which local model is best after adaptation, not just
which base model wins before tuning.

```bash
llb finetune-campaign \
  --models <model-a>,<model-b> \
  --backend vllm \
  --goldset <goldset> \
  --corpus <corpus-dir> \
  --rounds 1

make finetune-campaign MODELS=<model-a>,<model-b> BACKEND=vllm GOLDSET=<goldset> CORPUS=<corpus-dir>
```

Optional flags:

- `--manifest <models-yaml>` gives the feasibility planner model metadata;
- `--resume <campaign-dir>` replays `campaign.progress.jsonl` and skips completed roster entries;
- `--trainer fake` runs the control plane without CUDA training dependencies;
- `--limit <n>` caps each eval split for smoke runs.

Campaign state lands under `$DATA_DIR/finetune-campaign/<timestamp>/`:

- `shared-dataset/`: one SFT export reused across feasible roster entries;
- `<model>/round-<n>/preference-dataset/`: per-model preference export from that model's misses;
- `<model>/round-<n>/adapter/`: trained adapter manifest and weights or fake marker;
- `campaign.progress.jsonl`: append-only resume journal;
- `report.md`: tunability ranking by final-split gain, training wall-clock, and peak VRAM.

`llb recommend` appends the latest campaign ranking when a campaign progress journal exists.
Planner-rejected models stay in the report with their skip reason.

## Guardrails

`run-eval` refuses adapter-backed runs when the adapter manifest records calibration/final split
training data or any protected eval id overlap. It also refuses a tuned model judging itself.

Adapter serving is currently direct for vLLM LoRA modules. For Ollama or llama.cpp, merge the
adapter into a model artifact first, then evaluate that merged artifact as the model.
