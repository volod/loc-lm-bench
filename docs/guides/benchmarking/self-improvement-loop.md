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

## Track Adapters In The Registry

Every successful self-improvement round and campaign round registers its adapter in the append-only
log at `$DATA_DIR/adapters/registry.jsonl`. The registry id is the adapter digest, so an entry can
never be reassigned to different weights.

A bare `llb finetune-adapter` does NOT register, and an unregistered adapter never renders on the
board. Register it by hand:

```bash
llb register-adapter --adapter-dir <adapter-dir> --goldset <goldset> --corpus <corpus-dir>
make register-adapter ADAPTER_DIR=<adapter-dir> GOLDSET=<goldset> CORPUS=<corpus-dir>
```

Then list what the registry knows:

```bash
llb list-adapters
llb list-adapters --json
make list-adapters
```

Each row carries the base model, dataset digest, source run, eval evidence, and a staleness verdict:

- `current`: the goldset and corpus digests recorded at training time still match what is on disk;
- `stale`: one of them changed, so the recorded eval evidence no longer describes this benchmark;
- `unknown`: a digest was never recorded, or the goldset/corpus it names is gone. This is never
  reported as `current` -- absence of evidence is not evidence.

A stale adapter is stamped, never silently ignored. Its board row renders as
`<base>+adapter-<digest> [stale]`, and an adapter that is not registered at all does not render on
the board or in `llb recommend` -- a tuned number nobody can trace is not a comparable number.

Retrain a stale adapter to refresh it; the registry never retrains on your behalf.

## Serve A Registered Adapter

```bash
llb serve-adapter --adapter <adapter-id> --backend vllm
llb serve-adapter --adapter <adapter-id> --backend ollama --smoke
make serve-adapter ADAPTER=<adapter-id> BACKEND=llamacpp
```

`--adapter` accepts a full adapter id, a unique id prefix, or the adapter label. The command probes
the endpoint with one generation, then holds the backend open in the foreground until Ctrl-C;
`--smoke` exits right after the probe instead. There is no serving daemon.

vLLM loads the LoRA directly (`--enable-lora --lora-modules`). Ollama and llama.cpp serve whole
model artifacts, so the adapter is first merged into its base weights and converted to GGUF. That
merge is cached under `$DATA_DIR/adapters/merged/<short-id>/<backend>/` and recorded as a registry
`merge` event, so the merged artifact stays traceable to the adapter digest that produced it. The
GGUF conversion needs the llama.cpp checkout (`make build-llamacpp`) and the `[finetune]` extra.

## Evaluate Through The Registry

```bash
llb run-eval --adapter <adapter-id> --model <base-model> --backend vllm --goldset <goldset>
```

Passing `--adapter` resolves the adapter through the registry, so the contamination guard reads the
digests the registry *recorded* rather than the `adapter_manifest.json` sitting beside the weights.

## Collect Superseded Adapters

An adapter is superseded once a newer adapter exists for the same base model. Only superseded
adapters are GC candidates:

```bash
llb gc-adapters --dry-run
llb gc-adapters
llb gc-adapters --force
make gc-adapters GC_DRY_RUN=1
```

GC refuses to delete an adapter that any published run bundle still cites -- deleting one would
strand a board row that can no longer be reproduced. `--force` overrides that refusal. It does not
override the safety rule that GC only ever deletes directories inside `$DATA_DIR`, so committed
fixtures and hand-placed adapters are never touched. Deletions append a `delete` tombstone; the
original `register` event stays in history.

## Guardrails

`run-eval` refuses adapter-backed runs when the recorded training provenance includes
calibration/final split data or overlaps any protected eval id, and it refuses a tuned model judging
itself.

Provenance is read from the registry when the adapter is registered, and from
`adapter_manifest.json` only when it is not -- a freshly trained adapter registers after its first
eval. This matters: a manifest beside the weights is operator-writable, so a hand-edited one could
otherwise launder a final-split adapter past the gate. See
[`samples/finetune/laundered-adapter/`](../../../samples/finetune/laundered-adapter/) for the
fixture that pins this behavior.
