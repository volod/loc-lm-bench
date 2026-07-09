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
vLLM serving passes the adapter through `--enable-lora --lora-modules`; Ollama and llama.cpp require
a merged model artifact before they can serve the adapter. `llb recommend` appends a
self-improvement section when a campaign `state.json` exists and a fine-tune campaign section when
`$DATA_DIR/finetune-campaign/*/campaign.progress.jsonl` exists. The campaign section ranks completed
models by final-split delta, then shorter training wall-clock, then lower peak VRAM; skipped models
remain visible with the planner reason.

Tests:

```bash
uv run pytest tests/test_finetune.py tests/test_recommend.py
```

The committed poisoned manifest fixture at `samples/finetune/poisoned-adapter/adapter_manifest.json`
is used to keep the protected-split refusal path easy to exercise.

The campaign implementation is covered by fake eval/trainer/planner tests for scheduling order,
planner skip reasons, shared dataset digest reuse, JSONL resume, and report ranking.

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
