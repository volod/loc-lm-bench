# RAG Prompt-System Lane

This guide shows how to use prompt-system packages with the baseline RAG `run-eval` lane, using
the committed IP regulation sample corpus and local `gemma4:e4b`.

The workflow uses the selected-pipeline caution from
[arXiv:2406.18902](https://arxiv.org/abs/2406.18902): tune on one split, report on a held-out split,
and keep the final decision separate from the selected prompt.

## At a glance

```text
1. build + gate the index    build-index -> validate-retrieval   [gate: retrieval passes first]
2. generate candidates       prompt-system-prepare               [reviewable prompt package]
3. tune on the tuning split  run-eval --split tuning --prompt-system <id>
4. pin the winner [HUMAN]    prompt-system-review --action pin --id <id>
5. verify on the final split run-eval --split final [--prompt-system <id>]
6. compare                   prompt-system-compare --lane rag --model <model>
```

The rule that gates everything: **never promote a prompt that only wins the tuning split**. The
human actions are pinning the candidate (step 4) and reading the final-split comparison (step 6)
-- the committed example below shows a prompt that won tuning and regressed on final.

## Inputs

- Corpus: `samples/corpus/ip_regulation_uk.md`
- Gold set: `samples/goldsets/ip_regulation_uk/goldset.jsonl`
- Prompt package: `samples/prompt_system/ip_regulation_uk/`
- Tuned prompt package: `samples/prompt_system/ip_regulation_uk/tuned/`
- Curated graph sample: `samples/prompt_system/ip_regulation_uk/graph/`
- Expected results: `samples/prompt_system/ip_regulation_uk/example_results.json`

## Commands

Build and validate the vector index:

```bash
env DATA_DIR=.data/prompt_system_ip_example .venv/bin/python -m llb.main build-index \
  --corpus-root samples/corpus

env DATA_DIR=.data/prompt_system_ip_example .venv/bin/python -m llb.main validate-retrieval \
  --goldset samples/goldsets/ip_regulation_uk/goldset.jsonl --k 5
```

Generate reviewable prompt-system candidates:

```bash
PROMPT_INSTRUCTION="Відповідай коротко, переважно точними словами з контексту. \
Якщо питання просить перелік або строк, дай тільки потрібні елементи без пояснень. \
Не додавай зовнішніх фактів."

env DATA_DIR=.data/prompt_system_ip_example .venv/bin/python -m llb.main prompt-system-prepare \
  --corpus-root samples/corpus \
  --out-dir samples/prompt_system/ip_regulation_uk/tuned \
  --context-window 8192 --chunk-tokens 1024 --answer-tokens 512 --max-passages 10 \
  --instruction "$PROMPT_INSTRUCTION"
```

Run the tuning split and pin the selected prompt:

```bash
env DATA_DIR=.data/prompt_system_ip_example HF_HUB_OFFLINE=1 .venv/bin/python -m llb.main run-eval \
  --model gemma4:e4b --backend ollama \
  --goldset samples/goldsets/ip_regulation_uk/goldset.jsonl --split tuning \
  --prompt-system 14d263ea6a40 --prompt-package samples/prompt_system/ip_regulation_uk/tuned

.venv/bin/python -m llb.main prompt-system-review \
  --run-dir samples/prompt_system/ip_regulation_uk/tuned --action pin --id 14d263ea6a40 \
  --note "Pinned after tuning split; verify on final before using."
```

Run the held-out final split:

```bash
env DATA_DIR=.data/prompt_system_ip_example HF_HUB_OFFLINE=1 .venv/bin/python -m llb.main run-eval \
  --model gemma4:e4b --backend ollama \
  --goldset samples/goldsets/ip_regulation_uk/goldset.jsonl --split final

env DATA_DIR=.data/prompt_system_ip_example HF_HUB_OFFLINE=1 .venv/bin/python -m llb.main run-eval \
  --model gemma4:e4b --backend ollama \
  --goldset samples/goldsets/ip_regulation_uk/goldset.jsonl --split final \
  --prompt-system 14d263ea6a40 --prompt-package samples/prompt_system/ip_regulation_uk/tuned
```

Compare prompt systems on final `run-eval` bundles:

```bash
env DATA_DIR=.data/prompt_system_ip_example .venv/bin/python -m llb.main prompt-system-compare \
  --lane rag --model gemma4:e4b
```

## Observed Result

On the committed sample, retrieval passed with recall@5=1.000 and MRR=1.000.

Gemma tuning split:

- Baseline RAG objective: 0.709
- Pinned tuned prompt `14d263ea6a40`: 0.778
- Richer tuned prompt `913266aa4cb3`: 0.683

Held-out final split:

- Baseline RAG objective: 0.687
- Pinned tuned prompt `14d263ea6a40`: 0.578
- Default prompt `0a68e417ea71`: 0.487

Decision: the prompt improved the tuning slice but did not generalize to final. For this tiny
sample, keep the baseline as the final recommendation and use the prompt-system lane as the
audit trail that proves the regression.

## Notes for New Corpora

1. Create a corpus directory and a verified gold set with separate `tuning` and `final` items.
2. Build the index and require retrieval to pass before tuning prompts.
3. Generate prompt-system candidates with a stable `--out-dir` if the artifacts should live under
   `samples/`; otherwise use the default `$DATA_DIR/prompt-system/<run>/`.
4. Tune on `--split tuning`, pin the best candidate, then run `--split final`.
5. Use `prompt-system-compare --lane rag --model <model>` to rank final prompt-system runs.
6. Record the conclusion beside the artifacts; do not promote a prompt that only wins tuning.

The committed `graph/` directory is a curated tutorial graph in GraphRAG JSONL shape. A live
`build-graph --extract-model gemma4:e4b --extract-no-think` run was attempted for this sample, but
the model response was unparseable JSON, so the guide keeps the generated graph separate from the
runtime extraction attempt.
