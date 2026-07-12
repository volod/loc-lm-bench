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

## Context-Policy Comparison (chain-context)

The prompt-system idea extends past a single RAG prompt to the SEQUENCE of system prompts used
across a multi-step question. `bench-chain-context` ranks four context-management policies for one
fixed model over a verified chain-of-questions set, where each step's answer depends on the prior
steps. The model, chain set, retrieval, and scoring stay fixed; only the policy -- the row label --
varies (the same discipline as the agentic harness comparison).

The four policies each retrieve fresh per step and differ only in the memory carried forward:

- `fresh` -- no carryover (the naive baseline);
- `history` -- the full prior (question, answer) transcript;
- `summary` -- a running model-written summary of the prior steps;
- `roles` -- a staged librarian -> analyst -> answerer system-prompt sequence built from the
  `bench.chain_context.role_*` prompt-system templates, plus the transcript.

Inputs (committed fixture): the 20 human-verified chains and compact corpus under
`samples/goldsets/chain_context_uk_v1/`.

```bash
env DATA_DIR=.data .venv/bin/python -m llb.main bench-chain-context \
  --chains samples/goldsets/chain_context_uk_v1/chains.jsonl \
  --corpus samples/goldsets/chain_context_uk_v1/corpus \
  --model <model> --backend ollama \
  --policies fresh,history,summary,roles --top-k 4
```

Each policy writes its own run bundle under `$DATA_DIR/chain-context/<timestamp>/` tagged with the
policy, the `prompt_system_ids`, and the `chain_set_digest`. The command prints a policy-ranked
board (final-answer correctness with bootstrap CIs) and a recommendation naming the winning policy;
`llb recommend` renders a "Context policy" section per model. Context assembly per policy per step
is unit-tested over a fake endpoint (`tests/llb/bench/test_chain_context.py`), so the comparison is
provable without a GPU.

### CUDA evidence (RTX 4060 Ti 16 GB)

Run of 2026-07-11 over the committed 20-chain fixture (40 steps) with
`hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` on Ollama, all four policies in
one invocation (~11 min wall clock, reliability 1.000 for every policy):

| rank | policy | final objective | final CI | per-step objective |
| --- | --- | ---: | --- | ---: |
| 1 | roles | **0.789** | [0.635, 0.915] | 0.784 |
| 2 | history | 0.625 | [0.49, 0.76] | 0.604 |
| 3 | summary | 0.534 | [0.42, 0.66] | 0.564 |
| 4 | fresh | 0.431 | [0.29, 0.56] | 0.508 |

The ranking discriminates and matches the design intent: the naive `fresh` baseline (no carryover)
is worst, any memory of the prior steps helps, and the staged `roles` sequence
(librarian -> analyst -> answerer) wins with its CI resolved above the other three -- so for this
model and chain set the recommendation is to sequence the system prompt rather than dump the raw
transcript. Run bundles: `.data/chain-context/20260711T1938*` (one per policy); `llb recommend`
renders the "Context policy" section naming `roles` as the best policy for this model.

The same tuning caution applies as for the RAG prompt-system lane: treat a policy win as
provisional until it holds on a set larger than one committed fixture, and keep the run bundles as
the audit trail behind the recommendation.
