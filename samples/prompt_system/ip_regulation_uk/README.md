# IP regulation prompt-system sample

This directory is the M7.3 RAG prompt-system example for
`samples/goldsets/ip_regulation_uk/goldset.jsonl`.

Contents:

- Root `manifest.json`, `candidates.json`, `anthology.json`, `doc_metadata.json`, and
  `graph_rag_mapping.json`: default generated prompt-system package.
- `tuned/`: generated with a short-answer instruction override; candidate
  `14d263ea6a40` is pinned after the tuning split.
- `graph/`: curated tutorial knowledge graph in the same JSONL shape as the GraphRAG store.
- `example_results.json`: observed Gemma 4 E4B tuning/final scores and run bundle paths.

Recreate the local example:

```bash
env DATA_DIR=.data/m7-3r-ip-example .venv/bin/python -m llb.main build-index \
  --corpus-root samples/corpus

env DATA_DIR=.data/m7-3r-ip-example .venv/bin/python -m llb.main validate-retrieval \
  --goldset samples/goldsets/ip_regulation_uk/goldset.jsonl --k 5

env DATA_DIR=.data/m7-3r-ip-example .venv/bin/python -m llb.main prompt-system-prepare \
  --corpus-root samples/corpus \
  --out-dir samples/prompt_system/ip_regulation_uk/tuned \
  --context-window 8192 --chunk-tokens 1024 --answer-tokens 512 --max-passages 10 \
  --instruction "Відповідай коротко, переважно точними словами з контексту. Якщо питання просить перелік або строк, дай тільки потрібні елементи без пояснень. Не додавай зовнішніх фактів."

env DATA_DIR=.data/m7-3r-ip-example HF_HUB_OFFLINE=1 .venv/bin/python -m llb.main run-eval \
  --model gemma4:e4b --backend ollama \
  --goldset samples/goldsets/ip_regulation_uk/goldset.jsonl --split tuning \
  --prompt-system 14d263ea6a40 --prompt-package samples/prompt_system/ip_regulation_uk/tuned

env DATA_DIR=.data/m7-3r-ip-example HF_HUB_OFFLINE=1 .venv/bin/python -m llb.main run-eval \
  --model gemma4:e4b --backend ollama \
  --goldset samples/goldsets/ip_regulation_uk/goldset.jsonl --split final \
  --prompt-system 14d263ea6a40 --prompt-package samples/prompt_system/ip_regulation_uk/tuned

env DATA_DIR=.data/m7-3r-ip-example .venv/bin/python -m llb.main prompt-system-compare \
  --lane rag --model gemma4:e4b
```

Observed result: retrieval passed at recall@5=1.000 and MRR=1.000. The tuned prompt improved
the tuning objective from 0.709 to 0.778, but it regressed on the held-out final split
from 0.687 baseline to 0.578. Keep the final baseline decision separate from prompt selection.
