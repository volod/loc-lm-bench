# loc-lm-bench — Local Language Model Benchmark

Pick the best open-weight LLM for **your** Ukrainian RAG and text-analysis tasks, on **your**
hardware. Public leaderboards measure general capability on someone else's data with unlimited
VRAM; loc-lm-bench re-ranks a handful of candidate models on your own corpus, on a single
desktop GPU, so the choice is reproducible and defensible.

> **Status:** early. Milestone 0 (data prep: gold-item schema, validator, public-dataset
> ingestion, chunking) is **done and tested**. Milestones 1-3 (the CUDA-free eval skeleton,
> backends + telemetry, two-tier screen + leaderboard) are planned — see the docs.

## Main features

- **Corpus-grounded:** scores models on your documents and a span-labeled Ukrainian gold set,
  not transferred public scores.
- **Reuse public UA data:** one command pulls a real Ukrainian QA set (e.g. `HPLT/ua-squad`)
  into the canonical gold-item schema.
- **Source-span gold labels** (document char offsets, not chunk ids) — they survive
  `chunk_size` changes during tuning.
- **Chunking strategies:** build a RAG store with `fixed` / `sentence` / `recursive` chunking
  and compare them; optional FAISS index.
- **Backend-agnostic (planned):** Ollama / vLLM / llama.cpp behind one OpenAI-compatible
  interface, resolved per model.
- **Defensible scoring (planned):** objective reference-answer correctness + an LLM judge
  gated by Ukrainian calibration (Spearman rho); average-rank + Pareto leaderboard with
  confidence intervals.
- **Reproducible + lightweight:** canonical run manifests, deterministic disjoint splits, no
  heavy services.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) (it fetches Python 3.11 for you).

    make venv          # .venv (py3.11) + base deps + .env from .env.example
    make test          # run the test suite (27 tests)

Milestone 0 commands (data prep, output under `.data/`, gitignored):

    make gen-rag-items          # tiny sample gold set + corpus
    make validate-goldset       # validate spans resolve + splits disjoint
    make build-rag-store        # chunk samples/corpus with fixed/sentence/recursive
    make ingest-uk-squad        # real 250-item UA gold set from HPLT/ua-squad *
    make calibration-worksheet  # blank judge-calibration worksheet

`*` needs a Hugging Face token in `.env` (`HF_TOKEN=...`) and the datasets extra
(`uv pip install -e ".[goldset]"`). Run `make` with no target to list everything.

## Documentation

Start at the [docs index](docs/README.md). Highlights:

- [Design overview](docs/design/overview.md) — problem, wedge, what we build (full spec: [`docs/design.md`](docs/design.md)).
- [What's built today](docs/implementation/current.md) and the [forward plan](docs/implementation/plan.md).
- Guides: [dev setup](docs/guides/dev-setup.md), [data prep](docs/guides/data-prep.md).
- [`AGENTS.md`](AGENTS.md) — project guardrails for contributors and agents.

## Related projects
- [Ukrainian LLM Leaderboard HF](https://huggingface.co/spaces/lang-uk/ukrainian-llm-leaderboard)
- [Ukrainian LLM Leaderboard GitHub](https://github.com/lang-uk/ukrainian-llm-leaderboard)
- [Language Model Evaluation Harness](https://github.com/insait-institute/lm-evaluation-harness-uk)
- [MamayLLM](https://models.mamay.ai/)
- [MamamyLLM HF](https://huggingface.co/collections/INSAIT-Institute/mamaylm-v20-gemma-3)
- [LapaLLM HF](https://huggingface.co/spaces/lapa-llm/lapa)
- [LapaLLM GitHub](https://github.com/lapa-llm/lapa-llm)
