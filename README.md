# loc-lm-bench — Local Language Model Benchmark

Pick the best open-weight LLM for **your** Ukrainian RAG and text-analysis tasks, on **your**
hardware. Public leaderboards measure general capability on someone else's data with unlimited
VRAM; loc-lm-bench re-ranks a handful of candidate models on your own corpus, on a single
desktop GPU, so the choice is reproducible and defensible.

> **Status:** early but runnable. Milestone 0 (data prep: gold-item schema, validator,
> public-dataset ingestion, chunking) and Milestone 1 (the CUDA-free eval skeleton:
> retrieve -> generate -> score -> ranked row + manifest, on one Ollama model) are **done
> and tested** (105 tests). Milestones 2-3 (real backends + telemetry, two-tier screen +
> leaderboard) are planned — see the docs.

## Main features

- **Corpus-grounded:** scores models on your documents and a span-labeled Ukrainian gold set,
  not transferred public scores.
- **Reuse public UA data:** one command pulls a real Ukrainian QA set (e.g. `HPLT/ua-squad`)
  into the canonical gold-item schema.
- **Source-span gold labels** (document char offsets, not chunk ids) — they survive
  `chunk_size` changes during tuning.
- **Chunking strategies:** build a RAG store with `fixed` / `sentence` / `recursive` chunking
  and compare them; optional FAISS index.
- **Backend-agnostic:** one OpenAI-compatible interface; the Ollama backend ships today
  (CUDA-free), with vLLM / llama.cpp + a per-model resolver planned.
- **Hardware-aware:** `list-models` reports which candidates can run on your GPU + RAM,
  KV-cache-aware, with a GPU/CPU layer split (it optimizes ability to run, not speed).
- **Defensible scoring:** objective reference-answer correctness ranks models today, with
  an LLM judge gated by Ukrainian calibration (Spearman rho) that stays demoted until it
  earns trust. Average-rank + Pareto leaderboard with confidence intervals is planned.
- **Reproducible + lightweight:** canonical run manifests, deterministic disjoint splits, no
  heavy services.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) (it fetches Python 3.11 for you).

    make venv          # .venv (py3.11) + the package + all extras + .env (one-time setup)
    make test          # run the test suite

Milestone 0 commands (data prep, output under `.data/`, gitignored):

    make gen-rag-items          # tiny sample gold set + corpus
    make validate-goldset       # validate spans resolve + splits disjoint
    make build-rag-store        # chunk samples/corpus with fixed/sentence/recursive
    make ingest-uk-squad        # real 250-item UA gold set from HPLT/ua-squad *
    make calibration-worksheet  # blank judge-calibration worksheet

Milestone 1 -- run the eval skeleton (`make venv` already installed the deps; needs a
running Ollama):

    make list-models            # which candidate models fit this GPU + RAM (context, layer split)
    make prep-models            # detect GPU; pull Ollama tags + cache vLLM HF weights
    make build-index            # chunk + embed the gold-set corpus into a FAISS store
    make validate-retrieval     # recall@k / MRR of the pinned embedding
    make run-eval MODEL=llama3.2:3b   # one ranked row + a reproducible manifest

`*` needs a Hugging Face token in `.env` (`HF_TOKEN=...`). Run `make` with no target to
list everything. See the [run-the-skeleton guide](docs/guides/run-skeleton.md) for the full
flow.

## Documentation

Start at the [docs index](docs/README.md). Highlights:

- [Design](docs/design/README.md) — contents map into the full spec ([`docs/design/spec.md`](docs/design/spec.md)).
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
