# Dev setup

Requires [uv](https://docs.astral.sh/uv/) (it fetches Python 3.11 for you).

    make venv     # .venv (py3.11) + base deps + .env from .env.example
    make test     # 27 unit tests
    make ci       # lint (ruff) + tests -- exactly what GitHub CI runs
    make          # list all targets

## Extras (opt-in, kept out of the base install)

`uv pip install -e ".[NAME]"`, where NAME is one of:

| Extra | Pulls | For |
|-------|-------|-----|
| `dev` | pytest, ruff | tests + lint |
| `goldset` | datasets | `ingest_squad --hf-dataset` |
| `rag` | faiss-cpu, sentence-transformers, langchain-text-splitters, ragas | indexing + RAG eval |
| `track` | mlflow, duckdb, pyarrow, optuna | tracking + config search |
| `board` | streamlit | leaderboard |
| `prep` | litellm | frontier-API prep utils |
| `telemetry` | nvidia-ml-py, psutil | GPU/host telemetry |

vLLM / torch / flash-attn are hardware-matched (host CUDA/GPU) and installed via a separate
path per [AGENTS.md](../../AGENTS.md), never as plain deps.

## Conventions
- Runtime output under `.data/` (gitignored); secrets in `.env` (gitignored).
- Resolve paths from the project root; never hardcode absolute home paths.
- ASCII in logs/comments; UTF-8 only in data payloads.
