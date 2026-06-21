# Dev setup

Requires [uv](https://docs.astral.sh/uv/) (it fetches Python 3.11 for you).

    make venv     # .venv (py3.11) + the package + all extras + .env (one-time setup)
    make test     # unit tests
    make ci       # lint (ruff) + tests -- exactly what GitHub CI runs
    make          # list all targets

`make venv` installs every extra below so a fresh checkout can run every command without a
follow-up `uv pip install`. It is a larger one-time download; for a lean install trim it,
e.g. `make venv EXTRAS=dev` (or `EXTRAS=rag,eval` to just run the skeleton).

## Extras (what each group provides)

The groups installed by `make venv` (and what `EXTRAS=` selects from):

| Extra | Pulls | For |
|-------|-------|-----|
| `dev` | pytest, ruff | tests + lint |
| `goldset` | datasets | `ingest_squad --hf-dataset` |
| `rag` | faiss-cpu, sentence-transformers, langchain-text-splitters, DeepEval | indexing + local judge eval |
| `eval` | langgraph | the retrieve -> generate eval graph (`run-eval`) |
| `track` | mlflow, duckdb, pyarrow, optuna | tracking + config search |
| `board` | streamlit | leaderboard |
| `prep` | litellm | frontier-API prep utils |
| `telemetry` | nvidia-ml-py, psutil | GPU/host telemetry |

GitHub CI installs only `.[dev]` (it never runs `make venv`), so the lint+test job stays
light and never pulls the heavy/eval deps. vLLM / torch / flash-attn are hardware-matched
(host CUDA/GPU) and installed via a separate path per [AGENTS.md](../../AGENTS.md), never as
plain deps.

## Conventions
- Runtime output under `.data/` (gitignored); secrets in `.env` (gitignored).
- Resolve paths from the project root; never hardcode absolute home paths.
- ASCII in logs/comments; UTF-8 only in data payloads.
