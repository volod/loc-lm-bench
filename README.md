# loc-lm-bench -- Local Language Model Benchmark

Pick the best open-weight LLM for **your** Ukrainian RAG and text-analysis tasks, on **your**
hardware. Public leaderboards rank general capability on someone else's data with unlimited VRAM;
loc-lm-bench re-ranks a handful of candidate models on your own corpus, on a single desktop GPU
(validated on an RTX 4060 Ti 16 GB), so the choice is reproducible and defensible.

Status: runs end to end -- data prep, Ollama + vLLM + llama.cpp serving, telemetry, a ranked board,
GraphRAG, and category benchmarks (Milestones 0-6). Milestone 7 (extended agentic workflows) is the
open workstream. See [what's built today](docs/impl/current.md) and the
[forward plan](docs/impl/plan.md).

## Features

Each feature links to its manual.

| Feature | What you get |
|---|---|
| [Corpus-grounded gold set](docs/guides/goldset-from-scratch.md) | Char-offset source-span labels, stable under chunk tuning. |
| [Backend-agnostic serving](docs/guides/vllm-backend.md) | Ollama + vLLM + llama.cpp; resolver + VRAM-contention guard. |
| [Hardware-aware planning](docs/inference/config-example.md) | `list-models` reports which candidates fit this GPU + RAM. |
| [Gated LLM judge](docs/guides/calibration-tooling.md) | Judge ranks only after UA calibration (Spearman rho >= 0.6). |
| [Rigorous leaderboard](docs/guides/mlflow-analysis.md) | Average rank + Pareto front + bootstrap CIs; local MLflow. |
| [Graph + FAISS retrieval](docs/guides/graph-vs-faiss-comparison.md) | FAISS and GraphRAG stores, compared on one gold set. |
| [Human-in-the-loop gates](docs/guides/human-in-the-loop-evaluation.md) | Schema sign-off, cross-check, sample-verify. |

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) (it fetches Python 3.11) and a running Ollama
(`ollama serve`). A CUDA GPU and an `HF_TOKEN` are only needed for vLLM / llama.cpp and gated
weights. The first `make venv` seeds `.env` and stops with a setup notice (exit 0, **not** an
error); set `HF_TOKEN`, `DATA_DIR`, and `OLLAMA_HOST` / `VLLM_HOST`, then re-run. From a fresh
clone to a ranked leaderboard:

    # 0. One-time setup: .venv (py3.11) + all extras + .env (idempotent).
    make venv

    # 1. Smoke the whole chain end to end (idempotent): gold set -> index ->
    #    prep-models -> one ranked row + telemetry under .data/llb/.
    make demo-eval

    # 2. Point retrieval at your own corpus and validate it.
    make build-index CORPUS=<dir>
    make validate-retrieval        # recall@k / MRR of the pinned embedder

    # 3. Rank candidates, then tune and finalize.
    llb sweep --sweep-id run1      # VRAM + thermal gated, process-isolated
    llb pipeline                   # finalists -> Optuna tune -> final board

    # 4. Review the leaderboard.
    llb board                      # avg rank, Pareto front, bootstrap CIs
    make mlflow                    # per-run inspection at 127.0.0.1:5000

Run `make` with no target to list every command; `.env.example` documents every variable.

## Documentation

Start at the [documentation index](docs/README.md). Key entry points:

| Doc | What it covers |
|---|---|
| [What's built today](docs/impl/current.md) | Delivered milestones, modules, commands, results. |
| [Forward plan](docs/impl/plan.md) | Open roadmap (Milestone 7). |
| [Design spec](docs/design/spec.md) | Problem, wedge, architecture, recorded decisions. |
| [Dev setup](docs/guides/dev-setup.md) | Requirements, uv, venv, extras, apt deps, targets. |
| [AGENTS.md](AGENTS.md) | Contributor + agent guardrails (paths, MAX_JOBS, doc cycle). |
| [Learning path](docs/guides/learning-path.md) | Staged syllabus + links for the whole stack. |

## Related projects & benchmarks

loc-lm-bench is **not** a public leaderboard -- it consumes Ukrainian eval prior art as a free prior
and re-ranks on your private corpus. Re-verify each dataset's license and preserve attribution before
any redistribution (see the [design spec](docs/design/spec.md)).

- **Ukrainian LLM Leaderboard** -- public UA ranking / transfer baseline
  ([HF](https://huggingface.co/spaces/lang-uk/ukrainian-llm-leaderboard),
  [GitHub](https://github.com/lang-uk/ukrainian-llm-leaderboard)).
- **lm-evaluation-harness-uk** -- INSAIT EleutherAI fork; powers the Tier-1 public screen
  ([GitHub](https://github.com/insait-institute/lm-evaluation-harness-uk)).
- **Candidate models** -- [MamayLM v2](https://models.mamay.ai/) (Gemma-3 UA-specialized; top
  candidate + possible local judge), [Lapa LLM](https://huggingface.co/spaces/lapa-llm/lapa),
  [Gemma 4](https://huggingface.co/collections/google/gemma-4),
  [Qwen 3.6](https://huggingface.co/collections/Qwen/qwen36), and
  [Mistral Small](https://huggingface.co/collections/mistralai/mistral-small-4).
- **Datasets** -- UA-SQuAD ([`FIdo-AI/ua-squad`](https://huggingface.co/datasets/FIdo-AI/ua-squad),
  the committed gold-set fixture) and Belebele-uk
  ([`facebook/belebele`](https://huggingface.co/datasets/facebook/belebele), Tier-1 MCQ screen).
- **MamayLM v2 benchmarks** (reference only; (c) INSAIT / MamayLM) --
  [release post](https://models.mamay.ai/blog/mamaylm-v2-release-en/).
