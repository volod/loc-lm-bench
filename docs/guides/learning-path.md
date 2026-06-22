# Learning path -- the loc-lm-bench stack

A guided path to understanding every technology this project uses, with curated links and a
pointer to where each one lives in the code. It ends with a time-boxed syllabus for a learner who
has **basic knowledge** (some Python, some general ML / LLM familiarity) and wants to become
productive here.

## Who this is for

You can read code and write small Python scripts, you know roughly what a token, an embedding,
and a prompt are, and you have used the command line and git. You do NOT need prior experience
with RAG, GPU serving, Optuna, or LLM-as-judge evaluation -- those are what this path teaches.

## How to use it

- Work the stages **in order**; each builds on the previous one.
- Every stage has a **"In this repo"** pointer so you connect theory to the actual code.
- You do not need to finish a stage's deep links before moving on -- skim first, return for depth.
- The fastest way to learn the whole loop is to run `make demo-eval` once (see the
  [README](../../README.md)) and read the manifest it writes under `.data/llb/`.

## Syllabus at a glance

| Stage | Module | You will learn | In this repo |
|---|---|---|---|
| 0 | Prerequisites | Python, git, CLI, basic ML/LLM vocabulary | -- |
| 1 | Project tooling | uv, Typer, Pydantic, pytest | `main.py`, `config.py` |
| 2 | Retrieval + RAG | embeddings, FAISS, chunking, recall@k / MRR | `rag/` |
| 3 | Serving LLMs locally | Ollama, vLLM, llama.cpp, quantization, OpenAI-compatible API, KV cache / VRAM | `backends/` |
| 4 | Eval flow orchestration | LangGraph graphs + failure taxonomy | `eval/graph.py` |
| 5 | Scoring + LLM-as-judge | reference correctness, DeepEval G-Eval, judge calibration (Spearman, bootstrap CI), bias | `scoring/`, `judge/` |
| 6 | Tuning + tracking | Optuna, MLflow, DuckDB / Parquet, manifests, disjoint splits | `optimize/`, `tracking/` |
| 7 | Hardware + isolation | NVML / pynvml, VRAM, thermal, process-isolated sweeps | `executor/`, `backends/telemetry.py` |
| 8 | Public benchmarks + UA NLP | lm-evaluation-harness, lang-uk / INSAIT, UA morphology, datasets | `screen/` |
| 9 (forward) | Security / agentic / tooling / GraphRAG | MCP, function calling, agentic eval, LLM security, knowledge-graph RAG | `plan.md` (M5 / M6) |

## Stage 0 -- Prerequisites

- Python: [official tutorial](https://docs.python.org/3/tutorial/).
- Git basics: [Pro Git book](https://git-scm.com/book/en/v2).
- LLM mental model (tokens, context, sampling): Karpathy's
  [Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g).

## Stage 1 -- Python project tooling

What turns scripts into a reproducible package and a typed CLI.

- [`uv`](https://docs.astral.sh/uv/) -- the fast Python package + environment manager (this repo
  uses it for everything; see `make venv`).
- [Typer](https://typer.tiangolo.com/) -- the CLI framework behind the `llb` command.
- [Pydantic](https://docs.pydantic.dev/latest/) -- typed, validated config (`RunConfig`).
- [pytest](https://docs.pytest.org/) -- the test runner (`make test`).

In this repo: `src/llb/main.py` (CLI), `src/llb/config.py` (the canonical `RunConfig`),
`pyproject.toml` (deps + extras).

## Stage 2 -- Retrieval and RAG foundations

Retrieval-Augmented Generation: fetch relevant text, then let the model answer from it.

- RAG, the idea: [Lewis et al. 2020](https://arxiv.org/abs/2005.11401) and the practical
  [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/).
- Embeddings + semantic search: [sentence-transformers](https://www.sbert.net/); the pinned
  Ukrainian-capable model is [multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-large).
- Vector index: [FAISS](https://faiss.ai/) ([wiki](https://github.com/facebookresearch/faiss/wiki)).
- Chunking: [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
  -- why `chunk_size` / overlap / structure-aware splitting matter.
- Retrieval metrics: recall@k and
  [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank).

In this repo: `src/llb/rag/` (chunking, embedding, FAISS index, store, and source-span retrieval
metrics). Key idea: gold labels are **source spans** (char offsets), so retrieval scoring
survives `chunk_size` changes.

## Stage 3 -- Serving open-weight LLMs locally

All three backends speak one OpenAI-compatible HTTP API, so the eval code is backend-agnostic.

- [Ollama](https://github.com/ollama/ollama) -- easiest local serving (GGUF, CPU offload).
- [vLLM](https://docs.vllm.ai/en/latest/) -- high-throughput HF-weight serving; read its
  [PagedAttention paper](https://arxiv.org/abs/2309.06180) for the KV-cache idea.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) -- portable GGUF serving (planned backend).
- Quantization (why a 12B fits in 16 GB): [GGUF](https://huggingface.co/docs/hub/gguf) and the
  [HF quantization overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)
  (w4a16 / AWQ / GPTQ).
- The shared interface: [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat).

In this repo: `src/llb/backends/` (base launcher seam, OpenAI client, Ollama, vLLM, the
availability resolver, hardware detection, prepare, planner, telemetry).

## Stage 4 -- Orchestrating eval flows

Each eval case is a small graph (retrieve -> generate -> classify), built from reusable templates.

- [LangGraph](https://langchain-ai.github.io/langgraph/) -- stateful graphs of LLM steps; the
  substrate for the RAG flow now and the agentic loop later.

In this repo: `src/llb/eval/graph.py` (the retrieve -> generate flow + a typed failure taxonomy:
empty / malformed / refusal / timeout / backend_error / retrieval_miss).

## Stage 5 -- Scoring and LLM-as-judge

How models are ranked, and why the judge is gated.

- Objective scoring: exact / token-F1 / contains, plus optional embedding-cosine for paraphrase.
- LLM-as-judge: [DeepEval](https://github.com/confident-ai/deepeval) G-Eval metrics
  (faithfulness, answer relevancy); the method comes from
  [G-Eval](https://arxiv.org/abs/2303.16634) and the broader
  [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) (MT-Bench) study.
- Calibration: a judge is trusted only if it agrees with humans --
  [Spearman rank correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
  with a [bootstrap confidence interval](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
- Judge bias: a Gemma-family judge may self-prefer Gemma answers -- this project discloses and
  gates it (see `current.md`).

In this repo: `src/llb/scoring/` (correctness, the gated judge, the N-model board) and
`src/llb/judge/calibration.py` (rho + CI + trust decision).

## Stage 6 -- Tuning, tracking, reproducibility

Search configs without leaking into the leaderboard number, and record everything.

- [Optuna](https://optuna.org/) ([docs](https://optuna.readthedocs.io/en/stable/)) -- the
  two-stage RAG-config search (tuning split -> final split).
- [MLflow](https://mlflow.org/docs/latest/) -- local file/SQLite mirror of runs for its UI.
- [DuckDB](https://duckdb.org/docs/) + [Parquet](https://parquet.apache.org/docs/) -- columnar
  per-case results.
- Reproducibility: disjoint calibration / tuning / final splits and an immutable per-run manifest
  are the source of truth (MLflow only mirrors them).

In this repo: `src/llb/optimize/tuner.py`, `src/llb/tracking/` (manifest + MLflow mirror),
`src/llb/goldset/splits.py`.

## Stage 7 -- Hardware, telemetry, and isolation

Make measurements comparable and prevent one run from biasing the next.

- GPU telemetry: [NVML](https://developer.nvidia.com/management-library-nvml) via
  [pynvml / nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) -- VRAM, temperature, clocks,
  power.
- Why isolation: a leaked CUDA context or a hot GPU skews tokens/sec and VRAM; each (model,
  config) cell runs in its own process, then a PID-attributed VRAM-reclaim gate + a capped
  thermal cooldown run between cells.

In this repo: `src/llb/executor/` (runner, VRAM gate, `isolate_cell`) and
`src/llb/backends/telemetry.py` (steady-state tokens/sec, peak VRAM, tokenizer efficiency).

## Stage 8 -- Public benchmarks and Ukrainian NLP

The Tier-1 screen and the Ukrainian-specific gotchas.

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the
  [UA fork](https://github.com/insait-institute/lm-evaluation-harness-uk) -- standard tasks via
  `local-completions` against your own endpoint.
- Ukrainian NLP: [lang-uk](https://github.com/lang-uk),
  [spaCy uk](https://spacy.io/models/uk), [Stanza](https://stanfordnlp.github.io/stanza/) --
  morphology breaks naive string matching, which is why scoring uses source spans + embeddings.
- Datasets used as a prior + smoke baseline: SQuAD-uk and Belebele-uk.

In this repo: `src/llb/screen/public.py` (two never-cross-ranked tracks: logprob vs generation).

## Stage 9 (forward) -- Security, agentic, tooling, and GraphRAG

What the roadmap adds next (designed, not yet built; see
[`plan.md`](../implementation/plan.md) M5 / M6).

- Tool use: [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling),
  the [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html),
  and the [Model Context Protocol](https://modelcontextprotocol.io/)
  ([Python SDK](https://github.com/modelcontextprotocol/python-sdk)).
- LLM security: [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/),
  [garak](https://github.com/NVIDIA/garak), and adversarial sets like
  [JailbreakBench](https://jailbreakbench.github.io/) and [HarmBench](https://www.harmbench.org/).
- Knowledge-graph RAG: [Kuzu](https://kuzudb.com/) (embedded graph DB,
  [docs](https://docs.kuzudb.com/)) and Microsoft's
  [GraphRAG](https://microsoft.github.io/graphrag/) ([paper](https://arxiv.org/abs/2404.16130)).

## Syllabus for a learner with basic knowledge

A time-boxed plan (about 2-4 hours per session). Each session pairs reading with a concrete
action in this repo.

| Session | Read (stages) | Do in the repo |
|---|---|---|
| 1. Get it running | README + Stage 1 | `make venv`, `make test`, `make demo-eval`; open the manifest under `.data/llb/`. |
| 2. RAG basics | Stage 2 | `make build-index`, `make validate-retrieval`; change `--strategy` / `--size` and watch recall@k move. |
| 3. Serving | Stage 3 | `make list-models`, `make prep-models`, `make run-eval MODEL=...`; read the telemetry in the manifest. |
| 4. The eval flow | Stage 4 | Read `eval/graph.py`; trace one case from retrieve to a typed status. |
| 5. Scoring + judge | Stage 5 | `make calibration-worksheet`; read `scoring/` and `judge/calibration.py`; understand the rho >= 0.6 gate. |
| 6. Tuning + tracking | Stage 6 | `llb tune --model ... --backend ...` on a small budget; compare trials in `make mlflow`. |
| 7. Scale + isolation | Stage 7 + 8 | `llb sweep --sweep-id demo` then re-run it (resume); read `executor/isolation`; try `llb screen-public`. |
| 8. The roadmap | Stage 9 | Read [`plan.md`](../implementation/plan.md); pick one M5 category and sketch its scoring schema. |

By session 8 you can run the full pipeline, explain every number on the board, and reason about
the forward plan. From here, the deepest single source is the
[design spec](../design/spec.md); the [current state](../implementation/current.md) maps each
module to its behavior.

## A note on the project's philosophy

loc-lm-bench follows **reuse over rebuild**: it leans on maintained libraries (DeepEval, FAISS,
LangGraph, Optuna, MLflow) and writes only the glue, the Ukrainian gold set, and the judge
calibration. As you learn each technology, notice *where the project stops customizing* -- that
boundary is itself a lesson in scoping a defensible internal tool.
