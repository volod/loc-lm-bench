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

- **Stage 0 -- Prerequisites** (--): Python, git, CLI, basic ML/LLM vocabulary
- **Stage 1 -- Project tooling** (`main.py`, `config.py`): uv, Typer, Pydantic, pytest
- **Stage 2 -- Retrieval + RAG** (`rag/`): embeddings, FAISS, chunking, recall@k / MRR
- - **Stage 3 -- Serving LLMs locally** (`backends/`): Ollama, vLLM, llama.cpp, quantization,
- OpenAI-compatible API, KV cache / VRAM
- **Stage 4 -- Eval flow orchestration** (`eval/graph.py`): LangGraph graphs + failure taxonomy
- - **Stage 5 -- Scoring + LLM-as-judge** (`scoring/`, `judge/`): reference correctness, DeepEval
- G-Eval, judge calibration (Spearman, bootstrap CI), judge bias
- - **Stage 6 -- Tuning + tracking** (`optimize/`, `tracking/`): Optuna, MLflow, DuckDB / Parquet,
- manifests, disjoint splits
- - **Stage 7 -- Hardware + isolation** (`executor/`, `backends/telemetry.py`): NVML / pynvml, VRAM,
- thermal, process-isolated sweeps
- - **Stage 8 -- Public benchmarks + UA NLP** (`screen/`): lm-evaluation-harness, lang-uk / INSAIT,
- UA morphology, datasets
- - **Stage 9 -- LLM security** ([security learning path](learning-path-security.md)): Jailbreaks,
- prompt injection, instruction hierarchy, destructive actions, leakage, bias
- - **Stage 10 (forward) -- Agentic / tooling / GraphRAG** ([evaluation-categories learning
- path](learning-path-evaluation-categories.md)): MCP, function calling, agentic eval,
- knowledge-graph RAG


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
  Ukrainian-capable model is
  [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base).
- Vector index: [FAISS](https://faiss.ai/) ([wiki](https://github.com/facebookresearch/faiss/wiki)).
- Chunking: [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
  -- why `chunk_size` / overlap / structure-aware splitting matter.
- Retrieval metrics: recall@k and
  [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank).

In this repo: `src/llb/rag/` (chunking, embedding, FAISS index, store, and source-span retrieval
metrics). Key idea: gold labels are **source spans** (char offsets), so retrieval scoring
survives `chunk_size` changes.

## Stage 3 -- Serving open-weight LLMs locally

The serving design standardizes on one OpenAI-compatible HTTP API so eval code stays
backend-agnostic. Ollama and vLLM are implemented; llama.cpp is the planned third backend.

- [Ollama](https://github.com/ollama/ollama) -- easiest local serving (GGUF, CPU offload).
- [vLLM](https://docs.vllm.ai/en/latest/) -- high-throughput HF-weight serving; read its
  [PagedAttention paper](https://arxiv.org/abs/2309.06180) for the KV-cache idea.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) -- portable GGUF serving (planned backend).
- Quantization (why a 12B fits in 16 GB): [GGUF](https://huggingface.co/docs/hub/gguf) and the
  [HF quantization overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)
  (w4a16 / AWQ / GPTQ).
- The shared interface: [OpenAI Chat
  Completions](https://platform.openai.com/docs/api-reference/chat).

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
  gates it (see the [current-state disclosure](../impl/current.md)). This is distinct
  from social, cultural, and political bias, which Stage 9 treats as model behavior to measure
  directly.

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
  thermal cooldown runs between cells.

In this repo: `src/llb/executor/` (runner, VRAM gate, `isolation.py`) and
`src/llb/backends/telemetry.py` (steady-state tokens/sec, peak VRAM, tokenizer efficiency).

## Stage 8 -- Public benchmarks and Ukrainian NLP

The public-benchmark screen and the Ukrainian-specific gotchas.

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the
  [UA fork](https://github.com/insait-institute/lm-evaluation-harness-uk) -- standard tasks via
  `local-completions` against your own endpoint.
- Ukrainian NLP: [lang-uk](https://github.com/lang-uk),
  [spaCy uk](https://spacy.io/models/uk), [Stanza](https://stanfordnlp.github.io/stanza/) --
  morphology breaks naive string matching, which is why scoring uses source spans + embeddings.
- Datasets used as a prior + smoke baseline: SQuAD-uk and Belebele-uk.

In this repo: `src/llb/screen/public.py` (two never-cross-ranked tracks: logprob vs generation).

## Stage 9 -- LLM-specific security and responsible evaluation

Learn to distinguish model safety from application security, define the attacker and protected
assets, and measure failures without giving a model access to real secrets or destructive tools.
The benchmark implementation is planned, not delivered; see the
[security learning path](learning-path-security.md) and the
[forward plan](../impl/plan.md).

- Threat modeling: start with the
[OWASP Top 10 for LLM
Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/),
[NIST Generative AI
Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence),
  and [MITRE ATLAS](https://atlas.mitre.org/). Identify assets, trust boundaries, attacker
  access, and impact before selecting attacks.
- Jailbreaks and prompt injection: a jailbreak tries to bypass a model's safety behavior;
  prompt injection makes untrusted text compete with application instructions. Test direct user
  attacks, indirect instructions in retrieved documents or tool output, and multi-turn variants.
- Instruction following: measure whether the model follows the instruction hierarchy, ignores
  lower-trust conflicting text, completes valid tasks, and avoids both under-refusal and
  over-refusal. See [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208).
- Destructive actions: assume model output is untrusted. Use typed, allowlisted tools; least
  privilege; a sandbox; dry-run previews; approval for consequential writes; bounded retries;
  and an audit log. Never test deletion, external messages, purchases, or account changes on a
  real system.
- Leakage and unsafe output handling: use synthetic secrets and canaries to test prompt or data
  exfiltration. Validate generated SQL, shell, HTML, URLs, and tool arguments before any consumer
  sees them.
- Bias and censorship: evaluate the exact model, version, chat template, provider, and language.
  Research has found social bias in Chinese-language models and elevated political-content
  refusal or omission in some models developed in mainland China; these findings do not justify
  treating every Chinese model as equivalent. Compare Chinese-origin and non-Chinese controls,
  local weights and hosted endpoints, and matched prompts in Chinese, Ukrainian, and English.
  Useful starting points are [CBBQ](https://aclanthology.org/2024.lrec-main.260/),
  [McBE](https://aclanthology.org/2025.findings-acl.313/), and the comparative
  [political-censorship study](https://academic.oup.com/pnasnexus/article/5/2/pgag013/8487339).
- Evaluation: report clean task success, attack success rate (ASR), instruction-hierarchy
  violation rate, canary leakage, unsafe tool-call rate, and over-refusal on benign controls.
  Keep each attack family separate and attach confidence intervals.

Follow the [extended LLM security learning path](learning-path-security.md) for the full topic
map, safe lab rules, benchmark design, and an eight-session practical syllabus.

In this repo: the planned security suite will reuse `src/llb/eval/`, process isolation, manifests,
and confidence-interval reporting under a separate security tier. Do not interpret the current
RAG score as a security score.

## Stage 10 (forward) -- Agentic, tooling, and GraphRAG

What the roadmap adds after the current stack (designed, not yet built; see the
[evaluation-categories learning path](learning-path-evaluation-categories.md)).

- Tool use: [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling),
  the [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html),
  and the [Model Context Protocol](https://modelcontextprotocol.io/)
  ([Python SDK](https://github.com/modelcontextprotocol/python-sdk)).
- Agentic workflows: multi-step LangGraph tasks over a deterministic sandbox, with objective
  state assertions and tool-call efficiency. Security tests for those tools belong to Stage 9.
- Knowledge-graph RAG: [Kuzu](https://kuzudb.com/) (embedded graph DB,
  [docs](https://docs.kuzudb.com/)) and Microsoft's
  [GraphRAG](https://microsoft.github.io/graphrag/) ([paper](https://arxiv.org/abs/2404.16130)).

## Syllabus for a learner with basic knowledge

A time-boxed plan (about 2-4 hours per session). Each session pairs reading with a concrete
action in this repo.

- - **1. Get it running** (README + Stage 1): `make venv`, `make test`, `make demo-eval`; open the
- manifest under `.data/llb/`.
- - **2. RAG basics** (Stage 2): Run `llb build-index --strategy ... --size ...`, then `make
- validate-retrieval`; watch recall@k move.
- - **3. Serving** (Stage 3): `make list-models`, `make prep-models`, `make run-eval MODEL=...`;
- read the telemetry in the manifest.
- - **4. The eval flow** (Stage 4): Read `eval/graph.py`; trace one case from retrieve to a typed
- status.
- - **5. Scoring + judge** (Stage 5): `make calibration-worksheet`; read `scoring/` and
- `judge/calibration.py`; understand the rho >= 0.6 gate.
- - **6. Tuning + tracking** (Stage 6): `llb tune --model ... --backend ...` on a small budget;
- compare trials in `make mlflow`.
- - **7. Scale + isolation** (Stage 7 + 8): `llb sweep --sweep-id demo` then re-run it (resume);
- read `src/llb/executor/isolation.py`; try `llb screen-public --model ... --limit 10`.
- - **8. LLM security** (Stage 9): Complete sessions 1-2 of the [security
- path](learning-path-security.md); sketch one security case with a benign control and objective
- detector.
- - **9. The roadmap** (Stage 10): Read the [evaluation-categories learning
- path](learning-path-evaluation-categories.md); pick one tool-use, agentic, or knowledge-graph
- category and sketch its scoring schema.


By session 9 you can run the full pipeline, explain every number on the board, and reason about
the forward plan. From here, the deepest single source is the
[design spec](../design/spec.md); the [current state](../impl/current.md) maps each
module to its behavior.

## A note on the project's philosophy

loc-lm-bench follows **reuse over rebuild**: it leans on maintained libraries (DeepEval, FAISS,
LangGraph, Optuna, MLflow) and writes only the glue, the Ukrainian gold set, and the judge
calibration. As you learn each technology, notice *where the project stops customizing* -- that
boundary is itself a lesson in scoping a defensible internal tool.
