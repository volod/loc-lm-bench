# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every line is open work. What already exists (behavior + results), and the operator
workflows an operator re-runs as needed (new gold set, cross-check, sample-verify, calibration, the
verified-data gate, graph-vs-FAISS comparison), live in [`current.md`](current.md); the spec (source
of truth) is [`docs/design/spec.md`](../design/spec.md).

The one open workstream is **Milestone 7** (extended + deferred + forward-verification). Its
sequence number is a stable identifier (AGENTS.md); it appears only while it has open work.

---

## Milestone 7 -- Extended, deferred, and forward-verification tasks

Four parts are non-blocking or deferred: the extended agentic harness comparison (M7.1), the
non-blocking quality-verification actions (M7.2, provable in CI from fake endpoints), the
human-assisted RAG prompt-system generation lane (M7.3), and deferred work that needs more
hardware or a committed consumer (M7.4).

### M7.1 Extended agentic workflows (LangGraph vs CrewAI harness)

M5.3 ranks the MODEL under ONE fixed harness (the pure controller->execute->controller loop; see
`current.md`). This task adds exactly ONE alternative harness -- **CrewAI** -- so the comparison
axis is **LangGraph vs CrewAI**, holding everything else fixed. This isolates the HARNESS effect
("how much does the agent framework itself move the score" on the same model). The other frameworks
the spec lists as deferred (LangChain, LlamaIndex, Haystack, AutoGen) stay OUT of scope -- two
harnesses are enough to establish the comparison method.

**Decided scope:** the candidate model, the M5.3 task set + the deterministic `ToolWorld`, the
objective scoring (completion-rate + per-task CI), and the opt-in gated trajectory-quality judge are
all HELD FIXED; the HARNESS is the only variable. CrewAI is an OPT-IN, lazy-imported extra so the
base install stays light, and a fake crew proves the wiring with no dependency / GPU (the same
injectable-`complete` discipline as the rest of M5).

**Design (how to compare the two harnesses):**
1. A `Harness` seam -- a `Protocol` `(task, complete, tools, max_steps) -> Episode` -- so the pure
   loop, a LangGraph-compiled graph, and a CrewAI crew all return the SAME canonical `Episode`
   (final answer + tool-call transcript + final env-state). The existing `run_episode` is refactored
   to implement it with NO behavior change.
2. `build_agentic_graph` -- the LangGraph-compiled harness (mirroring `build_multi_hop_graph`) over
   the pure loop, so "LangGraph" is a named harness, not just the implicit substrate.
3. A CrewAI harness wrapper -- wrap the SAME `ToolWorld` tools as CrewAI tools and the SAME candidate
   `complete` as the crew's LLM, run a single-agent crew over the task, then adapt its result back
   into the canonical `Episode` so `check_success` + the scorer + the gated judge are UNCHANGED.
4. Record the harness id in the manifest (`harness: loop | langgraph | crewai`) and add it as a
   board axis under `TIER_AGENTIC` -- harness-tagged, never silently mixed (same discipline as the
   recorded backend capability); a comparison view ranks one model across `{langgraph, crewai}`.
5. Same isolation contract (`drive_with_backend` / `isolate_cell`), same bootstrap CIs; unit-test
   each harness from a fake endpoint (+ a fake crew), no GPU.

**Acceptance:** one model runs the SAME agentic task set under both the LangGraph and the CrewAI
harness, producing comparable completion-rate + trajectory-quality boards with the harness recorded
in the manifest; the objective scoring / isolation / gated judge are unchanged; CrewAI stays an
opt-in lazy extra so the base install is unaffected.

### M7.2 Non-blocking forward verification (quality gates -- no host)

Every forward verification provable WITHOUT a real CUDA host:
- **Harness board:** the M7.1 harness comparison ranks objectively under `TIER_AGENTIC` with
  bootstrap CIs from FAKE endpoints -- no GPU, harness-tagged (same discipline as the category
  boards).
- **Code-quality gate:** keep `make ci` green -- Ruff format + lint, mypy (strict), and the
  lightweight pytest group (`-m "not slow"`); the full suite (incl. `@pytest.mark.slow`) runs
  locally via `make test`. Every heavy dependency stays lazy-imported so the base install imports.
- **AGENTS.md guardrails:** paths under `.data/llb/`; ASCII logs; confirm the canonical `max_jobs()`
  helper (`scripts/shared/common.sh`) before any vLLM/llama.cpp source build.

### M7.3 Human-assisted RAG prompt-system generation and tuning

Add operator-facing tools that turn a supplied text corpus into candidate RAG prompt systems and
benchmark those systems across models and harnesses. The goal is to raise grounded-answer scores by
tuning the prompt template and its attached context package while preserving a fair, reproducible
measurement path.

**Scope:**
1. Corpus preparation command -- ingest a caller-provided text corpus and emit a compact anthology
   of selected passages, document metadata, and a knowledge-graph-to-RAG mapping artifact under
   `$DATA_DIR/<method_name>/<run_timestamp>/`.
2. Prompt-template generator -- create candidate system/additional prompts that embed the anthology,
   metadata summary, and graph/RAG mapping references in a structured template suitable for the
   existing RAG benchmark path.
3. Context-budget controller -- estimate per-model context windows and tokenizer costs, reserve
   space for the question, retrieved chunks, tool transcripts, and answer budget, then trim or
   summarize the anthology/metadata/graph sections so every prompt candidate fits the selected
   model context.
4. Human review loop -- expose the generated prompt candidates, budget breakdown, dropped-context
   report, and editable template fields so an operator can approve, revise, pin, or reject
   candidates before benchmarking.
5. Prompt-tuning loop -- search over prompt variants, metadata density, graph-reference style, and
   anthology size; keep all runs manifest-addressable so scores can be compared without mixing
   prompt systems.
6. Benchmark integration -- add a board axis for the prompt-system id and run the same corpus-backed
   RAG task set across selected models and harnesses, with CIs and manifests that record context
   budget, tokenizer, prompt template revision, corpus digest, and graph/RAG mapping digest.
7. Harness compatibility -- make the prompt package usable by the baseline RAG path and the
   agentic/harness comparison lane without changing objective scoring, so the benchmark can answer
   whether the additional system prompt helps a model, a harness, or both.

**Acceptance:** an operator can provide a corpus, generate and review prompt-system candidates with
bounded context size, run the same RAG benchmark across selected models/harnesses, and compare
scores by prompt-system id with all corpus, template, metadata, graph mapping, and context-budget
inputs recorded in the run artifacts.

### M7.4 Deferred / blocked (needs more hardware or a committed consumer)

Each unblocks differently:
- **Platform & matrix expansion (needs a committed consumer / more hardware).** Build last:
  - multi-backend comparison -- same model across vLLM / Ollama / llama.cpp (per-source quant
    metadata from M3.2 is the seam);
  - multi-vector-store -- Chroma / Qdrant / LanceDB behind the RAG-store seam (FAISS is v1);
  - full GPU-class matrix -- 12 / 24 / 48 GB beyond the validated 16 GB class (needs other GPUs);
  - quality-per-watt -- a derived metric over the NVML power already sampled per cell (M3.3).
