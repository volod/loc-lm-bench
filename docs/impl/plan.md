# loc-lm-bench -- Implementation Plan (forward work)

Forward-only. Everything DELIVERED -- Milestones 0-4 (live-validated on the CUDA host) and the
Milestone 5 BUILD (the eval-template + text-analysis-schema prerequisites and every scored
category: security / tooling / agentic / text-analysis / summarization / structured-output /
chat-period / reliability, the second-frontier verified-data gate, and the MH.5 sample-verify
tooling) -- lives in [`current.md`](current.md) and is NOT repeated here. Spec (source of truth):
[`docs/design/spec.md`](../design/spec.md).

**Quick start:** `make demo-eval` runs the pipeline end to end (needs a running Ollama); the real
vLLM path is `llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry` on a CUDA host.

**Operator workflows (not plan items).** Creating a new gold set, the second-frontier cross-check,
the MH.5 human sample-verify, and judge calibration are reproducible `make` workflows -- not open
milestones. When a future task needs human interaction (verify a new bundle, calibrate a judge on a
harder split), run the documented flow rather than tracking it here:
[create a gold set (end-to-end)](../guides/goldset-from-scratch.md) ·
[verification tooling](../guides/verification-tooling.md) ·
[calibration tooling](../guides/calibration-tooling.md). The committed `ua_squad_postedited_v1` set
is already verified + calibrated; the M6 ontology + scope sign-off is signed (the schema is
[`graph-ontology-schema.md`](../design/graph-ontology-schema.md), recorded in [`current.md`](current.md)).

Remaining work is engineering only: optional Milestone 6 residuals (the M6 GraphRAG build lives in
[`current.md`](current.md)) and the Milestone 7 extended-implementation + forward-verification lane.

---

## Ordered Implementation Sequence

Canonical order for picking up the remaining work. Sequence numbers are stable workstream
identifiers (AGENTS.md); a workstream appears only while it has open work. (The Milestone 5
residuals -- per-category sourcing breadth, transports, gated-judge wiring, and the data-prep items
-- now live in [`current.md`](current.md), no longer open work here.)

1. **Milestone 6 residuals** (optional) -- the GraphRAG build lives in `current.md` (DuckDB store,
   both strategies, manifest-recorded) and the ontology + scope is signed off, so the schema is
   trusted for HEADLINE use. Only the optional improvements + the graph-vs-FAISS comparison remain
   (below).
2. **Milestone 7** (parallel) -- extended + deferred + verification: the M7.1 LangGraph-vs-CrewAI
   harness comparison and the M7.2 non-blocking quality gates (both buildable now), plus M7.3 -- the
   deferred work moved out of M5 (the composite headline, the platform/matrix expansion, optional
   domain-specific judge calibrations).

A NEW gold set's real-model HEADLINE scoring first runs the
[data-verification workflow](../guides/goldset-from-scratch.md); the objective category boards do
not depend on the gated judge.

---

## Milestone 6 residuals -- GraphRAG (optional, forward)

The M6 GraphRAG build lives in `current.md` (DuckDB store, `local_khop` + `global_community`,
manifest-recorded backend + strategy, tagged-diagnostic community summaries) and the ontology +
scope is signed off (the schema is trusted). Remaining work is optional improvement + comparison.
Concepts: [evaluation-categories learning path](../guides/learning-path-evaluation-categories.md).

1. **Morphology-aware entity linking (optional).** The v1 entity-linker (`llb.graph.retrieval`)
   matches exact casefolded tokens of a node's name + aliases, so a Ukrainian inflected question
   form (e.g. "Франка" genitive vs the "Франко" node) can miss the link and return empty. Why it
   helps: it lifts graph recall on morphologically-varied questions toward the FAISS path's level
   without an embedder. Rough how: add lemma- or prefix-aware matching, or reuse the pinned embedder
   to embed node names + the question and link by cosine (the NetworkX-fallback "FAISS entity-link
   vectors" idea), kept behind the existing pure-linking seam.
2. **CLI flag for diagnostic community summaries (optional).** `summarize_communities` exists and is
   tested but is only reachable programmatically. Why it helps: lets an operator attach the
   narrative summaries during `build-graph`. Rough how: add `--summarize` + an endpoint to
   `build-graph`, reusing the M4.4 endpoint adapter; keep the summaries tagged-diagnostic (never
   span-scored).
3. **Graph-vs-FAISS retrieval comparison on the committed goldset (verification).** Build the graph
   from an M4.4 extraction over the committed corpus, then compare `recall@k`/`MRR` and answer
   quality across `{faiss, graph/local_khop, graph/global_community}` on the same goldset (the
   manifest already records backend + strategy). Why it helps: quantifies when the multi-hop /
   narrative paths beat flat vector retrieval. Needs the M4.4 extraction over the corpus (a real
   local-endpoint run); the committed goldset is already verified, so HEADLINE scoring needs no
   further data gate (a NEW corpus would first run the
   [data-verification workflow](../guides/goldset-from-scratch.md)).

---

## Milestone 7 -- Extended, deferred, and forward-verification tasks

Everything NOT in the immediate M5 automatable sequence. Two parts are non-blocking and buildable
now -- the extended agentic harness comparison (M7.1) and the non-blocking quality-verification
actions (M7.2, provable in CI from fake endpoints). The third part (M7.3) parks the deferred work
that needs more hardware, a committed consumer, or a separate calibration pass. (Verifications that
need a real CUDA host stay in `Verification (forward)`.)

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

Every forward verification that is provable WITHOUT a real CUDA host lives here (the host-blocked
ones stay in `Verification (forward)` below):
- **Category + harness boards:** each M5 category and the M7.1 harness comparison ranks objectively
  under its OWN Tier with bootstrap CIs from FAKE endpoints -- no GPU (a NEW gold set's real-model
  headline first runs the [data-verification workflow](../guides/goldset-from-scratch.md)). See
  `current.md`.
- **Code-quality gate:** `make ci` stays green -- Ruff format + lint, mypy (strict), and the
  lightweight pytest group (`-m "not slow"`); the full suite (incl. `@pytest.mark.slow`) runs
  locally via `make test`. Every heavy dependency stays lazy-imported so the base install imports.
- **AGENTS.md guardrails:** paths under `.data/llb/`; ASCII logs; confirm the canonical `max_jobs()`
  helper (`scripts/shared/common.sh`) before any vLLM/llama.cpp source build.

### M7.3 Deferred / blocked (needs more hardware, a committed consumer, or a separate calibration)

Moved out of M5 because they cannot be finished in the immediate sequence; each unblocks differently:
- **Domain-specific judge calibrations (optional).** The summarization-specific and agentic-specific
  judges reuse the M3.8 rho (fit on SQuAD QA, not summaries / agent trajectories). Tightening them
  means calibrating on a harder split for those tasks -- run the
  [calibration workflow](../guides/calibration-tooling.md) over that split; the wired faithfulness /
  trajectory-quality judges already rank mechanically, so this is a refinement, not a blocker.
- **Composite headline.** Turn on the spec's weighted composite over the M5 categories once every
  component carries a CI AND its gold data is verified (the
  [data-verification workflow](../guides/goldset-from-scratch.md)); until then each category reports
  its own board + CIs (the standing M5 constraint).
- **Platform & matrix expansion (needs a committed consumer / more hardware).** Build last:
  - multi-backend comparison -- same model across vLLM / Ollama / llama.cpp (per-source quant
    metadata from M3.2 is the seam);
  - multi-vector-store -- Chroma / Qdrant / LanceDB behind the RAG-store seam (FAISS is v1);
  - full GPU-class matrix -- 12 / 24 / 48 GB beyond the validated 16 GB class (needs other GPUs);
  - quality-per-watt -- a derived metric over the NVML power already sampled per cell (M3.3).

---

## Reuse (do not rebuild)

DeepEval G-Eval, FAISS, sentence-transformers, `openai` client (local backends incl. tool/function
calling for M5.2), litellm (frontier prep), Optuna, MLflow (local), LangGraph (eval templates incl.
the M5.3 agentic loop), DuckDB (also the M6 GraphRAG store -- recursive-CTE k-hop +
community grouping over node/edge JSONL, `[graph]` extra; see `current.md`), Streamlit, pynvml +
psutil, lm-evaluation-harness-uk (Tier-1 screen), CrewAI (M7, the opt-in second agent harness --
lazy extra), the
official `mcp` Python SDK (M5.2 MCP transport, `[mcp]` extra), spaCy `uk_core_news` (M5.6 opt-in
extraction adapter). Public UA datasets: SQuAD-uk + Belebele-uk; UA-adapted public sets feed the M5
adapters (BFCL for tooling; JailbreakBench / HarmBench / AdvBench for security). No servers
(no Celery/K8s/Neo4j), no cloud dependence.

## Verification (forward)

Host-BLOCKED verifications only -- each needs a real CUDA host. The non-blocking
quality-verification actions live in Milestone 7 (M7.2) above.
- **M6 (build lives in `current.md`; CI-proven from fakes; ontology signed off):** what remains is
  the real-host graph-vs-FAISS comparison on the committed goldset (M6 residual #3 above -- needs a
  local-endpoint extraction run).
- **A NEW gold set** runs the [data-verification workflow](../guides/goldset-from-scratch.md) before
  its `verified=true` items score real models; the committed `ua_squad_postedited_v1` set is already
  verified, so it needs no further data gate.

## Worktree parallelization

- **extended-agentic:** Milestone 7 (M7.1 LangGraph vs CrewAI) is its own non-blocking lane over
  the M5.3 harness seam + task set; M7.2 collects the non-blocking quality gates. No host gate to
  build.
