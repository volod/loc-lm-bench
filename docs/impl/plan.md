# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every line is open work. What already exists (behavior + results), and the operator
workflows an operator re-runs as needed (new gold set, cross-check, sample-verify, calibration, the
verified-data gate, graph-vs-FAISS comparison), live in [`current.md`](current.md); the spec (source
of truth) is [`docs/design/spec.md`](../design/spec.md).

The one open workstream is **Milestone 7** (extended + deferred + forward-verification). Its
sequence number is a stable identifier (AGENTS.md); it appears only while it has open work.

---

## Milestone 7 -- Extended, deferred, and forward-verification tasks

Two parts are non-blocking and buildable now in their own worktree lane (no host gate) -- the
extended agentic harness comparison (M7.1) and the non-blocking quality-verification actions (M7.2,
provable in CI from fake endpoints). The third part (M7.3) parks deferred work that needs more
hardware, a committed consumer, or a separate calibration pass.

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

### M7.3 Deferred / blocked (needs more hardware, a committed consumer, or a separate calibration)

Each unblocks differently:
- **Domain-specific judge calibrations (optional).** The summarization-specific and agentic-specific
  judges reuse the M3.8 rho (fit on SQuAD QA, not summaries / agent trajectories). Tightening them
  means calibrating on a harder split for those tasks -- run the
  [calibration workflow](../guides/calibration-tooling.md) over that split; the wired faithfulness /
  trajectory-quality judges already rank mechanically, so this is a refinement, not a blocker.
- **Composite headline.** Turn on the spec's weighted composite over the M5 categories once every
  component carries a CI AND its gold data is verified (the
  [data-verification workflow](../guides/goldset-from-scratch.md)); until then each category reports
  its own board + CIs.
- **Platform & matrix expansion (needs a committed consumer / more hardware).** Build last:
  - multi-backend comparison -- same model across vLLM / Ollama / llama.cpp (per-source quant
    metadata from M3.2 is the seam);
  - multi-vector-store -- Chroma / Qdrant / LanceDB behind the RAG-store seam (FAISS is v1);
  - full GPU-class matrix -- 12 / 24 / 48 GB beyond the validated 16 GB class (needs other GPUs);
  - quality-per-watt -- a derived metric over the NVML power already sampled per cell (M3.3).
