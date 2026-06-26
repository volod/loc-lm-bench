# loc-lm-bench -- Implementation Plan (forward work)

Forward-only. Everything DELIVERED -- Milestones 0-4 (live-validated on the CUDA host) and the
Milestone 5 BUILD (the eval-template + text-analysis-schema prerequisites and every scored
category: security / tooling / agentic / text-analysis / summarization / structured-output /
chat-period / reliability, plus the second-frontier verified-data gate) -- lives in
[`current.md`](current.md) and is NOT repeated here. Spec (source of truth):
[`docs/design/spec.md`](../design/spec.md).

**Quick start:** `make demo-eval` runs the pipeline end to end (needs a running Ollama); the real
vLLM path is `llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry` on a CUDA host.

Remaining work: optional Milestone 6 residuals (the M6 GraphRAG build lives in
[`current.md`](current.md)), the Milestone 7 extended-implementation + forward-verification lane,
and a human-only lane (Milestone H). (The Milestone 5 residuals -- per-category breadth + data-prep
hardening -- now live in [`current.md`](current.md), not here.)

---

## ⚠ HUMAN PREREQUISITES (irreducibly-human -- no AI substitute)

One gate still needs a human and CANNOT be done by GPT/Gemini/Claude: MH.5 eval-data verification.
It is human-paced, runs in PARALLEL with the build, and blocks specific outputs (below). All
drafting + cross-checking is already pipeline code; only the human sample-verify remains. (The MH.2
M6 ontology + scope sign-off is satisfied -- the signed schema is
[`docs/design/graph-ontology-schema.md`](../design/graph-ontology-schema.md), recorded in
[`current.md`](current.md). The judge-calibration gate is satisfied -- the gated judge is calibrated
and enabled per run with `JUDGE_RHO=`; details in [`current.md`](current.md).)

**The step-by-step manual for both is
[`docs/guides/human-in-the-loop-evaluation.md`](../guides/human-in-the-loop-evaluation.md)** -- it
has the procedure, the "done when", and the essential papers for each. Background learning paths:
[main](../guides/learning-path.md) ·
[security](../guides/learning-path-security.md) ·
[evaluation categories + GraphRAG](../guides/learning-path-evaluation-categories.md).

- **MH.5 data verification** ("Eval-data verification"): sample-verify a stratified sample of
  AI-drafted, cross-checked items, then flip via the ledger. Blocks any `verified=true`
  item SCORING REAL MODELS in any category (M5.1-M5.4 real runs, M6). This is now the ONLY open
  human gate (MH.2 is signed off; the OQ4 corpus facts are settled -- see `current.md`).

What is NOT human work (already automatable / built): schema/data DRAFTING, the second-frontier
cross-check, and the optional non-Gemma cross-check judge.

### Human-only ordered sequence (preferable order)

MH.5 runs PARALLEL with the remaining (M7 + optional M6 residual) work. It is NOT a one-shot upfront
task -- it verifies what the M5 producers (see [`current.md`](current.md)) DRAFT, per bundle:

1. **MH.5 first + continuous** (highest priority). It is the human CRITICAL PATH for the nearest
   deliverable -- real-model scores in the M5 categories (and M6 graph runs) -- and it is the slow,
   human-paced, PER-BUNDLE gate. Run it PULL-BASED: as each category's bundle is drafted +
   cross-checked and STABILIZES, verify that bundle, flip accepted items via the ledger, and
   real-model scoring unblocks for it. Start the moment the first drafted + cross-checked bundle
   lands; verify a bundle only once its drafting is stable (do not re-verify seeds a newer bundle
   supersedes). The objective boards never wait on this -- only real-model HEADLINE scoring does.
2. **(optional) Strengthen the judge calibration -- background, lowest priority.** The judge is
   already mechanically trusted and objective scores rank regardless; do it only with idle capacity.

(MH.2 -- the M6 ontology + scope sign-off -- is done; the OQ4 corpus facts + the OQ-egress
cross-check routing are settled -- see `current.md` -- so none of those remain human TODOs.)

### MH.5 -- gold/eval data verification (TODO, step by step)

Procedure + the four per-item checks:
[manual "Eval-data
verification"](../guides/human-in-the-loop-evaluation.md#eval-data-verification----human-sample-acceptance-of-ai-drafted-data).
1. Take a drafted bundle (`$DATA_DIR/prepare-goldset/<ts>/`, `verified=false`).
2. `make validate-goldset GOLDSET=<bundle>/goldset.jsonl CORPUS=<bundle>/corpus` (structural gate).
3. Draw a STRATIFIED sample (kind x difficulty x section x real/synthetic); document size + strata.
4. Verify each sampled item: grounded span / non-circular + answerable / correct reference / planted
   labels match the doc.
5. Accept if the error rate is within tolerance, else reject back to the pipeline.
6. Flip accepted items to `verified=true` THROUGH THE LEDGER (never hand-edit the boolean):
   `python -m llb.prep.ingest_squad ... --verified-goldset <accepted-ledger>`.

### (optional) Strengthen the judge calibration

The committed calibration is a borderline pass (its 95% CI dips below the 0.6 gate; see
[`current.md`](current.md)) because the SQuAD-uk calibration split is easy factual QA with little
human/judge disagreement to measure. To make the gate robust: add harder / ambiguous items and more
fluent-but-wrong candidate answers to the `calibration` split, then repeat the loop
(`make calibration-run` -> `calibration-rate` -> `calibration-score`) and re-commit the worksheet.
Optional -- the current judge is already mechanically trusted; objective scores rank regardless.

---

## Ordered Implementation Sequence

Canonical order for picking up the remaining work. Sequence numbers are stable workstream
identifiers (AGENTS.md); a workstream appears only while it has open work. (The Milestone 5
residuals -- per-category sourcing breadth, transports, gated-judge wiring, and the data-prep items
-- now live in [`current.md`](current.md), no longer open work here.)

1. **Milestone 6 residuals** (optional) -- the GraphRAG build lives in `current.md` (DuckDB store,
   both strategies, manifest-recorded) and MH.2 is signed off, so the schema is trusted for
   HEADLINE use. Only the optional improvements + the graph-vs-FAISS comparison remain (below);
   real-model graph headline numbers ride the standing MH.5 data gate, like every M5 category.
2. **Milestone 7** (parallel) -- extended + deferred + verification: the M7.1 LangGraph-vs-CrewAI
   harness comparison and the M7.2 non-blocking quality gates (both buildable now), plus M7.3 -- the
   DEFERRED / BLOCKED work moved out of M5 (human-gated calibrations, the composite headline, the
   platform/matrix expansion).
3. **Milestone H** (human-paced, parallel) -- MH.5 first + continuous (per-bundle); MH.2 is signed
   off. See the prerequisites block above (human-only ordered sequence).

Real-model scoring of any `verified=true` item still waits on MH.5 (the human gate); the objective
category boards do not depend on the gated judge.

---

## Milestone 6 residuals -- GraphRAG (optional, forward)

The M6 GraphRAG build lives in `current.md` (DuckDB store, `local_khop` + `global_community`,
manifest-recorded backend + strategy, tagged-diagnostic community summaries) and MH.2 is signed off
(the ontology + scope is trusted). Remaining work is optional improvement + comparison; the only
standing gate is MH.5 (it blocks real-model scoring of `verified=true` items, like every category).
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
   local-endpoint run); for real-model HEADLINE numbers it also rides the MH.5 data gate.

---

## Milestone 7 -- Extended, deferred, and forward-verification tasks

Everything NOT in the immediate M5 automatable sequence. Two parts are non-blocking and buildable
now -- the extended agentic harness comparison (M7.1) and the non-blocking quality-verification
actions (M7.2, provable in CI from fake endpoints). The third part (M7.3) parks the DEFERRED /
BLOCKED work that cannot be finished by AI alone: it needs human input, more hardware, or a
committed consumer. (Verifications that need a real CUDA host or a human gate stay in
`Verification (forward)`.)

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

### M7.2 Non-blocking forward verification (quality gates -- no host, no human)

Every forward verification that is provable WITHOUT a real CUDA host or a human gate lives here (the
blocked ones stay in `Verification (forward)` below):
- **Category + harness boards:** each M5 category and the M7.1 harness comparison ranks objectively
  under its OWN Tier with bootstrap CIs from FAKE endpoints -- no GPU, no human sample-verify (the
  real-model headline still rides the MH.5 data gate). See `current.md`.
- **Code-quality gate:** `make ci` stays green -- Ruff format + lint, mypy (strict), and the
  lightweight pytest group (`-m "not slow"`); the full suite (incl. `@pytest.mark.slow`) runs
  locally via `make test`. Every heavy dependency stays lazy-imported so the base install imports.
- **AGENTS.md guardrails:** paths under `.data/llb/`; ASCII logs; confirm the canonical `max_jobs()`
  helper (`scripts/shared/common.sh`) before any vLLM/llama.cpp source build.

### M7.3 Deferred / blocked (needs human input, more hardware, or a committed consumer)

Moved out of M5 because AI cannot finish them in sequence; each unblocks differently:
- **Domain-specific judge calibrations (human-gated).** Optional summarization-specific and
  agentic-specific judge calibrations: the wired faithfulness / trajectory-quality judges reuse the
  M3.8 rho (fit on SQuAD QA, not summaries / agent trajectories). Tightening them needs NEW HUMAN
  ratings over a harder split -- the same human loop as the "(optional) Strengthen the judge
  calibration" prerequisite (Milestone H).
- **Composite headline (blocked on MH.5).** Turn on the spec's weighted composite over the M5
  categories once every component carries a CI AND its data is MH.5-verified; until then each
  category reports its own board + CIs (the standing M5 constraint).
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

BLOCKED verifications only -- each needs a real CUDA host or a human gate. The non-blocking
quality-verification actions live in Milestone 7 (M7.2) above.
- **M5 (remaining):** the MH.5 human sample-verify gates real-model scoring.
- **M6 (build lives in `current.md`; CI-proven from fakes; MH.2 signed off):** what remains is the
  real-host graph-vs-FAISS comparison on the committed goldset (M6 residual #3 above -- needs a
  local-endpoint extraction run); real-model headline numbers ride the MH.5 data gate.
- **Milestone H (⚠ human):** a human sample-verify (MH.5) accepts the AI-drafted,
  frontier-cross-checked data before it scores models (MH.2 is already signed off). See
  [`human-in-the-loop-evaluation.md`](../guides/human-in-the-loop-evaluation.md).

## Worktree parallelization

- **extended-agentic:** Milestone 7 (M7.1 LangGraph vs CrewAI) is its own non-blocking lane over
  the M5.3 harness seam + task set; M7.2 collects the non-blocking quality gates. No human/host
  gate to build.
- **human-gated:** Milestone H (MH.5; MH.2 already signed off) runs on its own decision-paced lane.
