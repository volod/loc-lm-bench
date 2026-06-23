# loc-lm-bench -- Implementation Plan (forward work)

Forward-only. Everything DELIVERED -- Milestones 0-4 (live-validated on the CUDA host), the M5.0
prerequisites (text-analysis schema + map-reduce/multi-hop templates), and the MH.2 text-analysis
schema sign-off -- lives in [`current.md`](current.md) and is NOT repeated here. Spec (source of
truth): [`docs/design/spec.md`](../design/spec.md).

**Quick start:** `make demo-eval` runs the pipeline end to end (needs a running Ollama); the real
vLLM path is `llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry` on a CUDA host.

Remaining work: two implementation milestones (M5 -> M6) plus a human-only lane (Milestone H).

---

## ⚠ HUMAN PREREQUISITES (irreducibly-human -- no AI substitute)

Three gates need a human and CANNOT be done by GPT/Gemini/Claude. They are human-paced and run in
PARALLEL with the build, but they block specific outputs (below). All drafting + cross-checking is
already pipeline code; only the human ground-truth, sample-verify, and sign-off remain.

**The step-by-step manual for all three is
[`docs/guides/human-in-the-loop-evaluation.md`](../guides/human-in-the-loop-evaluation.md)** -- it
has the procedure, the "done when", and the essential papers for each. Background learning paths:
[main](../guides/learning-path.md) ·
[security](../guides/learning-path-security.md) ·
[evaluation categories + GraphRAG](../guides/learning-path-evaluation-categories.md).

- - **Stage **M3.8 judge calibration** -- Fill `human_rating` over the 86 calibration items, then
- score rho** ("Judge calibration"): EVERY judged headline (RAG board + M5 unsafe-content quality,
- summarization faithfulness, agentic trajectory, free-form text/chat analysis). Objective metrics
- rank alone meanwhile. **Critical path -- start EARLY.**
- - **Stage **MH.2 sign-offs + corpus facts** -- Approve the M6 ontology schema + M6 scope; confirm
- OQ4 corpus facts (do TA reference answers exist? real vs synthetic)** ("Schema and ontology
- sign-off"): Milestone 6 (ontology)
- - **Stage **MH.5 data verification** -- Sample-verify a stratified sample of AI-drafted,
- frontier-cross-checked items, then flip via the ledger** ("Eval-data verification"): Any
- `verified=true` item SCORING REAL MODELS in any category (M5.1-M5.4 real runs, M6)


What is NOT human work (already automatable / built): schema/data DRAFTING, the second-frontier
cross-check, and the optional non-Gemma cross-check judge.

### M3.8 -- judge calibration (TODO, step by step)

Scaffolding (stats, gate, worksheet pre-fill, scoring) is built + tested; only the human column
remains. Full procedure + rules:
[manual "Judge
calibration"](../guides/human-in-the-loop-evaluation.md#judge-calibration----validating-llm-as-judge-against-human-ratings).
1. Stand up a judge endpoint (12B judge can't co-reside with a vLLM candidate on 16 GB -- use
   GGUF/CPU offload, a smaller test judge, or another host). See
   [judge-experiments guide](../guides/judge-experiments.md).
2. `make calibration-run JUDGE_MODEL=<id> JUDGE_BASE_URL=http://127.0.0.1:8000/v1` -- pre-fills
   `model_answer` + ungated `judge_rating`.
3. Fill `human_rating` INDEPENDENTLY (hide `judge_rating` first), spanning the full range and
   deliberately including fluent-but-wrong answers.
4. `make calibration-score RATINGS=<filled.csv>` -> rho + bootstrap CI + the mechanical decision.
   `rho >= 0.6` admits the gated judge; else it stays demoted. The decision travels in the manifest.

### MH.2 -- remaining sign-offs (TODO, step by step)

Procedure + template:
[manual "Schema and ontology
sign-off"](../guides/human-in-the-loop-evaluation.md#schema-and-ontology-sign-off----accountable-approval).
1. When the M6 ontology draft lands, read it + its executable form; confirm the node/relationship
   type set, cap sizes, extraction constraints; record a dated sign-off line at the TOP of the
   proposal doc (until that line exists the schema stays un-trusted for headline use).
2. Approve the Milestone 6 scope / acceptance.
3. Confirm the OQ4 corpus facts only you have: whether text-analysis reference answers already
   EXIST or must be authored, and which corpus is real vs synthetic (reported separately, never
   merged).

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

---

## Ordered Implementation Sequence

Canonical order for picking up work. Sequence numbers are stable workstream identifiers (AGENTS.md);
keep them even as bodies shrink.

1. **Milestone 5.** Each category is its own Tier, never cross-ranked with the RAG board.
   1. **M5.0** residual -- the M5.4 emit/runner wiring only (prerequisites delivered + signed off).
   2. **M5.1** Security / robustness -- objective ASR; no human dep to BUILD.
   3. **M5.2** Tooling / MCP / function-calling -- objective; no human dep to BUILD.
   4. **M5.3** Agentic workflows -- builds on M5.0.
   5. **M5.4** Remaining taxonomy (summarization, structured output, chat-period [needs M5.0],
      reliability).
   6. **M5.5** Platform & matrix expansion -- optional; build last (needs a committed consumer).
   7. **M5.6** Carried-forward M4 residuals -- small run-path + data-prep hardening; rides whichever
      M5 lane first touches the host / the draft pipeline.
2. **Milestone 6** (after M5) -- GraphRAG (Kuzu). ⚠ needs MH.2 (M6 ontology + scope sign-off).
3. **Milestone H** (human-paced, parallel) -- M3.8, MH.2, MH.5. See the prerequisites block above.

**Objective metrics ship first.** M5.1 ASR, M5.2 call-correctness, M5.4 structured-output, and
retrieval recall do NOT depend on M3.8 and can land before calibration. Real-model scoring of any
`verified=true` item still waits on MH.5.

---

## Milestone 5 -- security, agentic, and tooling benchmark

Un-defers the next spec taxonomy categories (Appendix D). Principle: REUSE, not a new platform --
each category is a new TASK FAMILY on the existing substrate (LangGraph templates, `isolate_cell`,
`rank_board` with its Tier guard + CIs, the canonical manifest + per-case scores, Streamlit/MLflow).

**Cross-cutting rules for ALL M5 categories:**
- **New Tier per category, never cross-ranked.** Add `TIER_SECURITY` / `TIER_TOOLING` /
  `TIER_AGENTIC` (alongside `TIER_SCREEN` / `TIER_PRIVATE`) to the `aggregate` Tier guard.
- **Objective first, gated judge second.** Plant STRUCTURED ground-truth labels (M3.5
  `prepare-synthetic-corpus`); the gated judge (M3.8) enters only for residual free-form quality and
  only when trusted. Recovery scored on planted-label IDs + embedder-cosine (secondary).
- **Verified-data gate.** Every gold/eval item is AI-drafted (M4.4 / planter),
  frontier-cross-checked
  in-pipeline, then ⚠ human sample-verified (MH.5) before `verified=true` scores models.
- **Same isolation contract.** All runs go through `isolate_cell` (process per cell, PID-attributed
  VRAM reclaim gate, capped cooldown).
- **Record backend capability, don't assume it.** Tool-calling / logprob support varies by backend;
  record per-candidate and never cross-rank capable vs not.
- **Composite stays off until calibrated.** The spec default weights are recorded but NOT a headline
  until every component carries a CI; until then each category reports its own Pareto + CIs.

### M5.0 residual -- text-analysis emit/runner wiring
Prerequisites DELIVERED + schema SIGNED OFF (see `current.md`). Remaining:
- extend `prepare-synthetic-corpus` to emit the richer per-kind planted labels (today: QA-style
  `key_fact` only);
- build the scored runner + a `TIER_TEXT_ANALYSIS` board guard (mirroring
  `TIER_SCREEN`/`TIER_PRIVATE`);
- use a trend label's `attrs.direction` for direction-aware credit (surface-only today).

### M5.1 Security / robustness benchmark (objective ASR; no human dep to build)
Subcategories (spec Appendix D): prompt-injection, jailbreak, instruction-hierarchy violation,
unsafe-content generation, tool-abuse, RAG-injection (malicious instructions in retrieved chunks),
data-exfiltration resistance (corpus-secret / canary leakage).
- **Sourcing (decided, hybrid):** reuse public sets (JailbreakBench / HarmBench / AdvBench)
  UA-adapted
for the generic families; the M3.5 planter for the corpus-specific RAG-injection + canary families.
  Every malicious instruction + canary is a STRUCTURED label.
- Objective detector per family: planted-instruction-followed / canary-leaked match -> per-case
  binary -> ASR (lower better) + refusal-appropriateness (don't over-refuse benign UA prompts).
- Unsafe-content family: same objective match PLUS the gated judge for borderline quality only (no
  new safety classifier; ShieldGemma / frontier moderation stay opt-in).
- Reuse `isolate_cell`, manifest, `rank_board` under `TIER_SECURITY`.
- **Acceptance:** each family's detector unit-tested with planted fixtures; a fake endpoint proves
  the flow; ASR + refusal-appropriateness carry CIs; never cross-ranked with the RAG board.
- Deep dive: [security learning path](../guides/learning-path-security.md).

### M5.2 Tooling / MCP / function-calling benchmark (objective; no human dep to build)
- **Dataset (decided):** adapt the Berkeley Function-Calling Leaderboard (BFCL) cases to Ukrainian
  (OpenAI tool/function-calling JSON schema); serve the SAME catalog via the official `mcp` Python
  SDK server so native FC and MCP transports run from one source.
- Cases map a UA instruction -> expected tool name + argument JSON, scored objectively and
  **call-only** (validate the emitted call; tools are NOT executed here -- execution is M5.3):
  tool-selection accuracy, argument-exactness (schema-valid + value match), no-hallucinated-tool
  rate, well-formed-call rate.
- Tool-call parse/validate layer over the existing OpenAI-compatible client; record per-backend
  tool-call capability; never cross-rank tool-capable vs text-only.
- **Acceptance:** schema validation + scoring pure + unit-tested; a fake endpoint with canned tool
  calls proves the flow; per-backend capability recorded under `TIER_TOOLING`.

### M5.3 Agentic workflows benchmark (builds on M5.0)
- The agentic loop = the M5.0 multi-hop LangGraph template extended with tool calls + a controller
  node.
- **Environment (decided):** a small in-memory DETERMINISTIC tool-world (mock files/DB + search over
  the UA corpus + a calculator), tools EXECUTED in-sandbox (not tau-bench / AgentBench).
- A small task set; objective completion-rate from env-state / planted-label assertions; the gated
  judge scores only trajectory quality where a deterministic check cannot.
- LangGraph is the single fixed agent harness; the other five frameworks stay deferred as a
  comparison axis (M5.3 ranks the MODEL under one harness, not frameworks -- out of M5 scope).
- Run through `isolate_cell` (longer loops -- keep the thermal/VRAM gate); record trajectory length
  +
  tool-call count as efficiency. New `TIER_AGENTIC`.
- **Acceptance:** task environment + success checks unit-tested with a fake tool-calling endpoint;
  completion-rate carries CIs; efficiency metrics recorded.

### M5.4 Remaining taxonomy (summarization, structured output, chat-period, reliability)
- **summarization** -- reference coverage via pinned-embedder cosine (not ROUGE) + gated-judge
  faithfulness.
- **structured output** -- objective JSON-schema conformance + field accuracy via Pydantic (no new
  `jsonschema` dep).
- **chat-period analysis** -- the text-analysis sibling over chat logs; depends on M5.0 + the
  `prepare-synthetic-corpus` planted labels; real-corpus and synthetic results reported SEPARATELY.
- **reliability** -- aggregate the existing typed failure taxonomy
(empty/malformed/refusal/timeout/context-truncation/retrieval-miss/backend-crash/OOM/judge-failure)
  into a first-class reliability score.
- **Acceptance:** each category scores on a fixed seeded set with CIs; the full composite weights
  activate only once all components carry CIs.

### M5.5 Platform and matrix expansion (deferred within M5)
- multi-backend comparison -- same model across vLLM / Ollama / llama.cpp (per-source quant metadata
  from M3.2 is the seam);
- multi-vector-store -- Chroma / Qdrant / LanceDB behind the RAG-store seam (FAISS is v1);
- full GPU-class matrix -- 12 / 24 / 48 GB beyond the validated 16 GB class;
- quality-per-watt -- a derived metric over the NVML power already sampled per cell (M3.3).

### M5.6 Carried-forward M4 residuals (small code hardening; prerequisites done)
Run-path items land with whichever lane first sweeps the 16 GB host; data-prep items land before any
`verified=true` item scores models (the verified-data gate) and before the M6 extraction reuse.
- **Run-path:**
  1. M4.1 -- model Gemma 3/4 sliding-window KV (full-attention today); let a cached `config.json`
     OVERRIDE curated arch fields, not only fill gaps.
  2. M4.2 -- read all GPUs (guard reads GPU 0 only); derive the KV abort headroom from the served
     arch instead of the fixed floor.
  3. M4.3 -- auto-pin a host-compatible flashinfer when the bundled one fails; record the chosen
     sampler in the manifest; re-run the preflight on a driver change without a
     full rebuild.
  4. M4.5 -- handle further `/props` response shapes; exercise a real partial-offload split on an
     oversized GGUF (only the all-on-GPU path is confirmed).
- **Data-prep (M4.4; feeds the verified-data gate + the M6 extraction reuse):**
  1. Wire the second-frontier cross-check (grounding / non-circularity) as pipeline code -- it IS
     the
     verified-data gate; lands with M5's first scored category.
  2. Ship the opt-in Stanza / spaCy `uk_core_news` `ExtractionAdapter` plug-in (seam exists).
  3. Chunk over-long docs for extraction instead of one truncated call (`EXTRACT_MAX_CHARS`).
  4. Induce ontology-type confidence from a richer signal than raw frequency; carry the induced
     types
     into the drafting prompt as explicit constraints.

**M5 acceptance:** security + tooling produce objective, CI-bearing boards from fake endpoints with
no human dependency; agentic + chat-period build cleanly on M5.0; every category renders under its
own Tier, never cross-ranked with the RAG board; M5.6 run-path validations pass on the first real
host sweep.

---

## Milestone 6 -- GraphRAG (knowledge-graph RAG)

⚠ **Blocked on MH.2** (human sign-off of the AI-drafted ontology schema + the M6 scope) -- see the
prerequisites block. GO decided; an ADDED retrieval backend behind the RAG-store seam, FAISS stays
default. Architecture locked.

**Decided architecture:** graph store **Kuzu** (embedded, Apache-2.0 property graph, Cypher,
pip-install, native vector index); construction REUSES M4.4 extraction (no second extraction
framework); extraction LLM local by default, frontier opt-in via the M4.4 endpoint adapter.

Tasks:
1. A Kuzu-backed graph store behind the RAG-store seam, swappable via `--retrieval-backend graph`;
   ingest M4.4 extraction into nodes/edges keeping `doc_id` + char offsets.
2. Apply the AI-drafted, ⚠ human-signed-off (MH.2) constrained node/relationship ontology schema.
3. A graph-retrieval layer -- entity-link the question, expand k-hops, serialize the subgraph as
   context PRESERVING source spans so the M1.3 span metric still applies.
4. Record the retrieval backend in the manifest so graph-vs-FAISS runs are comparable.
5. Reuse the eval graph, scoring, isolation, and board unchanged.

**Acceptance:** a corpus builds a Kuzu graph from M4.4 extraction; graph retrieval returns
offset-bearing context that scores on the existing span metric; runs are reproducible +
manifest-recorded; the FAISS path is unchanged. Concepts:
[evaluation-categories learning path](../guides/learning-path-evaluation-categories.md).

---

## Reuse (do not rebuild)

DeepEval G-Eval, FAISS, sentence-transformers, `openai` client (local backends incl. tool/function
calling for M5.2), litellm (frontier prep), Optuna, MLflow (local), LangGraph (eval templates incl.
the M5.3 agentic loop), DuckDB, Streamlit, pynvml + psutil, lm-evaluation-harness-uk (Tier-1
screen), Kuzu (M6 graph store). Public UA datasets: SQuAD-uk + Belebele-uk. For M5: the official
`mcp` Python SDK (M5.2), BFCL cases (M5.2), and JailbreakBench / HarmBench / AdvBench (M5.1), all
UA-adapted. No servers (no Celery/K8s/Neo4j), no cloud dependence.

## Verification (forward)

- **M5:** security + tooling produce objective CI-bearing boards from fake endpoints (no human dep);
  agentic + chat-period build cleanly on M5.0; every category renders under its own Tier; the M5.6
  run-path validations pass on the first real CUDA-host sweep.
- **M6:** a corpus builds a Kuzu graph from M4.4 extraction and graph retrieval scores on the
  existing source-span metric, FAISS unchanged.
- **Milestone H (⚠ human):** M3.8 produces rho/CI over HUMAN ratings; the M6 ontology is signed off
  (MH.2); a human sample-verify (MH.5) accepts the AI-drafted, frontier-cross-checked data before it
  scores models. See [`human-in-the-loop-evaluation.md`](../guides/human-in-the-loop-evaluation.md).
- **AGENTS.md guardrails:** paths under `.data/llb/`; ASCII logs; confirm the canonical `max_jobs()`
  helper (`scripts/shared/common.sh`) before any vLLM/llama.cpp source build.

## Worktree parallelization

- **benchmark categories:** M5.1 (security) + M5.2 (tooling) are objective and parallelize now;
  M5.3 (agentic) + M5.4 chat-period build on M5.0.
- **M5.6 residuals:** attach to whichever M5 lane first touches the host / the draft pipeline.
- **graph:** Milestone 6 is its own lane after M5, reusing M4.4 extraction.
- **human-gated:** Milestone H (M3.8, MH.2, MH.5) runs on its own decision-paced lane.
