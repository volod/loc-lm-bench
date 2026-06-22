# loc-lm-bench -- Implementation Plan (forward work)

## Context

loc-lm-bench is a production-grade, reproducible internal tool that selects the best
open-weight LLM for Ukrainian RAG + text analysis on a single small desktop GPU (RTX
4060 Ti 16 GB). It re-ranks ~6-10 models on *their* data and *their* GPU so the pick is
defensible.

Full spec (source of truth, do not duplicate here): [`docs/design/spec.md`](../design/spec.md).

Milestones 0, 1, 2, and 3 are **delivered** and documented in [`current.md`](current.md): the
gold set + data-prep tooling (M0); the eval skeleton (compile-free: prebuilt Ollama, no
vLLM/flash-attn source build) + model prep / feasibility tooling (M1); one real vLLM backend +
steady-state telemetry, validated end to end on `gemma-4-E4B-it-w4a16` on the RTX 4060 Ti (M2);
and the two-tier + scale + rigor layer (M3) -- backend resolution, process-isolated resumable
sweeps, two-stage RAG tuning, public screen, frontier prep, N-model board, and the maintained
DeepEval judge engine. The only M3 residual is human-gated (judge calibration ratings), now
tracked in the human-action milestone below.

**Quick start:** `make demo-eval` runs the current pipeline end to end and idempotently
(venv -> gold set -> index -> prep-models -> run-eval + telemetry; needs a running Ollama).
The real vLLM path is `llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry` on a
CUDA host. See [`current.md`](current.md) for the per-command breakdown.

This file is the FORWARD plan only. It is organized into three implementation milestones
(M4 -> M5 -> M6) plus a human-only lane (Milestone H), sequenced by dependency:

- **Milestone 4 -- robustness + ontology data prep + third backend.** The next implementable
  step: non-blocking improvements (embedding-aware VRAM estimates, a pre-launch contention
  guard, vLLM serving ergonomics), the ontology-assisted corpus drafting pipeline, and the
  llama.cpp launcher (the third backend the resolver already routes to). No human gating; each
  item is independently shippable and unit-testable.
- **Milestone 5 -- security, agentic, and tooling benchmark.** The next eval categories,
  un-deferred from the spec taxonomy and designed in detail below. New task families (security
  / robustness, tooling-MCP / function-calling, agentic workflows, plus the remaining
  summarization / structured-output / chat-period / reliability categories), each with its own
  scoring schema, reusing the M3 isolation + board + manifest infra. Builds on M4; its
  prerequisites (M5.0: the AI-drafted text-analysis schema + the eval templates) are
  implementation, not human work.
- **Milestone 6 -- GraphRAG (knowledge-graph RAG).** GO decided (2026-06-22): an ADDED
  retrieval backend (Kuzu graph store + reuse of M4.4 extraction + a thin graph-retrieval layer)
  behind the existing RAG-store seam. Architecture fully decided; builds after M5.
- **Milestone H -- irreducibly-human tasks (no AI substitute).** Everything an AI service could
  do (schema drafting, data drafting, frontier cross-checking) lives in M4-M6. H keeps ONLY what
  GPT / Gemini / Claude cannot legitimately do: human ground-truth calibration ratings, human
  sample-verification of AI-drafted data, and human sign-off / scope approval. Human-paced,
  parallel to M4-M6.

## Approach: walking skeleton, then layer

The end-to-end vertical is proven on real backends through M3 (`current.md`). The forward
layers add robustness + a third backend (M4), broaden what is measured (M5), then add the
knowledge-graph retrieval backend (M6). The human-only lane (Milestone H) -- calibration
ratings, sample-verification, and sign-offs -- proceeds in parallel for BUILD, but M3.8
calibration is on the CRITICAL PATH for any judged metric (see the Ordered Implementation
Sequence); the design / drafting work H used to hold is now AI-implementable inside M4-M6.

## Ordered Implementation Sequence

Canonical order for picking up work, with cross-item dependencies. Sequence numbers are stable
workstream identifiers; keep them even as item bodies shrink to residual notes (AGENTS.md).

1. **Milestone 4 (no human gating; start here).** Run-path items are kept sequential because
   they share the launch/planner path; the CLI and prep items parallelize.
   1. **M4.1** Embedding-aware VRAM estimate -- DONE (refined the M3.2 resolver / planner).
   2. **M4.2** Pre-launch VRAM-contention guard -- DONE (auto-derate + evict/wait; vLLM launch path).
   3. **M4.5** llama.cpp launcher -- DONE (LlamaCppLauncher + telemetry + reclaim gate; run path).
   4. **M4.3** vLLM serving knobs as CLI flags + kernel preflight -- CLI-only; parallelizable.
   5. **M4.4** Ontology-assisted gold-set drafting -- independent prep subpackage; parallelizable.

2. **Milestone 5 (after M4).** Each category is its own Tier and is never cross-ranked with the
   RAG board. M5.0 prerequisites are AI-implementable (no human gating); only the schema
   sign-off is human (Milestone H).
   1. **M5.0** Prerequisites -- AI-drafted text-analysis scoring schema (human sign-off via MH.2)
      + map-reduce / multi-hop eval templates (M1.4-rest). Unblocks M5.3 + M5.4 chat-period.
   2. **M5.1** Security / robustness benchmark -- objective ASR scoring; no human dep.
   3. **M5.2** Tooling / MCP / function-calling benchmark -- objective; no human dep.
   4. **M5.3** Agentic workflows benchmark -- needs M5.0.
   5. **M5.4** Remaining taxonomy (summarization, structured output, chat-period [needs M5.0],
      reliability).
   6. **M5.5** Platform & matrix expansion -- optional; no committed consumer; build last.

3. **Milestone 6 (after M5; GraphRAG, GO decided).** Kuzu graph store + reuse of M4.4 extraction
   + a thin graph-retrieval layer behind the RAG-store seam. Needs the AI-drafted M6 ontology
   schema signed off (MH.2).

4. **Milestone H (human-paced; parallel for BUILD; no AI substitute).**
   1. **M3.8** Judge calibration -- human ratings (decided human-only). CRITICAL PATH for any
      judged metric (see below).
   2. **MH.2** Sign-offs + corpus facts -- approve the AI-drafted TA schema (M5.0) + the M6
      ontology + the M6 scope; confirm the OQ4 corpus facts.
   3. **MH.5** Gold/eval data verification -- human sample-verify of AI-drafted, frontier-cross-
      checked items before they score models.
   (Resolved 2026-06-22: MH.4 GraphRAG go/no-go -> GO -> Milestone 6; M1.4-rest -> M5.0.)

**Critical path (judged metrics).** The judge quality axis is GATED on M3.8: until the 86-item
human rating pass clears rho `>= 0.6`, the gated judge is demoted and OBJECTIVE correctness ranks
alone -- on the RAG board AND on every M5 category that uses the judge (borderline unsafe-content
in M5.1, summarization faithfulness, agentic trajectory quality, and the free-form parts of
text-analysis / chat-period). So M4-M6 BUILD in parallel with the human lane, but no judged
headline is trustworthy until M3.8 lands. Start the rating pass EARLY -- it needs only the
M3-era calibration scaffolding (already shipped, `current.md`), so it should not become the thing
everything waits on at the end. The purely objective metrics (M5.1 ASR, M5.2 call-correctness,
M5.4 structured-output, retrieval recall) do NOT depend on M3.8 and can ship first.

## Milestone 4 -- robustness + ontology data prep + third backend

M4.1-M4.3 are non-blocking improvements surfaced by the M2.4 real-model run
(gemma-4-E4B-it-w4a16 on the RTX 4060 Ti). M4.4 adds the requested advanced draft mode without
misrepresenting generated material as verified. M4.5 adds the third backend the M3.2 resolver
already routes to. None blocks the others except where the sequence above notes a shared run
path; each is independently shippable and unit-testable.

- **M4.1 Embedding-aware VRAM estimate. DONE (2026-06-22; details in `current.md`).** The planner
  prices the high-precision embedding mass separately (`weights_mib_detailed` + `hi_precision_params`,
  gated to partial quants), reads arch from a cached `config.json` (`enrich_arch` / `arch_from_config`),
  and feeds the corrected floor through `plan_model` to the M3.2 resolver + the M3.4 Optuna prune;
  E4B estimates 9.81 GiB vs the 9.8 GiB measured. Residual / possible improvements: derive the
  Gemma 3n Per-Layer-Embedding mass from `config.json` instead of the measurement-anchored
  `hi_precision_params_b`; model sliding-window KV (currently full-attention, conservative at long
  context); let `config.json` override curated arch values rather than only filling gaps.
- **M4.2 Pre-launch VRAM-contention guard. DONE (2026-06-22; details in `current.md`).**
  `llb.executor.contention` auto-derates `gpu-memory-utilization` to the actually-free fraction
  (non-destructive default), with `--evict` (unload Ollama) / `--wait` opt-in, aborting if even
  the derate cannot hold the M4.1 weight floor + KV; wired into `run-eval` for vLLM and recorded
  in `RunManifest.contention`. Residual / possible improvements: live validation on the CUDA host
  (a real contended vLLM launch); the guard reads GPU 0 only (single-GPU assumption); the abort
  KV headroom is a fixed floor rather than the arch-derived KV for the served context.
- **M4.3 vLLM serving knobs as CLI flags + a kernel preflight.** Tasks: (1) surface
  `--max-model-len` and `--gpu-memory-utilization` on `run-eval` (today only via `--config`;
  overrides revalidated by `RunConfig`); (2) add a `build-vllm` self-check that JIT-builds the
  flashinfer sampling kernel once and pins a host-compatible flashinfer, OR confirms the native
  sampler; (3) have `launch_env` re-enable the flashinfer sampler ONLY when the preflight
  confirms it compiles on sm_89, else keep the safe default off (flashinfer 0.6.x's
  `sampling.cuh` fails to build against newer CCCL/CUB on consumer sm_89). Acceptance: knobs
  settable without YAML; the preflight returns a definitive sampler verdict.
- **M4.4 Ontology-assisted corpus gold-set drafting.** Implement the reserved
  `GOLDSET_MODE=draft` as a multi-stage pipeline over a supplied text directory. **Decided
  extraction (default LLM-only, 2026-06-22):** all stages use the endpoint adapter (local
  default, frontier opt-in -- the Milestone H egress decision); a Python-native NER/coreference
  adapter (Stanza or spaCy `uk_core_news`) is pluggable opt-in, kept OUT of the base deps. Stages
  (each a grained task): (1) inventory + normalize supported docs preserving offsets; (2) extract
  named entities, aliases/coreference, events, claims, and evidence-backed subject-relation-
  object facts; (3) induce a constrained ontology candidate with confidence + source spans;
  (4) sample coverage across entity types, relations, sections, and difficulty; (5) draft
  Ukrainian question/reference/span triples; (6) exact-ground, deduplicate, and reject
  unsupported or circular items; (7) emit `verified: false` canonical drafts plus ontology /
  extraction / endpoint / prompt / model / cost / document-hash provenance under
  `$DATA_DIR/prepare-goldset/<timestamp>/`. This is a data-preparation ontology, NOT a GraphRAG
  runtime or a new retrieval backend (that is Milestone 6). Acceptance: injected unit tests cover
  every stage; a local fake endpoint proves the full flow; no draft scores until the frontier
  cross-check passes AND a human verifies a stratified sample (MH.5); generated ontology/facts
  link back to exact evidence.
- **M4.5 llama.cpp launcher. DONE (2026-06-22; details in `current.md`).** `LlamaCppLauncher`
  serves a GGUF via a `llama-server` subprocess behind the OpenAI-compatible `chat_once` seam
  (`-hf`/`-m` source, `-ngl` GPU/CPU offload split, `-c` ctx; `/health` readiness + `/props`
  served-context, startup log preserved on failure), records `n_gpu_layers` + served-vs-requested
  ctx in the telemetry/manifest, joins `GATE_BACKENDS` for the M3.3 reclaim gate, and is wired
  into `run-eval` via `_make_launcher` (`RunConfig.llamacpp_host` [env `LLAMACPP_HOST`] +
  `n_gpu_layers`). Injected process/HTTP/sleep make command building, readiness, chat, telemetry,
  resolver routing, and the reclaim gate unit-testable without llama.cpp/CUDA. Residual / possible
  improvements: live validation on a CUDA host serving a real GGUF; auto-derive `n_gpu_layers`
  from the planner's `gpu_layers` split (today config-set, default -1 = all on GPU); the `/props`
  served-context parse depends on the llama.cpp build's response shape.
- **M4 acceptance:** the planner's predicted weights land within tolerance of the measured load
  on the gemma-4 w4a16 candidates; a run launches cleanly when another process holds VRAM; the
  vLLM knobs are settable without YAML; M4.4 produces traceable unverified drafts from a nested
  corpus using both local and external endpoint adapters; and a GGUF-only candidate resolves to
  and serves through the llama.cpp launcher under the isolation gate.

## Milestone 5 -- security, agentic, and tooling benchmark

v1 deliberately scoped to RAG + text analysis (spec). Milestone 5 un-defers the next benchmark
categories from the spec taxonomy (Appendix D) and the deferred Premise 6 list. The design
principle is REUSE, not a new platform: every category is a new TASK FAMILY layered on the
existing substrate -- LangGraph templates, the shared `isolate_cell` gate, the `rank_board`
average-rank/Pareto/CI machinery with its Tier guard, the canonical manifest + per-case scores,
and the Streamlit/MLflow boards. Cross-cutting rules that hold for ALL M5 categories:

- **New Tier per category, never cross-ranked.** Extend the `aggregate` Tier guard with
  `TIER_SECURITY` / `TIER_TOOLING` / `TIER_AGENTIC` (alongside `TIER_SCREEN` / `TIER_PRIVATE`):
  a security ASR is not comparable to a RAG correctness score, so the board renders each
  category separately, exactly as the Tier-1/Tier-2 split does today.
- **Objective first, gated judge second.** Each category plants STRUCTURED ground-truth labels
  (via the M3.5 `prepare-synthetic-corpus` planter, planter != judge) so the headline metric is
  objective; the gated judge (M3.8) enters only for residual free-form quality, and only when
  trusted -- the same gate that governs the RAG board. Recovery is scored on planted-label IDs +
  embedder-cosine as a secondary signal (the MH.2 matching decision), uniformly across categories.
- **Verified-data gate (decided 2026-06-22).** Every gold/eval item is AI-drafted (M4.4 / the
  M3.5 planter), then a SECOND frontier model cross-checks it (grounding, non-circularity); a
  human spot-verifies a stratified SAMPLE before any `verified=true` item scores models (MH.5).
  The frontier cross-check is pipeline code; only the sample-verify is human (Spec Premise 3).
- **Same isolation contract.** Category runs go through `isolate_cell` (process per cell, PID-
  attributed VRAM reclaim gate, capped thermal cooldown) so longer agentic/tool loops cannot
  bias the next cell.
- **Backend capability is recorded, not assumed.** Like the screen's logprob/generation track
  split, tool-calling and logprob support vary by backend; record per-candidate capability and
  never cross-rank capable vs not.
- **Composite stays off until calibrated.** The spec's full default weights
  (`quality*0.60 + reliability*0.15 + security*0.10 + agentic*0.05 + tooling*0.05 +
  efficiency*0.05`) are recorded but NOT activated as a headline until every component carries a
  confidence interval; until then each category reports its own Pareto + CIs.

Dependencies: M5 builds on M4 (the llama.cpp launcher broadens the pool; the M4.1/M4.2 run-path
hardening keeps multi-category sweeps honest). M5.1 + M5.2 are fully objective and have NO human
dependency. M5.3 (agentic) and the chat-period part of M5.4 depend on M5.0 -- the text-analysis
scoring schema (AI-drafted; human sign-off via MH.2) and the multi-hop template (M1.4-rest).

- **M5.0 Prerequisites (AI-implementable; no human gating).** (1) **Text-analysis scoring
  schema:** I draft the full schema -- the unit of credit per sub-task, the planted-label
  taxonomy `prepare-synthetic-corpus` must emit, the objective-vs-judged split, and the
  label-ID + embedder-cosine thresholds / partial-credit rules (the MH.2 matching basis) -- as a
  concrete repo proposal; the human only signs off (MH.2). (2) **Eval templates (M1.4-rest):**
  the map-reduce (long-doc) + multi-hop LangGraph templates, following the built single-call
  template's node-closure shape; the multi-hop one is the M5.3 agentic substrate. Both unblock
  M5.3 and the M5.4 chat-period category. Acceptance: the schema proposal is committed for
  sign-off; both templates are unit-tested like the single-call one.
- **M5.1 Security / robustness benchmark.** A suite of adversarial cases scored by objective
  attack-success-rate (ASR). Subcategories (spec Appendix D security suite): prompt-injection,
  jailbreak, instruction-hierarchy violation, unsafe-content generation, tool-abuse,
  RAG-injection (malicious instructions hidden in retrieved chunks), and data-exfiltration
  resistance (corpus-secret / canary leakage). Build:
  - **Decided sourcing (hybrid, 2026-06-22):** reuse public jailbreak/injection/unsafe datasets
    (JailbreakBench / HarmBench / AdvBench) adapted to Ukrainian for the generic families, and
    the M3.5 planter for the corpus-specific RAG-injection + canary-exfiltration families (no
    public equivalent); every malicious instruction + canary is a STRUCTURED label. Not the
    garak harness and not a fully custom suite.
  - an objective detector per family: planted-instruction-followed / canary-leaked match -> a
    per-case binary outcome -> ASR (lower is better) plus refusal-appropriateness (do not
    over-refuse benign Ukrainian prompts). **Decided:** the unsafe-content family uses the same
    objective match PLUS the GATED judge for borderline quality -- no new safety classifier
    (ShieldGemma / a frontier moderation API stay opt-in only);
  - reuse `isolate_cell`, the manifest, and `rank_board` under `TIER_SECURITY`; the gated judge
    is a secondary signal only for borderline unsafe-content quality.
  - Acceptance: each attack family's detector is unit-tested with planted fixtures; a fake
    endpoint proves the full flow; ASR + refusal-appropriateness carry CIs; the security board
    is never cross-ranked with the RAG board.
- **M5.2 Tooling / MCP / function-calling benchmark.** Objective function-call correctness on a
  fixed tool catalog. Build:
  - **Decided dataset (adapt BFCL, 2026-06-22):** reuse the Berkeley Function-Calling
    Leaderboard cases adapted to Ukrainian (OpenAI tool/function-calling JSON schema), and serve
    the SAME catalog via the official `mcp` Python SDK server so both native FC and MCP transports
    run from one source (covers the deferred "MCP / tooling" category);
  - cases mapping a Ukrainian instruction -> expected tool name + argument JSON, scored
    objectively and **call-only** (validate the emitted call; tools are NOT executed here --
    execution lives in M5.3): tool-selection accuracy, argument-exactness (schema-valid + value
    match), no-hallucinated-tool rate, and well-formed-call rate;
  - a tool-call parse/validate layer over the existing OpenAI-compatible client (it already
    speaks tools); record per-backend tool-call capability and never cross-rank tool-capable vs
    text-only candidates.
  - Acceptance: schema validation + scoring are pure and unit-tested; a fake endpoint with
    canned tool calls proves the flow; per-backend capability is recorded under `TIER_TOOLING`.
- **M5.3 Agentic workflows benchmark.** Multi-step task completion in a sandboxed tool
  environment, scored by objective task success. Build:
  - the agentic loop as the multi-hop LangGraph template (M5.0) extended with tool calls +
    a controller node -- so this DEPENDS on M5.0 (the signed-off scoring schema + templates);
  - **Decided environment (custom deterministic tool-world, 2026-06-22):** a small in-memory,
    deterministic tool environment (mock files/DB + search over the UA corpus + a calculator),
    with tools EXECUTED in-sandbox -- not an external agent benchmark (tau-bench / AgentBench),
    keeping it lightweight + UA-native;
  - a small set of agentic tasks over that environment; objective completion-rate from the
    env-state / planted-label assertions, with the gated judge scoring only trajectory quality
    where a deterministic check cannot;
  - LangGraph is the single fixed agent harness (it is already the eval substrate). The other
    five frameworks (LangChain, LlamaIndex, Haystack, CrewAI, AutoGen -- spec Appendix D) stay
    deferred as a COMPARISON axis; M5.3 ranks the MODEL under one harness, not frameworks
    against each other (that is research-platform scope -> M5.5 / out of scope);
  - run through `isolate_cell` (agentic loops are longer; keep the thermal/VRAM gate); record
    trajectory length + tool-call count as efficiency. New `TIER_AGENTIC`.
  - Acceptance: the task environment + success checks are unit-tested with a fake tool-calling
    endpoint; objective completion-rate carries CIs; efficiency metrics recorded.
- **M5.4 Remaining benchmark taxonomy (summarization, structured output, chat-period analysis,
  reliability).** Fold in the rest of the spec Appendix D categories, each as a task family with
  its own schema:
  - summarization -- reference coverage via pinned-embedder cosine (the MH.2 basis, not ROUGE) +
    gated-judge faithfulness;
  - structured output -- objective JSON-schema conformance + field accuracy, validated with
    Pydantic (the project's existing validation layer; no new `jsonschema` dep);
  - chat-period analysis -- the text-analysis sibling over chat logs; depends on M5.0 + the
    `prepare-synthetic-corpus` planted labels; real-corpus and synthetic results reported
    SEPARATELY (never merged), per the spec;
  - reliability -- aggregate the existing typed failure taxonomy
    (empty/malformed/refusal/timeout/context-truncation/retrieval-miss/backend-crash/OOM/
    judge-failure) into a first-class reliability score.
  - Acceptance: each category scores on a fixed seeded set with CIs; the full composite weights
    are activated only once all components carry CIs.
- **M5.5 Platform & matrix expansion (deferred WITHIN M5; no committed consumer).** The
  Approach-B infrastructure expansions, listed so they have a home; each needs a consumer + a
  sign-off before building, and should be built last:
  - multi-backend comparison -- the SAME model across vLLM / Ollama / llama.cpp as a comparison
    axis (explicitly deferred in Premise 1; the per-source quant metadata from M3.2 is the seam);
  - multi-vector-store -- Chroma / Qdrant / LanceDB behind the existing RAG-store seam (FAISS is
    v1);
  - full GPU-class matrix -- 12 / 24 / 48 GB planning beyond the validated 16 GB class (the
    planner is already KV-cache-aware; this generalizes the host detection);
  - quality-per-watt -- a derived efficiency metric over the NVML power already sampled per cell
    (M3.3), trivial once a consumer wants it.
- **M5 acceptance:** security and tooling categories produce objective, CI-bearing boards from
  fake endpoints with no human dependency; agentic + chat-period categories build cleanly once
  M5.0 lands (and its schema is signed off); every category renders under its own Tier and is
  never cross-ranked with the RAG board.

## Milestone 6 -- GraphRAG (knowledge-graph RAG)

GO decided (2026-06-22). An ADDED retrieval backend behind the existing RAG-store seam, not a
replacement -- FAISS stays the default. The component architecture is locked; the only human
residual is the ontology-schema sign-off + the milestone scope acceptance (Milestone H).

**Decided (architecture, 2026-06-22):**
- **Graph store: Kuzu** -- an embedded, Apache-2.0 property graph (Cypher, pip-install, no server,
  native vector index), over server DBs (ArcadeDB / Dgraph) and the commercial-restricted Neo4j,
  to keep the "no servers, single desktop, low-maintenance" ethos.
- **Construction: reuse M4.4 extraction.** Feed M4.4's already-extracted entities / relations /
  SRO-facts (with source spans) into Kuzu -- NO second extraction framework (LlamaIndex
  `PropertyGraphIndex` / langchain `LLMGraphTransformer` are deliberately not pulled in).
- **Extraction LLM: local default, frontier opt-in** -- via the M4.4 endpoint adapter (no corpus
  egress by default), matching the OQ2 stance.

Tasks: (1) a Kuzu-backed graph store behind the RAG-store seam, swappable with FAISS via
`--retrieval-backend graph`; ingest M4.4 extraction into nodes/edges keeping `doc_id` + char
offsets; (2) apply the AI-drafted, human-signed-off constrained node/relationship ontology
schema (MH.2); (3) a graph-retrieval layer -- entity-link the question, expand k-hops, serialize
the subgraph as context while PRESERVING source spans so the M1.3 span metric still applies;
(4) record the retrieval backend in the manifest so graph-vs-FAISS runs are comparable; (5) reuse
the eval graph, scoring, isolation, and board unchanged. Dependencies: M4.4 (extraction) + the
signed-off ontology schema; scheduled after M5. Acceptance: a corpus builds a Kuzu graph from
M4.4 extraction; graph retrieval returns offset-bearing context that scores on the existing span
metric; runs are reproducible + manifest-recorded; the FAISS path is unchanged.

## Milestone H -- irreducibly-human tasks (no AI substitute)

Per the 2026-06-22 decisions, every task an AI service could perform -- schema drafting, data
drafting, frontier cross-checking -- now lives in M4 / M5 / M6 as implementation work. Milestone
H keeps ONLY what GPT / Gemini / Claude cannot legitimately do: provide human ground-truth,
human sample-verification, and human sign-off / scope approval. Human-paced, parallel to M4-M6.

- **M3.8 Judge calibration -- human ratings (DECIDED human-only, 2026-06-22).** A frontier proxy
  was rejected: the whole point is to measure the LOCAL judge against HUMAN judgment, so an
  LLM-vs-LLM calibration would not establish the "defensible vs human" claim (Spec Premise 2).
  The endpoint setup, worksheet pre-fill (`model_answer` + ungated `judge_rating` via
  `make calibration-run`), and scoring are already implemented (`current.md`); until rho clears
  the `>= 0.6` gate the judge stays demoted across the RAG board AND every M5 category. The
  irreducible human residual:
  1. Independently fill the `human_rating` column over the 86 verified calibration items WITHOUT
     looking at `judge_rating` first; span the full score range, including fluent-but-wrong
     adversarial answers (exercise the failure mode the judge is most likely to miss).
  2. Run `make calibration-score RATINGS=<filled.csv>` -> rho + bootstrap CI + the (mechanical)
     trust decision; rho `>= 0.6` admits the gated judge, else it stays demoted. The decision
     travels in the manifest.
  (The optional non-Gemma cross-check judge is an AI task, automatable -- not human work.)
- **MH.2 Sign-offs + corpus facts (human approval).** All drafting is AI; only the approvals and
  the facts only you know remain:
  1. Approve the AI-drafted text-analysis scoring schema (M5.0) before any dependent benchmark
     scores models. (Engine already decided: planted-label-ID matching + embedder cosine, not
     lemmatization or LLM-entailment.)
  2. Approve the AI-drafted GraphRAG ontology schema and the Milestone 6 scope / acceptance.
  3. Confirm the OQ4 corpus facts only you have: whether text-analysis reference answers already
     EXIST or must be authored, and which corpus is real vs synthetic (reported separately).
- **MH.5 Gold/eval data verification -- human sample-verify (DECIDED frontier cross-check + human
  sample, 2026-06-22).** Every gold/eval item (the RAG gold set, every M5 category, the M6
  ontology) is AI-drafted (M4.4 / the M3.5 planter) and frontier-cross-checked IN the pipeline;
  the irreducible human gate (Spec Premise 3) is to spot-verify a stratified SAMPLE and accept it
  before `verified=true` items score models. AI cannot own this gate without dropping the
  human-ground-truth guarantee for private model-selection data.

(Resolved 2026-06-22 and removed from H: MH.4 GraphRAG go/no-go -> GO -> Milestone 6; the
text-analysis schema DRAFT + the eval templates (M1.4-rest) -> M5.0, both AI-implementable.)

## Reuse (do not rebuild)

DeepEval G-Eval (maintained judge metrics), FAISS, sentence-transformers, `openai` client
(local backends, incl. tool/function calling for M5.2), litellm (frontier prep utils), Optuna,
MLflow (local), LangGraph (eval templates incl. the M5.3 agentic loop), DuckDB, Streamlit,
pynvml + psutil, lm-evaluation-harness-uk (INSAIT, Tier-1 public screen), Kuzu (embedded
Apache-2.0 property graph -- the Milestone 6 GraphRAG store, no server). Reuse public UA
datasets: SQuAD-uk + Belebele-uk (screen/baseline). Candidate seeds incl. MamayLM v2 12B/27B,
Lapa, Gemma 3. For M5: the official `mcp` Python SDK (M5.2 MCP transport), BFCL function-calling
cases (M5.2), and public adversarial sets JailbreakBench / HarmBench / AdvBench (M5.1), all
UA-adapted. All lightweight; no servers (no Celery/K8s) and no heavy service dependence
(no cloud, no Neo4j or similar with commercial restricted licence).

## Verification (forward)

- **M4:** the embedding-aware estimate predicts measured weights within tolerance; the
  pre-launch guard handles resident VRAM users; the vLLM knobs are settable without YAML; the
  ontology-assisted draft pipeline emits traceable, exact-grounded, unverified candidates from
  nested corpora; and a GGUF-only candidate resolves to and serves through the llama.cpp
  launcher under the isolation gate.
- **M5:** the security + tooling categories produce objective, CI-bearing boards from fake
  endpoints (no human dependency); agentic + chat-period categories build cleanly once M5.0
  lands (schema signed off); every category renders under its own Tier and is never cross-ranked
  with the RAG board.
- **Milestone 6:** a corpus builds a Kuzu graph from M4.4 extraction and graph retrieval scores
  on the existing source-span metric, with the FAISS path unchanged.
- **Milestone H:** judge calibration produces rho/CI over the HUMAN ratings; the AI-drafted TA
  schema (M5.0) and the M6 ontology are signed off; and a human sample-verify accepts the
  AI-drafted, frontier-cross-checked gold/eval data before it scores models.
- **AGENTS.md guardrails:** paths under `.data/llb/`; ASCII logs; confirm/create the MAX_JOBS
  helper before any vLLM/llama.cpp source build (the canonical `max_jobs()` helper is in
  `scripts/shared/common.sh`).

## Worktree parallelization

The forward work proceeds in mostly independent lanes:
- **robustness/run-path:** M4.1 estimator + M4.2 VRAM guard + M4.5 llama.cpp launcher all touch
  the run/launch path -- keep them sequential with each other.
- **CLI:** M4.3 is CLI-only and parallelizes with everything.
- **data:** M4.4 ontology drafting is an independent prep subpackage.
- **benchmark categories:** M5.1 (security) and M5.2 (tooling) are objective and parallelize once
  M4 lands; M5.3 (agentic) + M5.4 chat-period wait on M5.0 (AI-drafted schema + templates).
- **graph:** Milestone 6 (GraphRAG) is its own lane after M5, reusing M4.4 extraction.
- **human-gated:** Milestone H (M3.8 human ratings, MH.2 sign-offs, MH.5 sample-verify) runs on
  its own decision-paced lane.

## NOT in scope (resolved / out of v-next scope)

- Resolved in M2: candidate-model list (OQ3) + vLLM repo ids verified, and the MAX_JOBS helper
  path (OQ6, canonical `max_jobs()` in `scripts/shared/common.sh`).
- Resolved in M3.8: judge locality (OQ2) -- a LOCAL Gemma-4 judge, tiered by GPU class
  (12/16/32 GB), chosen for no corpus egress + reproducibility, with the Gemma-family
  self-preference bias disclosed (`current.md`); the residual is human ratings (M3.8), not the
  scorer implementation or model choice.
- Rejected Codex pushbacks (ruled the other way, do not revisit): defer-Optuna-to-finalists,
  LangGraph-only-where-needed, drop-MLflow, drop-thermal-gate, defer-vLLM.
- Moved INTO the forward plan (no longer deferred): the security / agentic / MCP-tooling
  benchmark categories and the remaining taxonomy are now Milestone 5; GraphRAG is Milestone 6
  (GO decided 2026-06-22); the multi-backend, multi-vector-store, full GPU-matrix, and
  quality-per-watt expansions are M5.5 (built only with a committed consumer).
- Still genuinely out of scope: the 6 agent frameworks as a comparison axis (M5.3 ranks the
  model under one fixed LangGraph harness, not frameworks against each other); loc-lm-bench as a
  public leaderboard (it consumes lang-uk / INSAIT results as a prior, never duplicates them).
