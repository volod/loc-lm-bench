# loc-lm-bench -- Implementation Plan (forward work)

Forward-only. Everything DELIVERED -- Milestones 0-4 (live-validated on the CUDA host) and the
Milestone 5 BUILD (the eval-template + text-analysis-schema prerequisites and every scored
category: security / tooling / agentic / text-analysis / summarization / structured-output /
chat-period / reliability, plus the second-frontier verified-data gate) -- lives in
[`current.md`](current.md) and is NOT repeated here. Spec (source of truth):
[`docs/design/spec.md`](../design/spec.md).

**Quick start:** `make demo-eval` runs the pipeline end to end (needs a running Ollama); the real
vLLM path is `llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry` on a CUDA host.

Remaining work: the Milestone 5 residuals (per-category breadth + host-dependent hardening), the
Milestone 6 build, and a human-only lane (Milestone H).

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

- **M3.8 judge calibration** ("Judge calibration"): fill `human_rating` over the 86 calibration
  items, then score rho. Blocks EVERY judged headline (RAG board + M5 unsafe-content quality,
  summarization faithfulness, agentic trajectory, free-form text/chat analysis). Objective metrics
  rank alone meanwhile. **Critical path -- start EARLY.**
- **MH.2 sign-offs + corpus facts** ("Schema and ontology sign-off"): approve the M6 ontology
  schema + M6 scope; confirm the OQ4 corpus facts (do text-analysis reference answers exist? real
  vs synthetic). Blocks Milestone 6 (ontology).
- **MH.5 data verification** ("Eval-data verification"): sample-verify a stratified sample of
  AI-drafted, frontier-cross-checked items, then flip via the ledger. Blocks any `verified=true`
  item SCORING REAL MODELS in any category (M5.1-M5.4 real runs, M6).

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
3. Rate INDEPENDENTLY via `make calibration-rate` (the interactive rater below; `judge_rating`
   hidden by default) -- author your own `human_answer` and set `human_rating`, spanning the full
   range and deliberately including fluent-but-wrong answers.
4. `make calibration-score RATINGS=<filled.csv>` -> rho + bootstrap CI + the mechanical decision.
   `rho >= 0.6` admits the gated judge; else it stays demoted. The decision travels in the manifest.

#### M3.8 calibration tooling -- implementation plan 

Two helpers to make step 2 (artifact) reproducible for any goldset and step 3 (human ratings)
ergonomic instead of hand-editing a CSV. Naming follows the existing `calibration-*` namespace
(DECIDED): the artifact step stays `calibration-run` (no `judge-corpus` alias) and the new rater is
`calibration-rate`.

**Part 1 -- the calibration artifact (ALREADY BUILT; document only).**
The artifact a human rates is the pre-filled worksheet CSV (`CAL_WS`, columns: `item_id, split,
question, reference_answer, model_answer, human_rating(blank), judge_rating`). It is produced TODAY
by `make calibration-run` (= `run-eval --split calibration --worksheet $(CAL_WS) --judge-model ...`),
which fills `model_answer` (the candidate's answer) + the ungated `judge_rating`. No new generation
logic or alias is needed (DECIDED: keep `calibration-run`); the only work here is documentation.
Generating it for the three cases:
- **committed fixture (default):** `make calibration-run MODEL=<cand> JUDGE_MODEL=<judge>
  JUDGE_BASE_URL=<url>` -- runs over the 86 verified calibration items of
  `samples/goldsets/ua_squad_postedited_v1`.
- **a new goldset:** add `GOLDSET=<path/goldset.jsonl>`; it must have a `calibration` split with
  `verified=true` items (`run-eval` scores only verified items).
- **a draft from a text corpus (M4.4 `prepare-goldset-draft`):** the draft is `verified=false`, so
  first cross-check it (`make cross-check-goldset ...`) and human sample-verify a calibration subset
  (MH.5) into a verified ledger, then point `GOLDSET=` at that ledger. (A draft's reference answers
  are unverified, so rating against them would add noise -- calibrate only on verified items.)
  RESIDUAL to decide at review: whether to also support a "quick judge sanity" worksheet over an
  UNVERIFIED draft calibration split (clearly marked not-for-headline) -- would need the worksheet
  pre-fill path to bypass the `verified` filter.

**Part 2 -- `calibration-rate` interactive human rater (NEW; the substantive build).**
A terminal session that walks the worksheet item-by-item and writes the human columns in place. New
focused module `llb/judge/rate.py` (interactive I/O kept OUT of the pure-stats `calibration.py`);
exposed as a third subcommand `python -m llb.judge.calibration rate --worksheet <csv> [opts]` and a
Makefile target `calibration-rate` (wrapping it with `CAL_WS`). Design:
- **Data model -- goldset vs worksheet (analysis; DECIDED: no new goldset field).** Two distinctions
  the rater needs are NOT goldset properties: (1) WHO authored an item is already
  `GoldItem.provenance` (`human-authored` / `human-verified` vs `frontier-drafted` /
  `ontology-drafted` / `public-reused` / `sample-generated`) -- no new field; (2) whether an item is
  RATED-by-human is a property of one *(judge, candidate-answer, item)* calibration run, not of the
  immutable item, so it lives in the WORKSHEET, never the goldset (putting it on `GoldItem` would
  couple the reproducible goldset to transient per-judge progress and break across judges/candidates).
  The human's authored answer is likewise calibration data in the worksheet; promoting a superior
  human answer into the goldset is a separate MH.5-ledger path (-> provenance `human-authored` /
  `human-verified`), out of the rater's scope.
- **Worksheet columns:** the human authors their OWN answer in addition to rating (DECIDED), so
  `WORKSHEET_COLS` gains `human_answer` (free text) + `human_rating`, an optional `human_note`, and a
  passthrough `provenance` (copied from the `GoldItem`, so the card shows the item's source and
  `calibration-score` can stratify rho by provenance -- e.g. human-authored vs AI-drafted). An
  optional `human_status` (pending / rated / skipped) makes a DELIBERATE skip distinguishable from
  not-yet-reached (default resume keys on empty `human_rating`, so this is a refinement, not
  required). The rater ADDS any missing column on load, so existing worksheets upgrade
  transparently; `emit_worksheet` / `write_filled_worksheet` populate `provenance` + leave the human
  columns blank. `calibration-score` is unchanged (reads `human_rating` + `judge_rating`).
- **Durability across regeneration (DECIDED):** re-running `make calibration-run` MUST MERGE existing
  human columns (`human_answer` / `human_rating` / `human_note` / `human_status`) into the freshly
  pre-filled worksheet by `item_id`, never clobber them; an item whose regenerated `model_answer`
  CHANGED (different candidate) has its stale human rating cleared with a warning (the rating no
  longer applies to the shown answer). So a human's work survives a re-run with the same deterministic
  candidate, and is correctly invalidated only when the rated answer actually changes.
- **Item "card" (per item):** progress line `item k/N (rated R, remaining N-R)`, then `item_id`,
  `question`, `reference_answer`, `model_answer`, and the current `human_answer` / `human_rating` if
  already set. The `judge_rating` is HIDDEN by default (anchoring control -- the manual requires
  rating INDEPENDENTLY); a `--show-judge` flag reveals it for post-hoc review only.
- **Rating scale:** integer Likert `RATING_MIN..RATING_MAX` (CONFIRMED `1..5`, named constants;
  Spearman is rank-based so the 1-5 human vs [0,1] judge scales are compatible). Anchors printed in
  the help: 1 = wrong / unfaithful; 2 = mostly wrong; 3 = partially correct; 4 = mostly correct;
  5 = fully correct + faithful.
- **Prompt commands:** a number in range = set rating + advance; `a` = author/edit the current
  item's `human_answer` (single-line text input; empty input clears it); `n`/Enter = next (no
  change); `p`/`b` = previous (navigate back to change an answer); `j <N>` = jump to item N; `u` =
  jump to next UNRATED; `c` = clear the current item's `human_rating`; `?`/`h` = help; `q` = save +
  quit. (An optional `note` command edits `human_note`.)
- **Persistence (the "write immediately?" question -- investigated):** calibration sets are SMALL
  by design (committed = 86; a calibration split is always a small fraction), so v1 WRITES THROUGH
  after every edit -- rewrite the whole CSV atomically (temp file + `os.replace`, reusing the
  `_atomic_write_text` pattern), preserving all columns/order. The CSV IS the state, so resume and
  crash-safety are free; no separate journal. (Documented fallback for a hypothetical huge set:
  debounced / batch-on-quit writes + an append-only edit journal -- NOT v1.)
- **Resume / default mode:** with no `--start`, begin at the FIRST item whose `human_rating` is
  empty (so re-running continues where you left off).
- **Options:** `--worksheet` (default `CAL_WS`); `--start N` (1-based) to begin at a specific item;
  `--clear` to wipe ALL human columns (`human_rating` + `human_answer` + `human_note` +
  `human_status`) first (confirmation-gated) and start fresh; `--show-judge` (default off).
- **KeyboardInterrupt / EOF:** Ctrl-C (and EOF) are caught and treated as save + quit (never data
  loss; the last edit is already on disk via write-through), printing a resume hint + the
  `make calibration-score` next step.
- **Testability (acceptance):** the pure pieces -- `format_card`, `parse_command`,
  `first_unrated_index`, column-upgrade-on-load, the merge-on-regenerate (preserve human columns by
  `item_id`; clear only on a changed `model_answer`), and the atomic load/save round-trip -- are
  unit-tested; the session loop is driven by an INJECTED input iterator + output sink (scripted:
  author an answer, rate, go back, jump, clear, quit) asserting the resulting CSV + that an injected
  `KeyboardInterrupt` still saves. No model / endpoint / GPU needed (it operates only on the CSV).
- **Decisions (confirmed 2026-06-24):** (a) the human BOTH authors `human_answer` AND sets
  `human_rating` (the `a` command + the numeric rating); (b) the 1-5 scale + anchors above are
  confirmed; (c) command names are `calibration-run` (artifact, unchanged) + `calibration-rate`
  (rater) -- no `judge-corpus`/`judge-human` aliases.

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

Canonical order for picking up the remaining work. Sequence numbers are stable workstream
identifiers (AGENTS.md); a workstream appears only while it has open work.

1. **Milestone 5 -- residuals** (the category BUILD is delivered; see `current.md`). Each category
   stays its own Tier, never cross-ranked with the RAG board.
   1. **M5.1** Security -- sourcing breadth (public sets UA-adapted + the M3.5 planter for the
      corpus-specific RAG-injection/canary families) + the opt-in unsafe-content gated judge.
   2. **M5.2** Tooling -- the MCP-SDK transport + a selectable native OpenAI `tools=` caller; the
      full BFCL UA dataset adaptation + per-argument value tolerance.
   3. **M5.3** Agentic -- the LangGraph harness wrapper + the trajectory-quality gated judge;
      broaden the task set.
   4. **M5.4** Remaining-taxonomy residuals -- the gated-judge wiring (text-analysis judged
      sub-tasks + `long_doc`, summarization faithfulness), structured nested/array validation, the
      chat-period chat-log planter; the composite stays off until calibrated.
   5. **M5.5** Platform & matrix expansion -- optional; build last (needs a committed consumer).
   6. **M5.6** Host-dependent run-path hardening + the remaining data-prep items (spaCy adapter,
      long-doc chunking, richer ontology confidence); rides the first real-host sweep.
2. **Milestone 6** (after M5) -- GraphRAG (Kuzu). ⚠ needs MH.2 (M6 ontology + scope sign-off).
3. **Milestone H** (human-paced, parallel) -- M3.8, MH.2, MH.5. See the prerequisites block above.

Real-model scoring of any `verified=true` item still waits on MH.5 (the human gate); the objective
category boards already do not depend on the M3.8 judge calibration.

---

## Milestone 5 -- residual work

The category build is delivered in [`current.md`](current.md). These constraints still govern the
remaining M5 work (residuals below + the M5.5 expansion):
- **New Tier per category, never cross-ranked.** A new category stamps its own `ModelResult.tier`;
  the `aggregate` guard refuses a board mixing distinct tiers.
- **Objective first, gated judge second.** The gated judge (M3.8) enters only for residual
  free-form quality and only when trusted; objective recovery is the headline meanwhile.
- **Verified-data gate.** Every gold/eval item is AI-drafted + frontier-cross-checked in-pipeline,
  then ⚠ human sample-verified (MH.5) before `verified=true` scores models.
- **Same isolation contract.** All real runs go through `isolate_cell` (reuse
  `llb.bench.common.drive_with_backend`).
- **Record backend capability, don't assume it.** Tool-calling / logprob support varies by backend;
  record per-candidate and never cross-rank capable vs not.
- **Composite stays off until calibrated.** The spec default weights are recorded but NOT a headline
  until every component carries a CI; until then each category reports its own Pareto + CIs.

### M5.1 Security / robustness -- residuals
- **Sourcing breadth:** wire the public-set adapters (JailbreakBench / HarmBench / AdvBench,
  UA-adapted) for the generic families and the M3.5 planter for the corpus-specific RAG-injection +
  canary families (over a real corpus) -- today only a committed hand-authored UA seed exists.
- **Unsafe-content gated judge:** add the opt-in gated judge for borderline unsafe-content quality
  (objective `refuse` detection PLUS the judge); no new safety classifier (ShieldGemma / frontier
  moderation stay opt-in).
- **Verified-data gate:** the attack set still needs a human sample-verify (MH.5) before headline use.
- Deep dive: [security learning path](../guides/learning-path-security.md).

### M5.2 Tooling / function-calling -- residuals
- **MCP transport:** serve the SAME catalog via the official `mcp` Python SDK server, and wire a
  selectable NATIVE OpenAI `tools=` caller (the parser already handles native responses; the
  default driver uses a universal text protocol) -- so native FC and MCP run from one source.
- **Dataset breadth:** adapt the real Berkeley Function-Calling Leaderboard (BFCL) cases to
  Ukrainian (today a small hand-authored UA catalog); add per-argument tolerance for free-text
  values (exact-match only today); MH.5 human sample-verify before headline use.

### M5.3 Agentic workflows -- residuals
- **LangGraph harness wrapper:** a `build_agentic_graph` (mirroring `build_multi_hop_graph`) over
  the pure loop -- LangGraph stays the single fixed harness; the other five frameworks stay
  deferred as a comparison axis (ranks the MODEL under one harness -- out of M5 scope, by design).
- **Trajectory-quality gated judge:** wire the gated judge for trajectory quality a deterministic
  check cannot cover; broaden the task set (real-UA-corpus search tasks); MH.5 verify before headline.

### M5.4 Remaining taxonomy -- residuals
- **summarization** -- wire the opt-in gated-judge faithfulness signal.
- **structured output** -- nested-object / array-item validation + per-field value tolerance
  (schemas are flat exact-match today).
- **chat-period** -- a chat-log-shaped planter prompt + a real chat corpus (OQ4, human-gated).
- **text-analysis judged sub-tasks** -- wire the gated judge into `llb.bench.text_analysis` for
  `narrative` / `insight` (objective floor only today) and drive `long_doc` through the map-reduce
  template; use a `contradiction`'s paired-span `attrs`; load the per-tier text-analysis runs into
  the Streamlit board.
- **composite** -- the full composite weights stay OFF (each category reports its own board + CIs)
  until calibration; activate only once every component is calibrated. MH.5 verify before headline.

### M5.5 Platform and matrix expansion (deferred within M5)
- multi-backend comparison -- same model across vLLM / Ollama / llama.cpp (per-source quant metadata
  from M3.2 is the seam);
- multi-vector-store -- Chroma / Qdrant / LanceDB behind the RAG-store seam (FAISS is v1);
- full GPU-class matrix -- 12 / 24 / 48 GB beyond the validated 16 GB class;
- quality-per-watt -- a derived metric over the NVML power already sampled per cell (M3.3).

### M5.6 Carried-forward M4 residuals (small code hardening)
Run-path items land with whichever lane first sweeps the 16 GB host; the remaining data-prep items
land before the M6 extraction reuse.
- **Run-path (host-dependent):**
  1. M4.1 -- model Gemma 3/4 sliding-window KV (full-attention today); let a cached `config.json`
     OVERRIDE curated arch fields, not only fill gaps.
  2. M4.2 -- read all GPUs (guard reads GPU 0 only); derive the KV abort headroom from the served
     arch instead of the fixed floor.
  3. M4.3 -- auto-pin a host-compatible flashinfer when the bundled one fails; record the chosen
     sampler in the manifest; re-run the preflight on a driver change without a full rebuild.
  4. M4.5 -- handle further `/props` response shapes; exercise a real partial-offload split on an
     oversized GGUF (only the all-on-GPU path is confirmed).
- **Data-prep (feeds the M6 extraction reuse):**
  1. Ship the opt-in Stanza / spaCy `uk_core_news` `ExtractionAdapter` plug-in (seam exists).
  2. Chunk over-long docs for extraction instead of one truncated call (`EXTRACT_MAX_CHARS`).
  3. Induce ontology-type confidence from a richer signal than raw frequency; carry the induced
     types into the drafting prompt as explicit constraints.

**Remaining M5 verification:** the M5.6 run-path validations pass on the first real CUDA-host sweep,
and the MH.5 human sample-verify accepts a stratified sample before any `verified=true` item scores
real models.

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
screen), Kuzu (M6 graph store). Public UA datasets: SQuAD-uk + Belebele-uk. For the M5 residuals:
the official `mcp` Python SDK (M5.2), BFCL cases (M5.2), and JailbreakBench / HarmBench / AdvBench
(M5.1), all UA-adapted. No servers (no Celery/K8s/Neo4j), no cloud dependence.

## Verification (forward)

- **M5 (remaining):** the M5.6 run-path validations pass on the first real CUDA-host sweep; the
  MH.5 human sample-verify gates real-model scoring. (The category boards are already objective +
  CI-bearing from fake endpoints under their own Tiers -- see `current.md`.)
- **M6:** a corpus builds a Kuzu graph from M4.4 extraction and graph retrieval scores on the
  existing source-span metric, FAISS unchanged.
- **Milestone H (⚠ human):** M3.8 produces rho/CI over HUMAN ratings; the M6 ontology is signed off
  (MH.2); a human sample-verify (MH.5) accepts the AI-drafted, frontier-cross-checked data before it
  scores models. See [`human-in-the-loop-evaluation.md`](../guides/human-in-the-loop-evaluation.md).
- **AGENTS.md guardrails:** paths under `.data/llb/`; ASCII logs; confirm the canonical `max_jobs()`
  helper (`scripts/shared/common.sh`) before any vLLM/llama.cpp source build.

## Worktree parallelization

- **M5 residuals:** the per-category open items (sourcing breadth, native-FC/MCP transport,
  gated-judge wiring, judged-subtask + long_doc, structured nested validation) parallelize.
- **M5.6 residuals:** the host-dependent run-path items attach to whichever lane first sweeps the
  16 GB host; the remaining data-prep items (spaCy adapter, long-doc chunking, ontology confidence).
- **graph:** Milestone 6 is its own lane after M5, reusing M4.4 extraction.
- **human-gated:** Milestone H (M3.8, MH.2, MH.5) runs on its own decision-paced lane.
