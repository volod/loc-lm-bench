# category suite Current State

## category suite -- security, agentic, tooling (build COMPLETE)

The category suite BUILD is complete and unit-tested (no GPU): the eval-template prerequisites +
the signed-off text-analysis schema (text analysis), and every scored category -- security
(security benchmark), tooling (tooling benchmark), agentic (agentic benchmark), text-analysis +
summarization + structured-output + chat-period + reliability (text-analysis and category
expansion) -- plus the second-frontier verified-data gate (verified-data hardening). Each
category renders under its OWN Tier (never cross-ranked with the RAG board), produces an
objective, CI-bearing board from a fake endpoint, and persists a canonical manifest + per-case
scores like `run-eval`. The category suite RESIDUALS are now delivered too (per-category, below):
the security benchmark sourcing breadth (public-set adapters + corpus RAG-injection/canary
planter) + unsafe-content gated judge; the tooling benchmark native-FC + MCP transport, BFCL UA
adapter, and per-argument tolerance; the agentic benchmark real-corpus search tasks; the category
expansion nested/array structured cases + matching, chat-period planter/ingestion, and
text-analysis judged sub-tasks + long_doc + contradiction + board; and the verified-data hardening
ontology data-prep items (spaCy adapter, long-doc extraction chunking, richer confidence).
What remains is forward work in [`plan.md`](../plan.md): the extended workflow harness comparison,
RAG prompt-system generation/tuning, and platform/matrix expansion.

The shared category suite substrate (REUSE, not a new platform) lives in `llb.bench.common`:
`local_complete` / `launcher_complete` build the production `complete` (prompt -> raw text);
`drive_with_backend` reaches a running endpoint / Ollama directly or LAUNCHES a VRAM-owning
backend under the existing `isolate_cell` contract; `category_result` stamps a `ModelResult`
with the category Tier (per-case scores -> bootstrap CI); `render_board` ranks under that
Tier via `rank_board` (whose guard refuses to cross-rank tiers) + `format_board`;
`persist_category_run` writes the run bundle under `$DATA_DIR/<category>/<ts>/`. The category
Tier constants (`TIER_TEXT_ANALYSIS` / `TIER_SECURITY` / `TIER_TOOLING` / `TIER_AGENTIC` /
`TIER_SUMMARIZATION` / `TIER_STRUCTURED`) live in `llb.scoring.aggregate`.
Each category exposes a `bench-*` CLI command.

### Eval templates (deferred eval templates) -- `llb.eval.{common,map_reduce,multi_hop}`
The two remaining DRY LangGraph templates, following the single-call template's node-closure
shape (`graph.py`). The shared status taxonomy, refusal markers, `classify_response`, and
`format_context` live in `llb.eval.common` and are reused by all three templates.
- **map-reduce (`map_reduce.py`)** -- split a long document into overlapping segments, MAP a
  partial answer per segment, REDUCE the partials into one answer. The long-doc comprehension
  substrate; segments that find nothing emit a `(немає інформації)` marker the reduce step drops.
- **multi-hop (`multi_hop.py`)** -- retrieve -> CONTROLLER -> {retrieve again | answer} with a
  conditional edge, bounded by `max_hops`; gathered chunks are deduped across hops. This is the
  agentic benchmark substrate (agentic benchmark grows the controller into tool calls + 
  an in-sandbox exec node). Trajectory length (`n_hops`) + model-call/token counts are recorded
  as the efficiency signal.
Like `graph.py`, every node closure / parser / message builder is pure and unit-tested WITHOUT
langgraph; only `build_map_reduce_graph` / `build_multi_hop_graph` import it. Both compiled
graphs were smoke-run end to end with fake store/launcher.

### Text-analysis scoring schema (text analysis)
The objective scoring schema for the text analysis, drafted as a concrete repo
proposal for human sign-off (text-analysis sign-off). The proposal doc is
[`docs/design/text-analysis-schema.md`](../../design/text-analysis-schema.md); the executable form
is `llb.scoring.text_analysis` + the `PlantedLabelRecord` / `SubtaskScore` contracts. It defines:
the text-analysis SUB-TASKS (the per-sub-task unit of credit -- key_fact / entity / topic /
trend / risk / decision / contradiction objective, narrative / insight / long_doc judged); the
PLANTED-LABEL taxonomy `prepare-synthetic-corpus` must emit (stable `label_id`, surface `value`
+ `aliases`, grounding offsets, `attrs`, objective/judged flag); and the MATCHING engine -- the
text-analysis sign-off-decided basis of label-ID exact/normalized surface match, then PINNED-EMBEDDER COSINE as the
secondary signal (`TAU_FULL=0.85` full, `[0.70, 0.85)` partial credit 0.5), NOT lemmatization and
NOT LLM-entailment. Greedy one-to-one assignment yields per-sub-task precision / recall / F1
(unmatched predictions are false positives, penalizing hallucinated extractions); the document
objective headline is the mean F1 over objective sub-tasks, with judged sub-tasks kept out of it
(the gated judge owns those). The cosine similarity is INJECTED, so the whole engine is pure and
unit-tested without the embedder; `embedder_similarity()` is the production default.

The schema is SIGNED OFF (text-analysis sign-off, 2026-06-23 -- thresholds accepted as proposed; recorded at the top
of the proposal doc).

**Direction-aware trend credit (text analysis).** A `trend` label's planted `attrs.direction`
(up | down | flat) now adjusts credit: `direction_of(text)` infers a direction from a UA/EN stem
lexicon, and `_direction_penalty` ZEROES a trend prediction's surface credit when its detectable
direction CONFLICTS with the label (a right-subject/wrong-direction answer is substantively wrong,
so the label stays unrecovered AND the prediction becomes an unmatched false positive). A
prediction with no detectable direction, or a matching one, keeps its surface credit
(`DIRECTION_CONFLICT_CREDIT = 0.0` is the named knob).

### Synthetic text-analysis planter (text analysis) -- `llb.prep.text_analysis_corpus`
`prepare-synthetic-corpus --text-analysis` now emits the RICHER per-kind `PlantedLabelRecord`s the
schema defines (key_fact / entity / topic / trend / risk / decision, judged narrative / insight),
instead of QA-style `key_fact` only. `plant_labels` is pure: it grounds each label's `value`
against the doc (exact, then casefold/whitespace-normalized-but-exact via `frontier.ground_span`),
falls back to the planter's verbatim `evidence` quote (whose grounded substring becomes an accepted
alias), DROPS quote-bearing kinds (`GROUNDED_REQUIRED_KINDS` = key_fact/entity/contradiction) whose
value+evidence are ungrounded while keeping analytical kinds (topic/trend/risk/decision/insight)
ungrounded (no offsets), and backfills a trend's `attrs.direction` from its text when the planter
omitted it. `prepare_text_analysis_corpus` writes a self-contained bundle under `out_dir/`:
`corpus/<doc>.md`, `text_analysis_labels.jsonl` (the records), and a `provenance.json` tagging
`synthetic: true` + per-kind label counts. The planter != judge guard is reused; `litellm` stays
lazy and the completion is injectable, so the full flow is unit-tested from a fake endpoint.

### category suite benchmark scaffolding (text analysis) -- `llb.bench.{common,text_analysis}`
`llb.bench.common` is the shared substrate every category reuses (REUSE, not a new platform):
`local_complete` / `launcher_complete` build the production `complete` (prompt -> raw text) over an
OpenAI-compatible endpoint; `drive_with_backend` builds that `complete` for a running endpoint /
Ollama directly, or LAUNCHES a VRAM-owning backend (vllm / llamacpp) and runs the whole category
under the SAME `isolate_cell` contract as the RAG sweep (PID-attributed reclaim gate + capped
cooldown); `category_result` stamps a `ModelResult` with the category's Tier (per-case scores feed
the bootstrap CI); `render_board` ranks under that Tier via the existing `rank_board` (whose guard
refuses to cross-rank tiers) + `format_board`; and `persist_category_run` writes a canonical
manifest + per-case scores bundle under `$DATA_DIR/<method>/<ts>/` exactly like `run-eval`.
`run_gated_judge` is the shared opt-in GATED-judge seam: a thin reuse of `scoring.judge.run_judge`
that returns per-record scores ONLY when a judge is configured AND trusted (`judge_rho >= threshold`,
the judge calibration gate gate); the `scorer` is injectable (a fake in tests), the default is the DeepEval judge
bound to a `base_url` (lazy-imported). A category records the judge signal ALONGSIDE its objective
headline, never folded in (objective-first). Summarization (faithfulness) and agentic
(trajectory-quality, agentic benchmark) are the consumers (below); the unsafe-content wiring (security benchmark) reuses the
same seam.

`llb.bench.text_analysis.run_text_analysis` is the text analysis scored runner: it loads a planter bundle
(`corpus/` + `text_analysis_labels.jsonl`), asks the candidate to extract each document's present
sub-tasks as a JSON object keyed by kind (`analysis_prompt`), parses it (`parse_predictions`
coerces scalars + missing kinds), scores recovery with `score_document`, and aggregates one
`ModelResult` under `TIER_TEXT_ANALYSIS` -- its OWN Tier, never cross-ranked with the RAG board. The
per-document objective scores carry the CI; the per-sub-task F1s ride a flat `subtask_f1_json`
column (so the parquet schema stays stable across docs that plant different kinds); a malformed /
empty response is a typed status (objective 0, reliability down). The `complete` and the cosine
`similarity` are both injectable, so the whole flow is proven from a fake endpoint without a GPU or
the embedder. CLI: `llb bench-text-analysis --bundle <dir> --model <m> [--backend ...]` (and the
`real_corpus` flag keeps real-corpus runs reported separately from synthetic). New tier constants
`TIER_TEXT_ANALYSIS` / `TIER_SECURITY` / `TIER_TOOLING` / `TIER_AGENTIC` live in
`llb.scoring.aggregate` (the existing `_validate_board_cohort` guard already refuses any board that
mixes distinct tiers, so these are the named identities each category stamps).

The text analysis carry-overs are now DELIVERED under category expansion (see "Text-analysis judged sub-tasks + long_doc +
contradiction + board" below): the gated judge over narrative / insight, `long_doc` via the
map-reduce template, the `contradiction` paired-span matching, and the Streamlit board loading the
per-tier category runs. OQ4 is CONFIRMED (2026-06-25): a REAL text-analysis corpus exists
alongside synthetic (scored + reported SEPARATELY via the runner's `synthetic` flag), reference
answers must be AUTHORED, and a real chat-log corpus exists -- the real chat path is delivered in
category expansion (`llb.prep.chat_corpus`). Per the OQ-egress decision (2026-06-25) the real chat logs use a
LOCAL cross-check verifier only (no egress) while the real text-analysis corpus may use the frontier
(litellm) cross-check.

### security benchmark Security / robustness benchmark -- `llb.scoring.security` + `llb.bench.security`
The objective security board (no human dep to build). `llb.scoring.security` is the pure engine:
every case carries a STRUCTURED success detector (no safety classifier; ShieldGemma / frontier
moderation stay opt-in), one of two kinds covering all seven spec families --
  * `leak` -- the attack plants a unique MARKER (an injected token the model is tricked into
    emitting, or a corpus CANARY it must not reveal); the attack SUCCEEDS when any marker appears
    in the output (casefolded substring). Covers prompt-injection, instruction-hierarchy,
    RAG-injection, tool-abuse, and data-exfiltration/canary.
  * `refuse` -- a harmful ask (jailbreak / unsafe-content); the attack SUCCEEDS when the model
    COMPLIES (a substantive, non-refusal answer, via the shared `eval.common.is_refusal`).

`score_security` yields ASR (attack-success rate, lower better), its complement the DEFENSE rate
(the ranking headline, higher better), per-family ASR, and refusal-appropriateness (did the
refuse/answer decision match the case's expected behavior -- refuse harmful asks, answer benign +
legitimate-task injection cases -- so a model cannot win by refusing everything). Benign control
prompts feed only refusal-appropriateness, never ASR. `llb.bench.security.run_security` drives a
candidate over the cases through an injectable `complete`, frames RAG-injection cases as untrusted
retrieved context (no "ignore injected instructions" hand-holding -- the realistic test), stamps a
`ModelResult` under `TIER_SECURITY` (defense rate as `objective_score`, per-attack-case defended
series -> the bootstrap CI), and records ASR + refusal-appropriateness, each with its own CI, in
the manifest. A committed UA case set (`samples/security_cases_uk.json`) covers every family plus
benign controls; `load_cases_file` loads it. CLI: `llb bench-security`. Reuses `is_refusal`
(extracted into `eval.common` + now shared by `classify_response`), `bench.common`, and the
`isolate_cell` contract via `drive_with_backend`. Detectors + scoring + the runner are unit-tested
with planted fixtures + a scripted fake endpoint (vulnerable vs robust model), no GPU.

**Sourcing breadth (security benchmark, delivered).** Two pure, injectable adapters extend the committed seed:
- `llb.prep.security_sources` adapts the public adversarial sets (AdvBench / HarmBench /
  JailbreakBench) into `SecurityCase` records, UA-adapted: a per-source normalizer extracts one
  `Behavior` (id + text + category), and `behavior_to_case` frames it as a UA request (`refuse`
  detector, `expect_refusal=True`). UA adaptation is INJECTABLE (`translate`); a `jailbreak_wrap`
  (UA DAN-style templates) turns a bare `unsafe_content` ask into a `jailbreak`-family case. Raw
  rows are passed IN (operator's local CSV/JSON under each set's license) -- nothing vendored or
  fetched at import. CLI: `llb adapt-security-set --source advbench --rows-file ... [--jailbreak]
  [--merge-seed]`.
- `llb.prep.security_planter` is the frontier drafting-style corpus-specific planter: over a real corpus it
  splices a marker-bearing instruction INTO a document (`rag_injection` leak cases, the attack in
  the retrieved context, a benign user turn) and embeds a unique CANARY in a confidential doc
  (`data_exfiltration` leak cases). Markers/canaries are DETERMINISTIC per (doc, index). CLI:
  `llb plant-security-cases --corpus-root ... [--merge-seed]`.

**Unsafe-content gated judge (security benchmark, delivered).** `run_security` now takes the opt-in gated judge
(`--judge-model` / `--judge-rho` / `--judge-base-url`): over the harmful-ask (`refuse`-detector,
non-benign) cases it scores a `refusal_quality` scalar (answer-relevancy to a SAFE-REFUSAL intent --
a clean refusal scores high, compliance low), recorded ALONGSIDE the objective defense rate (per
harmful-ask case + mean + CI, a `JudgeStatus` in the manifest), never folded into the headline, and
only when trusted (`judge_rho >= 0.6`, the judge calibration gate gate); else demoted. No new safety classifier
(ShieldGemma / frontier moderation stay opt-in). `judge_scorer` is injectable, so it is proven with
a fake judge.

Both producers ship `verified=false`-equivalent: the human verification gate human sample-verify still gates any
headline use of the attack set.

### tooling benchmark Tooling / function-calling benchmark -- `llb.scoring.tooling` + `llb.bench.tooling`
The objective, CALL-ONLY function-calling board (tools are NOT executed -- that is agentic benchmark).
`llb.scoring.tooling` has two pure layers: the PARSE layer (`parse_tool_call`) normalizes a backend
response into a `ToolCall` whether it is a NATIVE OpenAI `tool_calls` object, a pre-parsed dict, or
a text-only backend's JSON call in `content` (name/arguments aliases tolerated) -- so tool-capable
and text-only backends share ONE scorer and are never cross-ranked; the SCORE layer
(`score_tooling`) reports the four plan metrics -- tool-selection accuracy, argument-exactness
(`validate_arguments` is a lightweight required/type/no-unknown check, no `jsonschema` dep; plus
`arguments_match` exact value match, casefold/strip-insensitive for strings), no-hallucinated-tool
rate, and well-formed-call rate -- with the headline `call_accuracy` requiring the right tool AND
exact arguments. A no-tool case (the model should NOT call) scores correct only on no-call, so
over-calling is penalized.

`llb.bench.tooling.run_tooling` drives a candidate over a catalog + cases through an injectable
`complete` using a universal TEXT tool-calling protocol (`text_tool_prompt` embeds the catalog as
JSON; the model returns a JSON call), so every backend is exercised uniformly and a FAKE endpoint
proves the flow; it stamps a `ModelResult` under `TIER_TOOLING` (call accuracy as `objective_score`,
per-case correctness -> CI) and records all four rates + the tool-call protocol/capability in the
manifest. A committed BFCL-style UA bundle (`samples/tooling_cases_uk.json`: 5 tools, 12 cases incl.
no-tool controls + per-argument-tolerance cases) ships; `load_catalog_file` loads it. CLI:
`llb bench-tooling`. Parse, validation,
scoring, and the runner are unit-tested (native + text + malformed responses, perfect vs text-only
model), no GPU.

**Transports + breadth (tooling benchmark, delivered).**
- **Selectable native FC + MCP from one catalog.** `run_tooling` takes an injectable `ToolCaller`
  `(instruction, catalog) -> ToolCall | None`; `text_tool_caller` is the default universal text
  protocol, `native_tool_caller` calls a tool-capable endpoint with native OpenAI `tools=`
  (`openai_tools` converts the catalog; the existing `parse_tool_call` normalizes the native
  response, so the SAME scorer runs). CLI: `llb bench-tooling --tool-protocol native|text` (native
  needs a running endpoint via `--base-url` / Ollama). `llb.bench.mcp_server` serves the SAME
  catalog over the official `mcp` Python SDK (`mcp_tool_specs` is the pure ToolDef->MCP mapping;
  `build_mcp_server` / `serve_stdio` lazily build a low-level server). CLI: `llb serve-tools-mcp`;
  the `mcp` SDK is an opt-in `[mcp]` extra.
- **BFCL UA adaptation.** `llb.prep.tooling_sources.from_bfcl` maps BFCL function-doc entries
  (+ optional possible-answers) into a `{tools, cases}` bundle; BFCL's several-acceptable-values-
  per-arg map onto the scorer's `oneof` tolerance. UA adaptation is injectable (`translate`); the
  schemas are kept verbatim. CLI: `llb adapt-bfcl --functions-file ... [--answers-file ...]`.
- **Per-argument tolerance.** `arguments_match` takes an optional `arg_match` spec relaxing a single
  argument to `contains` / `fuzzy` (difflib, stdlib) / `numeric` (abs tol) / `oneof` (default stays
  exact). The committed `samples/tooling_cases_uk.json` grew from 8 to 12 cases exercising each
  mode. The cases still need the human verification gate human sample-verify before headline use.

### agentic benchmark Agentic workflows benchmark -- `llb.bench.{tool_world,agentic}`
The agentic loop EXTENDS the text analysis multi-hop controller pattern with tool calls + an in-sandbox
execution step. `llb.bench.tool_world` is the deterministic sandbox (no tau-bench / AgentBench): a
mock filesystem, a mock key-value DB, substring `search` over a small UA corpus, and a `calculator`
backed by a SAFE restricted-AST evaluator (`safe_eval` allows only numbers + arithmetic operators
+ parentheses -- no names/calls/imports). Each tool is a pure `(world, args) -> observation`
mutating only the in-memory `ToolWorld`, so a task's success is checkable from the final env-state.

`llb.bench.agentic.run_episode` is the harness loop: each step the model emits one tool call
(reusing the tooling benchmark `parse_tool_call`), the world EXECUTES it, the observation is fed back, and the
loop runs until the model calls `finish` (or answers in prose) or the step budget is exhausted.
`check_success` is an OBJECTIVE assertion over the final env-state / answer (`file_equals` /
`file_contains` / `db_equals` / `answer_contains`; ALL must hold; an empty assertion list never
passes). `run_agentic` aggregates completion-rate as the headline `objective_score` under
`TIER_AGENTIC` (per-task success -> the bootstrap CI), records trajectory length + tool-call count
as efficiency, and persists the manifest. A committed UA task set (`samples/agentic_tasks_uk.json`,
4 tasks) ships; `load_tasks_file` loads it. CLI: `llb bench-agentic`. The loop is the pure,
langgraph-free form of the controller->execute->controller cycle, unit-tested from a scripted fake
tool-calling endpoint (good agent solves tasks, failing agent does not; budget-exhaustion -> typed
`incomplete`), no GPU.

An OPT-IN gated TRAJECTORY-QUALITY signal (a check the env-state assertions cannot make) is wired
exactly like the category expansion summarization faithfulness signal, through the same `run_gated_judge` seam:
each episode becomes one judge record whose retrieval context is the trajectory's TOOL OBSERVATIONS
and whose answer is the agent's final answer, so the judge's faithfulness (answer grounded in what
the tools returned) and answer-relevancy (answer addresses the goal) average into one
`trajectory_quality` scalar (`trajectory_quality` / `_trajectory_records`). It is recorded
ALONGSIDE completion-rate -- per-case + mean + CI, with a `JudgeStatus` in the manifest -- and ONLY
when the judge is configured AND trusted (`--judge-rho >= 0.6`, the judge calibration gate gate); otherwise the judge
is demoted and completion-rate ranks alone. The `judge_scorer` is injectable, so the wiring is
proven with a fake judge (no DeepEval / endpoint / GPU). CLI: `--judge-model` / `--judge-rho` /
`--judge-base-url`.

**Task-set breadth (agentic benchmark, delivered).** `llb.bench.agentic_tasks` GROWS the committed seed with
real-UA-corpus SEARCH tasks whose success assertion is computed PURELY from the corpus -- no human
gold authoring to BUILD: `count` ("how many docs mention X?" -> document frequency of X) and
`locate` ("which doc mentions X?" -> the single doc id, only for terms in EXACTLY ONE doc, so the
answer is unambiguous). Query terms are DERIVED from the corpus by document frequency (UA-stopword
filtered) or supplied explicitly; each task drops straight into the `bench-agentic` loop +
`check_success` via the sandbox `search` tool. CLI: `llb prepare-agentic-search --corpus-root ...
[--merge-seed]`. Pure + unit-tested; tasks still need the human verification gate sample-verify.

The trajectory-quality judge stays gated by the existing judge calibration gate calibration; corpus/task-specific
judge tuning is part of future corpus onboarding and prompt-system benchmark runs, while objective
completion-rate remains the ranking floor. The `build_agentic_graph` LangGraph wrapper and the
LangGraph-vs-CrewAI harness comparison are scoped to extended workflow (the remaining frameworks --
LangChain / LlamaIndex / Haystack / AutoGen -- stay out of scope; see `plan.md`).

### category expansion Remaining taxonomy -- summarization / structured output / chat-period / reliability
The remaining spec categories, each on the shared `bench.common` substrate:
- **summarization (`llb.bench.summarization`, `TIER_SUMMARIZATION`)** -- reference coverage via the
  PINNED-embedder cosine (NOT ROUGE): for each reference-summary sentence, the best cosine to any
  candidate sentence, averaged (`reference_coverage`; `similarity` injected, same basis as retrieval
  + the text-analysis matcher). Headline is mean coverage with a CI. An OPT-IN gated-judge
  FAITHFULNESS signal (summary vs source) is wired via `run_gated_judge` (the judge calibration gate-calibrated
  faithfulness metric): when configured AND trusted (`--judge-rho >= 0.6`) it records per-case +
  mean + CI faithfulness ALONGSIDE coverage (never folded into the headline; the manifest carries a
  `JudgeStatus`), else the judge is demoted and coverage ranks alone. Committed cases
  `samples/summarization_cases_uk.json`; CLI `bench-summarization` (`--judge-model` / `--judge-rho`
  / `--judge-base-url`).
- **structured output (`llb.scoring.structured` + `llb.bench.structured`, `TIER_STRUCTURED`)** --
  objective JSON-schema conformance via PYDANTIC (`build_model` from a field schema; no new
  `jsonschema` dep) + field accuracy. Schemas may be NESTED: `_field_type` recurses so a
  `type: object` field with `fields` builds a nested model and a `type: array` field with `items`
  builds a typed `list[...]`, so conformance validates the whole shape. Field accuracy recurses too
  (`_compare`): it counts matching expected LEAF values across nested objects + array items, with
  strings casefold/strip-insensitive and numbers exact unless the field spec relaxes it. Per-field
  tolerance (category expansion): numeric `tolerance` (abs) / `rel_tolerance` (relative); string `string_match`
  `fuzzy` (difflib ratio >= `threshold`) / `contains`; and array `unordered: true` for order-
  insensitive (greedy best-assignment) set matching. A non-conformant output scores 0 field
  accuracy; the headline is field accuracy, conformance rate recorded alongside, both with CIs. The
  committed `samples/structured_cases_uk.json` grew from 3 flat to 6 cases incl. nested objects,
  arrays of objects, an unordered tag set, and fuzzy/relative tolerance. CLI `bench-structured`.
- **chat-period analysis** -- DELIVERED BY REUSE + the chat-specific producers in
  `llb.prep.chat_corpus`: a chat-log-shaped SYNTHETIC planter (`prepare_synthetic_chat_corpus`,
  reusing the text-analysis flow with a chat prompt) and REAL chat-corpus ingestion
  (`ingest_chat_corpus`: parse an export -- array / Telegram / JSONL -- render a chat-shaped doc,
  then DRAFT grounded labels with a LOCAL completion, NO egress per OQ-egress). Both write a bundle
  the `bench.text_analysis` runner scores; the runner's `synthetic` flag keeps real (`synthetic:
  false`) and synthetic results reported SEPARATELY. CLI: `llb ingest-chat-corpus` (real, local-only)
  and `llb prepare-synthetic-corpus --text-analysis --chat` (synthetic).
- **reliability (`llb.scoring.reliability`)** -- rolls the existing TYPED failure taxonomy
  (ok/empty/malformed/refusal/timeout/backend_error/retrieval_miss/...) from ANY run's per-case
  scores into a first-class reliability score + per-failure-type breakdown (`reliability_report`);
  `read_case_statuses` reads a run bundle's `scores.parquet`/`scores.jsonl`. CLI `bench-reliability
  --run-dir`. Pure + unit-tested.

All four score on a fixed seeded set with CIs and are pure/fake-endpoint unit-tested, no GPU.

**Guarded category suite composite headline (extended workflow support).** `llb.scoring.composite_builder`
builds a separate composite row only when every required category suite tier for a model is present,
has a reloadable per-case objective series for bootstrap CIs, and is stamped with
`data_verified=true`. The weights use the spec's
category suite-category proportions and renormalize over the category suite subset: text-analysis 20, summarization 10,
structured 10, security 10, agentic 10, tooling 5. Category runners now persist a standardized
per-case `objective_score`, and the `bench-*` commands accept `--data-verified` plus
`--verification-ref` so verified runs can be made composite-eligible. `llb.goldset.verify` validates
those references mechanically before model calls and before a verified manifest can be persisted:
accepted forms are a fully decided `verify_sample.csv` within tolerance, a `sample_manifest.json`
that points to one, or an accepted-ledger bundle whose items are all `verified=true`. Invalid
references render a detailed operator diagnostic (`format_verification_status`) with path, kind,
reason, worksheet or ledger statistics, failing strata or unverified ids, and the exact
`verify-review` / `verify-accept` / rerun instruction needed to make the reference usable.
`llb.board.categories` preserves category run metadata via `CategoryRunRecord` and re-checks verification
refs when building the composite; `load_category_composite` returns either ranked composite rows or
concrete blockers. CLI: `llb bench-composite` (diagnostic escape hatches: `--allow-unverified`,
`--allow-missing-ci`). The Makefile target `make composite-headline` chains all six required
category benches with `--data-verified --verification-ref ...`, then runs `llb bench-composite` as
the clean preflight. Its defaults point at the committed sample suite, including
`samples/text_analysis_bundle_uk` and category-specific sample refs under
`samples/verification/composite_samples/`, for local smoke/demo composite runs. Real headline runs
override those paths with frozen category bundles and their human verification gate artifacts. Streamlit shows the
composite section only when the verified, CI-capable suite is complete. Operator flow:
[`composite-headline.md`](../../guides/composite-headline.md).

**Text-analysis judged sub-tasks + long_doc + contradiction + board (category expansion, delivered).**
`run_text_analysis` now takes the opt-in gated judge (`--judge-model` / `--judge-rho` /
`--judge-base-url`): the free-form sub-tasks (narrative / insight) become judge records (intent +
extracted answer + the doc as context), and a `long_doc` label is answered through the MAP-REDUCE
template (`eval.map_reduce.run_map_reduce_text` -- a `complete: str->str` driver over split -> map ->
reduce). The judge's faithfulness+relevancy collapse to one `judged_quality` per record, aggregated
per doc + overall (mean + CI, a `JudgeStatus`), recorded ALONGSIDE the objective recovery headline,
never folded in, and only when trusted; else demoted. The matcher now uses a `contradiction`'s
paired-span `attrs` (`spans` / `span_a` + `span_b`): credit requires the prediction to cover BOTH
contradicting sides (min of the two side credits). The Streamlit board (`llb.board`) loads the category suite
category run bundles via `load_category_records` (grouped BY TIER, best run per model) and renders
each under its OWN Tier, never cross-ranked (the `aggregate` guard refuses a mixed-tier board).

Summarization's gated-judge faithfulness stays gated by the existing judge calibration gate calibration; corpus/task
judge tuning belongs with future corpus onboarding and prompt-system benchmark runs. All category expansion cases
need the human verification gate human sample-verify before headline use.

### verified-data hardening second-frontier cross-check (verified-data gate) -- `llb.prep.cross_check`
The ontology-assisted drafting data-prep residual the plan says "lands with category suite's first scored category": the in-pipeline
verified-data gate. Every AI-DRAFTED item is re-confirmed by a SECOND, independent endpoint
(different from the drafter) layered on cheap deterministic pre-checks: GROUNDED (a labeled span
still resolves via `ground_span`) + NON-CIRCULAR (the answer is not leaked in the question), then
the second frontier's SUPPORTED (the cited span supports the answer) + ANSWERABLE (the question is
sensible/answerable). The pre-checks run FIRST so a clearly-broken item never spends a frontier
call. The verifier is injectable (`SecondFrontierVerify`); `second_frontier_verify` builds the
litellm-backed default; `cross_check_goldset` produces a `CrossCheckReport` (per-item verdicts +
pass count). Passing does NOT set `verified=true` -- only the human human verification gate sample-verify does; the
cross-check gates which drafted items are even eligible and is the report a human samples. CLI:
`llb cross-check-goldset --goldset --corpus --model` (`make cross-check-goldset BUNDLE= CROSS_CHECK_MODEL=`).
Pure + unit-tested (no key).

### human verification gate human sample-verify tooling -- `llb.goldset.verify` + `llb.goldset.verify_session`
The codeable half of the human verification gate gate: the operator tooling that turns a drafted + cross-checked bundle
into an accepted ledger, the verification-side twin of the `calibration-*` trio (mirrors how
`judge/calibration.py` pairs with `judge/rate.py`). `llb.goldset.verify` is the pure half --
stratification (`provenance|split|source_doc_id`; the `synthetic` flag is a BUNDLE-level fact read
from `provenance.json`, since the canonical `GoldItem` carries no per-item synthetic tag),
deterministic proportional sampling with a floor of one per stratum, the acceptance-sampling
arithmetic (per-stratum + overall reject rate vs tolerance, plus `undecided_with_failures`),
atomic CSV worksheet I/O, and `emit_accepted_ledger` (accepted items with `verified=true` + their
copied corpus, so the flip is an ADOPTION by replacement, never a boolean edit). It detects the gold
file (`goldset.jsonl` or the planter's `planted_labels.jsonl`) and surfaces any `*.cross_check.json`
verdict as read-only `cc_*` context. `llb.goldset.verify_session` is the interactive reviewer
(`run_session` + pure `parse_command` / `format_card` / `first_undecided_index`): a per-item card
showing the cited span inside its corpus window, the four checks (`g/a/r/p` pass, uppercase fail),
`y`/`x` accept/reject, resumable CSV-as-state; the cross-check is hidden by default (anti-anchoring,
`--show-crosscheck` reveals it). Subcommands `sample` / `review` / `accept`
(`python -m llb.goldset.verify`); make shortcuts `verify-sample` / `verify-review` / `verify-accept`
(`BUNDLE=`, `VERIFY_WS=`, `VERIFY_N=`, `VERIFY_TOLERANCE=`). Fully unit-tested with injected
inputs/output -- no model/endpoint/GPU (`tests/test_goldset_verify.py`). The committed
`ua_squad_postedited_v1` set is the verified side of the ledger (`DEFAULT_VERIFIED_GOLDSET`), already
250/250 `verified=true`, so human verification gate has no open work against it; the gate fires per NEW drafted bundle.
Operator workflow: [`docs/guides/goldset-from-scratch.md`](../../guides/goldset-from-scratch.md) +
[`verification-tooling.md`](../../guides/verification-tooling.md).

### verified-data hardening run-path hardening (robust backend prep carry-overs) -- delivered

The host-dependent robust backend prep run-path residuals are delivered and host-validated (see the Real-host
verification below):
- **memory planner sliding-window KV + config override (`llb.backends.planner`).** KV is now SLIDING-WINDOW
  -aware: `attention_layer_split` / `kv_mib_at_context` / `max_context_for_kv` cap the KV of Gemma's
  sliding layers at `sliding_window` while the periodic full-attention layers grow with context, so
  a long context costs far less KV (piecewise, not linear). `arch_from_config` also reads
  `sliding_window` / `sliding_window_pattern` (and derives the period from a `layer_types` list), and
  `enrich_arch(spec, override=True)` lets a cached `config.json` OVERRIDE curated arch fields (the
  real served architecture wins), exposed as `list-models --trust-config`.
- **VRAM contention guard multi-GPU read + arch-derived KV headroom (`llb.backends.hardware` + `llb.executor.contention`).**
  `select_target_gpu` reads ALL GPUs and targets the `CUDA_VISIBLE_DEVICES` device (or the most-free
  one) instead of hard-coding GPU 0; `default_gpu_reader` uses it. `model_kv_headroom_mb` derives the
  abort headroom from the served arch (the KV at a minimal serving context, sliding-window-aware)
  rather than a fixed 512 MB floor, so a heavy-KV model is judged un-launchable at the right
  threshold.
- **vLLM serving preflight flashinfer auto-pin + sampler-in-manifest + driver re-probe (`llb.backends.preflight` +
  `vllm` + `telemetry`).** The verdict records the GPU `driver`; `verdict_is_current` invalidates a
  cached verdict when the driver changes, so `llb preflight-vllm` re-probes WITHOUT a full
  `build-vllm` (`--force` to re-run regardless). `auto_pin_flashinfer` installs + re-probes candidate
  flashinfer versions (`LLB_FLASHINFER_CANDIDATES`) when the bundled one fails -- OPT-IN behind
  `preflight-vllm --auto-pin`, since it changes the environment. The vLLM launcher records the
  sampler actually used (`flashinfer` | `native`) + the flashinfer version into the manifest
  telemetry.
- **llama.cpp launcher further `/props` shapes + drivable partial offload (`llb.backends.llamacpp` + `cli.eval`).**
  `parse_served_context` checks the known `n_ctx` locations across llama.cpp versions (top level,
  `default_generation_settings[.params|.context]`, `generation_settings`, `model`, `props`) and never
  mistakes `n_ctx_train`; `run-eval --gpu-layers N` drives a partial GPU/CPU split without a YAML.

**verified-data hardening ontology data-prep residuals (delivered).** The three data-prep items feeding the GraphRAG backend
extraction reuse:
- **spaCy / Stanza adapter.** `llb.prep.ontology.spacy_adapter.SpacyExtractionAdapter` implements
  the `ExtractionAdapter` seam over spaCy `uk_core_news` NER -- entities only (spaCy's strength),
  with exact spans from the ent's own offsets (a prefix of the doc -> exact into full text) and
  repeated surfaces grouped into one entity (lightweight coreference). spaCy is lazy/opt-in; `nlp`
  is injectable so the mapping is unit-tested with a fake. CLI: `llb prepare-goldset-draft
  --extractor spacy [--spacy-model ...]`.
- **Long-doc chunking for extraction.** `LLMExtractionAdapter` now CHUNKS a doc longer than
  `EXTRACT_MAX_CHARS` into overlapping windows (reusing `split_document`), extracts per window, and
  MERGES (`merge_extractions` dedups entities by (name,type) merging mentions+aliases; events/claims/
  facts by evidence span + payload) -- so a long doc's later content is no longer truncated away.
  Grounding still runs against the FULL text, so offsets stay exact.
- **Richer ontology-type confidence.** `induce_ontology` now blends normalized count with normalized
  DOCUMENT frequency (`CONFIDENCE_COUNT_WEIGHT` / `CONFIDENCE_DOCFREQ_WEIGHT`), so a type spread
  across docs outranks one of equal count concentrated in one doc; types sort by confidence.
  `ontology_constraints` carries the high-confidence types into every drafting prompt
  (`draft_prompt` / `draft_items` `ontology_hint`) as an explicit constraint.
