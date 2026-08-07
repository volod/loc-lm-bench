# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

The forward work is split into two sections by **who must act to complete it**:

- **[Agent Implementation Tasks](#agent-implementation-tasks)** land to `make ci` green with
  committed fixtures, injected fakes, and deterministic harnesses. A few carry a heavy real-model
  run for durable evidence; those runs are deterministic and execute on the CUDA host without any
  human judgment, so they stay in this section.
- **[Human-Assisted Tasks](#human-assisted-tasks)** cannot reach their stated acceptance without a
  human in the loop: the deliverable *is* human judgment (verification-gate reviews, drafting
  oversight, measured reviewer throughput) or requires a human authorization (egress consent +
  API spend). An agent can still build the supporting code and unit tests; the marked **human
  step** is what gates completion.

Task numbers are stable ids and never change; every task carries an explicit `Dependencies` line,
and the recommended build order within each section follows those lines.

For remaining tasks that depend on retrieval behavior, use the current RAG baseline documented in
[RAG core](current/rag-core.md) and the mixed-corpus ingestion baseline documented in
[data prep](current/data-prep.md). For tasks that depend on scoring or judging, the calibrated
local-judge baseline and tuning/sweep behavior live in
[evaluation rigor](current/rigor-board-judge.md); the prompt-system package flow and other
extended workflows live in [extended workflows](current/extended-workflows.md).

Every task below carries an explicit `Agent status` line with one of four markers:

- **CLEAR** -- agent-buildable to `make ci` green with fixtures/fakes; no run evidence, no human
  gate.
- **RUN NEEDED** -- agent-buildable, but acceptance requires a heavy deterministic run; every dev
  box is a proper CUDA host, so the agent executes these runs itself on the current machine.
- **BLOCKED BY HUMAN** -- the acceptance gate consumes an artifact only a human step can produce.
- **HUMAN-GATED** -- the deliverable itself is human judgment or authorization; supporting code and
  unit tests are agent-buildable.

## Agent Implementation Tasks

Add new agent-buildable work here per [Adding Future Tasks](#adding-future-tasks).

### agent-policy-change-replay-records-the-refused-prompt (optional)

The replay records a prompt by intercepting the injected `complete`, so a prompt the guard REFUSES
is built, never sent, and never compared: `run_episode` checks `budget.fits` before the model call
and ends the episode as `context_overflow`. Two arms that overflow at the same step therefore read as
byte-identical whatever the refused prompts measured, and the audit reports `prompt_invariant` for a
change that moved the very prompt that ended the run
([extended workflows](current/extended-workflows.md#the-audit-runs-in-ci-on-the-act-that-creates-the-problem)).
Cap-fitting cells are chosen not to overflow, but nothing in the audit asserts that, and the
observation cap's own band conditions now name the refused prompt as the thing that keeps a
discarded size unobservable
([extended workflows](current/extended-workflows.md#the-caps-silence-is-about-the-prompts-the-loop-builds)),
so the replay should be able to see it. Record the refused prompt (or its char count plus the
terminal status) alongside the sent ones, compare it in `arm_comparison`, and add the fixture case
that a change moving only an overflowing prompt no longer reads as invariant.

- Agent status: CLEAR
- Dependencies: the recording seam is `replay_prompts` in
  `src/llb/bench/agentic_policy_change_replay.py`; the refusal is the `budget.fits` branch of
  `run_episode` in `src/llb/bench/agentic/episode.py`, which already stamps `prompt_chars` for the
  prompt it refuses.
- User-visible outcome: "this change invalidates nothing" covers the prompt that ended the episode,
  not only the ones a model saw.
- Scope boundary: in scope -- the recorded refusal, its comparison, and the fixture case. Out of
  scope -- changing the overflow rule itself, changing any shipped constant, and re-running a
  published cell.
- Documentation target:
  [extended workflows](current/extended-workflows.md#the-audit-runs-in-ci-on-the-act-that-creates-the-problem).

### agent-policy-change-interaction-band-skips-the-unfoldable-first-step (optional)

`reachable_fold_steps` calls step 1 a reachable fold step because some trigger sits below the first
prompt, but no episode can fold there: the step-1 prompt is built from ZERO entries and
`compact_state` returns false with nothing to fold. Every coupling therefore states three conditions
about a step the loop never folds at, and because `_blocking_reasons` deduplicates by condition name
the vacuous step-1 row is the one that LEADS the no-band report -- the reader's first line is the
least informative one available
([extended workflows](current/extended-workflows.md#one-pair-separates-and-the-other-fourteen-are-answered)).
Restrict the solved steps to those with at least one live entry
(`live_entries_at_fold_step(step) >= 1`), and check that no solved band moves as a result.

- Agent status: CLEAR
- Dependencies: the step list is `reachable_fold_steps` in
  `src/llb/bench/agentic_memory_boundary_probe.py`, consumed by `separating_guard_bands` in
  `src/llb/bench/agentic_policy_change_interaction_band.py`; the entry count is
  `live_entries_at_fold_step` in `..._interaction_terms.py`.
- User-visible outcome: a blocked-step report opens with a fold that can actually happen, so a
  drifted geometry is read from the first line rather than the third.
- Scope boundary: in scope -- the step filter, the report, and the test that the two committed bands
  are unchanged. Out of scope -- changing any shipped constant and re-running a published cell.
- Documentation target:
  [extended workflows](current/extended-workflows.md#one-pair-separates-and-the-other-fourteen-are-answered).

### agent-policy-change-interaction-scan-sweeps-the-moved-values (optional)

The replay scan asks each pair about ONE concrete move per field (`FIELD_MOVES`, plus a second
alternative set run by hand), so "no geometry separates this pair" is backed at two points of a
value space the audit accepts continuously
([extended workflows](current/extended-workflows.md#one-pair-separates-and-the-other-fourteen-are-answered)).
The separating pair shows why that matters: the band exists for `compact_share` 0.5 -> 0.48 and
vanishes for 0.5 -> 0.55, because the direction of the move decides whether the candidate elides. Let
the scan sweep the moved VALUES as well as the guards -- a small grid per field, sized so the whole
sweep still fits a `slow` test -- and record which pairs stay silent across it.

- Agent status: CLEAR
- Dependencies: the per-field moves are `FIELD_MOVES` in
  `src/llb/bench/agentic_policy_change_interaction_couplings.py`; the scan is
  `scan_separating_cells` in `src/llb/bench/agentic_policy_change_interaction_scan.py`.
- User-visible outcome: the enumeration's negative answers hold across the values a commit could
  plausibly ship, not only the neighbour the fixture happens to name.
- Scope boundary: in scope -- the value grid, the scan runtime budget, and the recorded result. Out
  of scope -- changing any shipped constant and re-running a published cell.
- Documentation target:
  [extended workflows](current/extended-workflows.md#one-pair-separates-and-the-other-fourteen-are-answered).

### agent-policy-change-interaction-band-past-the-first-fold (optional)

The band solver reports a fold step only when the episode compacts exactly ONCE there, because
`summary_input_chars` is a sum over folds and the elision inequality is a statement about one
offered transcript
([extended workflows](current/extended-workflows.md#the-compound-guarantee-has-a-geometry-that-tests-it)).
Every multi-fold geometry -- a deep episode under a small guard, which is the ordinary case away
from the cap-fitting band -- is therefore reported as "no band" whether or not one exists, so the
solver's negative answer is weaker than it reads. Give the probe a per-fold breakdown (the telemetry
already counts `n_compactions`; the offered spans want recording per fold rather than summed), state
the inequality against the FIRST fold whose elision the candidate share flips, and either widen the
solved bands or record that the extra folds never separate.

- Agent status: CLEAR
- Dependencies: the summed telemetry is `summary_input_chars` in `ContextTelemetry`
  (`src/llb/bench/agentic/context.py`), surfaced by `compact_fold_input_probe` in
  `src/llb/bench/agentic_memory_boundary_probe.py`; the refusal is `_offered_at_fold_step` in
  `src/llb/bench/agentic_policy_change_interaction_band.py`.
- User-visible outcome: "no separating band at this depth" means no band exists rather than none
  the solver can see.
- Scope boundary: in scope -- per-fold offered spans, the multi-fold band arithmetic, and the test
  that a known single-fold band is unchanged by the generalization. Out of scope -- changing any
  shipped constant and re-running a published cell.
- Documentation target:
  [extended workflows](current/extended-workflows.md#the-compound-guarantee-has-a-geometry-that-tests-it).

### agent-policy-change-replay-untouched-fields-from-the-pins (optional)

The replay builds each arm from three sources: two fields out of the design's `held_fixed`, one out
of the cell, and the remaining three out of the shipped dataclass defaults (`_policy` in
`src/llb/bench/agentic_policy_change_replay.py`). For a field the change moves that is irrelevant --
the override wins -- but for a field it does NOT move the baseline arm is only the pinned policy
because every design happens to `agree` with its pin today. A pin marked `restated` for
`observation_cap_chars` or `observation_head_share` would silently make the baseline arm replay the
design's stale value instead of the pinned one, which is the same class of bug the compound audit
just closed, one level down
([extended workflows](current/extended-workflows.md#the-audit-runs-in-ci-on-the-act-that-creates-the-problem)).
Feed the untouched fields from the PINS when the caller has them (the gate always does), keep the
design values as the fallback for a hand-run CLI audit, and add the fixture case that proves a
`restated` pin on a held field moves the baseline arm.

- Agent status: CLEAR
- Dependencies: the pins already carry the full pinned policy and their `designs` claim
  (`samples/benchmarks/agentic_context_policy_pins.json`, checked by
  `src/llb/bench/agentic_policy_pin_gate.py`); the seam is `_policy` in
  `src/llb/bench/agentic_policy_change_replay.py`.
- User-visible outcome: the baseline arm is the policy the published numbers were measured under for
  every field, not only for the fields the designs happen to agree on.
- Scope boundary: in scope -- the pinned-policy source for untouched fields, the CLI fallback, and
  the `restated`-pin fixture case. Out of scope -- changing any shipped constant or any pin value,
  and re-running a published cell.
- Documentation target:
  [extended workflows](current/extended-workflows.md#the-audit-runs-in-ci-on-the-act-that-creates-the-problem).

### agent-published-number-provenance-pins (optional)

The pin gate names the invalidated CELLS and the doc sections that publish their numbers, but the
numbers themselves are prose: nothing ties `21862` in the restatement table or `+1610.3` in the
fold-step table to the cell and the run artifact it came from, so a failure still leaves a human to
find every affected figure by reading. Extend the pinning idea from constants to published values --
a committed provenance fixture mapping each published agentic number to `(study kind, cell id,
artifact path, metric)` -- and have the gate print the exact figures a drifted constant retires,
not only the cells. The same fixture makes a second check cheap: assert every mapped artifact path
still resolves, which catches a number whose evidence was garbage-collected.

- Agent status: CLEAR
- Dependencies: the cell ids and re-run scope come from
  `src/llb/bench/agentic_policy_pin_gate.py`; the artifact paths are the run roots already recorded
  in the evidence sections of
  [extended workflows](current/extended-workflows.md#cap-fitting-boundary-surface).
- User-visible outcome: a drifted constant fails CI with the LIST OF FIGURES to restate, so nobody
  greps the docs to find what a change retired.
- Scope boundary: in scope -- the provenance fixture, the figure list in the gate message, and the
  artifact-path resolution check. Out of scope -- rewriting any published figure automatically,
  re-running cells, and provenance for non-agentic evidence.
- Documentation target:
  [extended workflows](current/extended-workflows.md#the-audit-runs-in-ci-on-the-act-that-creates-the-problem).

### agent-policy-change-audit-coverage-beyond-cap-fitting (optional)

The audit walks the three cap-fitting memory studies (22 cells) and nothing else, so "this change
invalidates NO published number" is only true of that slice
([extended workflows](current/extended-workflows.md#what-a-policy-constant-change-invalidates)).
The context-policy constant sweep, the keep-long lane, and the harness-comparison rows rest on the
same constants and are not walked, and the `keep_last_n` result advertises the gap: the audit calls
keep=1 free precisely because no cap-fitting cell runs that policy, while the sweep that EXPOSED
keep=1 is built on cells that do. Extend the audit's study registry to those lanes -- each needs its
own cell-geometry reader and a task builder other than the memory-chain one -- so the invariance
answer covers the evidence a `keep_last_n` or observation-cap change actually threatens. The CI pin
gate now fails a build on that same registry
([extended workflows](current/extended-workflows.md#the-audit-runs-in-ci-on-the-act-that-creates-the-problem)),
so widening it widens the gate's re-run scope at no extra wiring: both read
`AUDITED_DESIGN_PATHS` in `src/llb/bench/agentic_policy_change_audit.py`.

- Agent status: CLEAR
- Dependencies: the audit's per-kind geometry extraction
  (`declared_geometry` in `src/llb/bench/agentic_policy_change_audit.py`), which currently hardcodes
  the memory-chain task builder in its replay.
- User-visible outcome: the "invalidates nothing" verdict means the whole agentic evidence base
  rather than one family of studies, so an operator can trust it without knowing which studies were
  walked.
- Scope boundary: in scope -- a task-builder seam per registered study, geometry readers for the
  sweep and keep-long lanes, and their cells in the audit. Out of scope -- re-running anything the
  wider audit invalidates, and changing any shipped constant.
- Documentation target:
  [extended workflows](current/extended-workflows.md#what-a-policy-constant-change-invalidates).

### agent-context-policy-imperfect-play-guard-margin (optional)

The deterministic cap-peak probe
(`src/llb/bench/agentic_memory_boundary_probe.py`) walks the workflow with an ORACLE controller, so
the band it certifies is the perfect-play band: a real controller that repeats a step or mis-reads a
token grows the transcript past that peak, and a guard chosen just above it can still overflow on
the run. Price that gap instead of leaving it implicit: extend the probe to the worst case the step
budget allows (max steps rather than depth), record the measured extra steps per episode from the
existing bundles, and turn the difference into a stated safety margin the design validation applies
when it certifies a cell as cap-fitting. The same probe now also certifies published cells as
bound-invariant ([extended workflows](current/extended-workflows.md#published-crossovers-under-the-shipped-cap)),
which inherits the identical perfect-play limitation: a longer real transcript can reach a
summarize-input cap the oracle transcript never touched, so extend the worst-case probe to that
verdict too and state the invariance for the worst case the step budget allows.

- Agent status: CLEAR
- Dependencies: the probe, the band check in
  `src/llb/bench/agentic_memory_boundary_surface_cells.py`, and the invariance verdict in
  `src/llb/bench/agentic_memory_cap_audit.py`; the per-episode step counts are already persisted in
  the compact-vs-cap bundles.
- User-visible outcome: a predeclared cap-fitting cell that is cap-fitting for the model that
  actually runs it, not only for a perfect controller, and a bound-invariance verdict that holds for
  the transcripts a real controller produces.
- Scope boundary: in scope -- the worst-case probe, the margin constant, the validation change, and
  the worst-case invariance verdict. Out of scope -- re-running the surface, changing the
  interpolation rule, or relaxing the activation floor.
- Documentation target:
  [extended workflows](current/extended-workflows.md#cap-fitting-boundary-surface).

### agent-context-policy-summary-elision-under-the-window-bound (optional)

The step-aligned summarize-input bound elides the folded transcript ONLY when that transcript cannot
fit the resolved window, which no cap-fitting ladder reaches: every cell measured so far folds a
transcript comfortably under the guard, so the shipped bound's elision path is unexercised
([extended workflows](current/extended-workflows.md#the-summarize-input-cap-is-step-aligned)). That
is the regime where an elision is unavoidable rather than incidental, and it is the one where the
completion cost of losing the middle of a folded transcript actually matters. Build a geometry whose
folded transcript EXCEEDS the window minus the summary template (deeper memory chains, or a larger
`pad_chars` at a fixed window), verify with the deterministic probe that the shipped bound elides
there, and read completion against a control whose transcript fits -- the answer says whether an
unavoidable elision needs a smarter fold (per-entry budgets, oldest-first dropping) rather than a
head-and-tail trim.

- Agent status: RUN NEEDED
- Dependencies: `compact_fold_input_probe` in `src/llb/bench/agentic_memory_boundary_probe.py`
  predicts the elided span with no model, so the geometry is checkable before a GPU is warmed.
- User-visible outcome: an operator running a transcript too big to summarize whole learns what that
  costs, instead of finding out through a wrong answer read from a middle-elided summary.
- Scope boundary: in scope -- the over-window geometry, the probe-backed predeclaration, and the
  completion reading. Out of scope -- implementing a new folding strategy (that is what the reading
  would justify), and changing the shipped bound.
- Documentation target:
  [extended workflows](current/extended-workflows.md#the-summarize-input-cap-is-step-aligned).

### agent-context-policy-hysteresis-second-fold (optional)

Every cap-fitting cell measured so far folds EXACTLY once per episode, which is why the guard drops
out of the cost once the trigger is fixed: after the first summary, trigger hysteresis raises the
next trigger to the full guard, and no tested transcript grows back that far
([extended workflows](current/extended-workflows.md#the-routing-rule-lives-on-the-trigger-axis)).
The trigger-only rule is therefore established only in the one-fold regime, and the regime where
compact is most interesting -- long agent sessions that fold repeatedly -- is unmeasured. Push depth
(or shrink the guard toward the cap peak) until at least two folds fire per episode, then re-run one
equal-trigger family: if the deltas separate there, the guard re-enters through hysteresis and the
portable rule needs a stated validity limit. The second fold also carries the running summary into
the summarize input, and the shipped `window` bound sizes that input from the budget rather than the
trigger, so record the per-fold summarize input beside the deltas -- a growing prior summary is the
other way the guard could re-enter.

- Agent status: RUN NEEDED
- Dependencies: the deterministic probe predicts the post-fold prompt growth that a second trigger
  crossing requires (`compact_fold_input_probe` reports the summarize input per fold); reuse the
  collapse design, the cell gate, and the equivalence band unchanged.
- User-visible outcome: either the trigger-only routing rule extended to repeated compaction, or an
  explicit "one fold only" boundary on the rule an operator would otherwise over-apply.
- Scope boundary: in scope -- a depth/guard geometry that forces two or more folds, one equal-trigger
  family inside it, and the validity statement. Out of scope -- new families, new task shapes, and
  changing shipped compaction hysteresis.
- Documentation target:
  [extended workflows](current/extended-workflows.md#the-routing-rule-lives-on-the-trigger-axis).

### agent-operating-profile-recommendation

Every ingredient of an agent configuration is measured somewhere in this repo and nowhere together:
`llb recommend` renders host-adaptive model picks plus separate miss-analysis, self-improvement,
fine-tune-campaign, and context-policy sections (`src/llb/board/recommend/sections.py`), the
context-order recommendation comes out of the position probe, the retrieval knobs out of the
comparison lanes, the prompt-system id out of `prompt-system-compare`, and the adapter out of the
registry. An operator wanting to stand up an agent must read five sections and hand-assemble, and
nothing checks that the pieces were measured on the same corpus, store, or model. Compose them:
`llb recommend --agent-profile` emits ONE `agent_profile.json` plus a markdown rationale naming
model and backend, prompt-system id, adapter (or none), context policy, context order, `top_k` /
reranker / context budget, and the loop policy. Each field carries its value, the artifact path the
value came from, that lane's own verdict and uncertainty, and its freshness -- and a field whose
lane never ran is emitted as `unmeasured`, never as a default dressed up as a recommendation, which
is the whole failure mode a composed profile invites.

- Agent status: RUN NEEDED
- Dependencies: the
  [agent loop-policy recommendation](current/extended-workflows.md#agent-loop-policy-recommendation)
  supplies the loop-policy field; the context-policy field comes from the `agentic-context` bundles
  ([extended workflows](current/extended-workflows.md#agent-context-management-policies)), and for
  memory-dependent work its guard-dependent routing rule comes from the cap-fitting boundary surface
  ([extended workflows](current/extended-workflows.md#cap-fitting-boundary-surface)); the
  rest are current behavior. Reuse `src/llb/board/recommend/`
  (sections, build, render), the adapter registry's `staleness()` and its retrieval-fingerprint axis
  ([extended workflows](current/extended-workflows.md#staleness)), and the shared borderline
  vocabulary in `src/llb/rag/fusion_evidence/stability.py` so a field resting on a knife-edge row is
  marked the same way every lane marks it.
- User-visible outcome: one artifact an operator (or a runtime) can act on, where every recommended
  value is traceable to the run that measured it and every gap is visible as a gap.
- Scope boundary: in scope -- the composition, the per-field evidence/verdict/freshness record, the
  `unmeasured` state, the consistency guard (fields measured against a different corpus, store
  fingerprint, or model are refused rather than mixed), the JSON schema, and the markdown rationale.
  Out of scope -- running any lane on the operator's behalf, inventing a value for an unmeasured
  field, ranking policy changes, and shipping a runtime that consumes the profile.
- Data and artifact paths: `$DATA_DIR/agent-profile/<run>/{agent_profile.json,profile.md}`, composed
  from the existing per-lane roots; no new evidence root.
- Execution path: `make recommend-agent-profile` after the component lanes have run on the CUDA
  host; CI covers composition, the `unmeasured` state, the consistency guard, and the staleness
  demotion over fixture bundles -- no GPU.
- Acceptance gates: `make ci` green; a profile built with no bundles at all is entirely `unmeasured`
  and still a valid artifact; every populated field's evidence path resolves and its verdict matches
  the lane artifact it cites; a stale adapter or a store whose retrieval fingerprint changed demotes
  every field that depends on it, with the changed knob named; the recommended values replay as
  `run-eval` / `bench-agentic` flags that reproduce the recommended configuration.
- Documentation target: a recommendation-composition section in
  [extended workflows](current/extended-workflows.md) and the recommendation entry in
  [overview](current/overview.md).

### reranker-bake-off

The cross-encoder is pinned to one model (`BAAI/bge-reranker-v2-m3`, `DEFAULT_RERANKER` in
`src/llb/rag/rerank.py`) and has never been compared with anything, while the adoption evidence shows
the reranked cell is where a retrieval change actually reaches the answer for some models
([RAG core](current/rag-core.md#the-scoped-first-hit-rank-adoption-bar)). A reranker is also the
cheapest place to buy first-hit rank on a 16 GiB host, and the multilingual cross-encoder field has
moved. Bake off the current candidates that cover Ukrainian -- `BAAI/bge-reranker-v2-m3` (incumbent),
`jinaai/jina-reranker-v2-base-multilingual`, `Alibaba-NLP/gte-multilingual-reranker-base`,
`mixedbread-ai/mxbai-rerank-base-v2`, `Qwen/Qwen3-Reranker-0.6B` -- on the accepted ledger at a fixed
encoder and chunking, reporting recall@k / MRR / first-hit rank with the standard paired verdict plus
the cost columns a reranker is actually chosen on (rerank latency per query, VRAM while the generator
is resident).

- Agent status: RUN NEEDED
- Dependencies: reuse the paired lane and verdict machinery documented in
  [RAG core](current/rag-core.md#paired-lane-uncertainty-and-verdict); this task feeds
  `embedder-decision-on-a-resolvable-item-set`. Reuse `CrossEncoderReranker` and the `+rerank` row
  seam in `src/llb/rag/compare.py`.
- User-visible outcome: the shipped reranker is a measured choice with a cost, not a default nobody
  has questioned, and the operator learns whether the rank gain is worth the second model in VRAM.
- Scope boundary: in scope -- the candidate lane, the paired verdict, the latency/VRAM columns, and a
  keep-or-swap recommendation. Out of scope -- reranker fine-tuning, hosted rerankers, listwise/LLM
  rerankers, and changing `rerank_candidates` defaults before the bake-off supports it.
- Data and artifact paths: `$DATA_DIR/compare-rerankers/<run>/{report.md,report.json}`.
- Execution path: `make compare-rerankers GOLDSET=<accepted> CORPUS=<dir> NOISE_FLOOR=1` on the CUDA
  host; a candidate that needs `trust_remote_code` is refused unless explicitly opted into, and a
  candidate that does not fit beside the generator is reported as skipped with its measured
  footprint rather than silently omitted. CI covers scoring, ranking, and the verdict over a fake
  cross-encoder -- no download, no GPU.
- Acceptance gates: `make ci` green; every candidate is scored on the identical item set at the same
  seed with its own documented query/passage input format; each row carries a paired delta against
  the incumbent plus rerank latency; the report states keep or swap and names the cost of the swap.
- Documentation target: [RAG core](current/rag-core.md#reranking-and-context-order-rerank-context-order)
  and the recommendation line in [platform matrix](current/platform-vector-matrix.md).

### embedder-candidate-roster-refresh

The bake-off's default candidate list is the 2023-2024 multilingual generation, and the paired lane
now says the choice is undecidable on the item sets the repo has partly because the candidates are
close together ([RAG core](current/rag-core.md#the-recommendation-re-read-with-paired-uncertainty)).
Add the current multilingual retrieval encoders that fit a 16 GiB host beside the incumbents --
`intfloat/multilingual-e5-large-instruct`, `Alibaba-NLP/gte-multilingual-base`,
`jinaai/jina-embeddings-v3`, `Qwen/Qwen3-Embedding-0.6B` -- and register each one's convention
FIRST. That registration is the substance of the task, not a detail: `embedding_family` resolves by
substring, so `multilingual-e5-large-instruct` currently resolves to the plain `e5` family and would
be scored with the plain `query:` / `passage:` prefixes instead of its instruction format, and an
unrecognized id
falls through to `plain` with no instruction at all -- the exact silent recall loss the family table
exists to prevent ([RAG core](current/rag-core.md#embedder-conventions-and-bake-off)).

- Agent status: RUN NEEDED
- Dependencies: none, but the decision it feeds is `embedder-decision-on-a-resolvable-item-set`.
  Reuse `src/llb/rag/embedding.py` (family table, query/passage conventions) and the bake-off lane
  unchanged.
- User-visible outcome: the Ukrainian embedder recommendation is made against the encoders an
  operator would actually consider today, each scored under its own documented convention.
- Scope boundary: in scope -- the family entries and their unit tests, the candidate list, VRAM/
  throughput/index-size measurement per candidate, and a re-run of the paired bake-off on both scored
  corpora. Out of scope -- fine-tuning (that is `ua-embedder-domain-finetune`), late-interaction /
  multi-vector retrieval, hosted API encoders beyond the existing opt-in row, and changing the
  shipped default before a candidate separates.
- Data and artifact paths: the existing `$DATA_DIR/compare-embeddings/<run>/` layout.
- Execution path: `make compare-embeddings MODELS=<roster> NOISE_FLOOR=1` on the CUDA host; a
  candidate requiring `trust_remote_code` is opt-in and recorded as such. CI covers each new family's
  query/passage convention against its model card's documented format and asserts that an unknown id
  does not silently resolve to `plain`.
- Acceptance gates: `make ci` green; every candidate's convention is unit-tested and cited; the
  incumbent rows reproduce their recorded numbers; each new row carries its paired delta, throughput,
  index size, and peak VRAM; the verdict is recorded as adopt, retain, or undecidable at the reached
  sample size.
- Documentation target: the embedder sections of [RAG core](current/rag-core.md) and
  [platform matrix](current/platform-vector-matrix.md#embedding-bake-off).

### cross-lingual-query-lane

The query-robustness lane perturbs CHARACTERS -- transliteration, apostrophe variants, mixed script,
keyboard typos ([evaluation rigor](current/rigor-board-judge.md#ukrainian-query-robustness-benchmark))
-- and never changes the LANGUAGE of the query. Ukrainian deployments routinely receive Russian and
code-switched questions against a Ukrainian corpus, and the repo already treats Russian as a
first-class second language on the security side, where a model that refuses in Ukrainian and
complies in Russian is a measured finding
([category suite](current/category-benchmark-suite.md)). Retrieval and answering have no equivalent.
Add a query-language lane: a committed Russian and mixed UA/RU variant of an existing gold set's
QUESTIONS with the gold spans and documents unchanged (so retrieval is scored by the same
source-span metric), then report recall@k, MRR, and answer quality per language against the
Ukrainian baseline.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the variant-class seam in `src/llb/eval/query_robustness_variants.py`,
  the lane runner, and the per-class reporting; the drafted variants ride the existing
  drafted-grounding rules until a reviewer accepts them.
- User-visible outcome: an operator learns whether their Ukrainian RAG stack answers a Russian
  question about a Ukrainian document, and whether the loss is in retrieval or in generation.
- Scope boundary: in scope -- the committed variant fixture, the language lane, the per-language
  retrieval and answer reporting, and a mitigation reading (does query normalization or translation
  in `query_prep` recover the loss). Out of scope -- multilingual corpora, translating the CORPUS,
  a translation model in the shipped query path before the measurement supports it, and any change
  to the security lane's Russian probes.
- Data and artifact paths: `samples/goldsets/<fixture>_ru/` for the committed variants;
  `$DATA_DIR/query-robustness/<run>/` for the lane output.
- Execution path: `make bench-query-robustness QUERY_ROBUSTNESS_CLASSES=language_ru,language_mixed`
  on the CUDA host; CI covers variant generation, the unchanged-span invariant, and the per-language
  report over fixtures.
- Acceptance gates: `make ci` green; every variant item keeps its gold spans byte-identical and
  passes `validate-goldset`; the report carries recall@k and answer quality per language with paired
  intervals against the Ukrainian baseline; the reading states whether the loss is retrieval-side or
  answer-side, and the fixture is marked drafted until a reviewer accepts it.
- Documentation target: the query-robustness section of
  [evaluation rigor](current/rigor-board-judge.md) and the query-side processing section of
  [RAG core](current/rag-core.md).

### vector-store-bake-off-paired-uncertainty (optional)

`compare-vector-stores` still ranks backends on a point estimate plus the measurement floor, the
one reading the embedder bake-off already carries: its `best (recall@k)` line is label order when
the backends tie ([platform matrix](current/platform-vector-matrix.md#embedding-bake-off)), and
nothing states how large a backend difference the item set could even resolve. Give it the same
paired lane -- per-item metric vectors against a baseline backend, shared resample index sets, the
delta interval and win/loss/tie ledger per row, and an adopt-or-retain verdict -- so a backend swap
is decided the same way an embedder swap now is.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `src/llb/rag/embedding_bakeoff_uncertainty.py` wholesale (it takes
  metric vectors, not embedder rows) and the store seam in `src/llb/cli/rag/compare_stores.py`.
- User-visible outcome: the operator learns whether a vector-backend difference is real or is the
  order the labels happened to sort in.
- Scope boundary: in scope -- the paired columns, the verdict, and a re-run on both scored
  corpora. Out of scope -- new backends and any change to the retrieval metrics.
- Data and artifact paths: the existing `$DATA_DIR/compare-vector-stores/<run>/` layout.
- Execution path: `make compare-vector-stores NOISE_FLOOR=1` on the CUDA host; CI covers the
  interval columns over fake stores.
- Acceptance gates: `make ci` green; every backend row carries a paired delta interval against the
  baseline backend and the report states adopt or retain.
- Documentation target: the vector-store section of
  [platform matrix](current/platform-vector-matrix.md).

### graph-lane-score-ties (optional)

The graph lane's own recall is decided by tie order for two thirds of the questions it is scored
on: its link-relevance scores saturate into long exact-tie blocks, the rank-k cut falls inside one
for 68 of 95 items (33 of 35 on the multi-hop slice), and that is what sets the fusion sweep's
whole measurement floor at `+/-0.021` recall@10 overall and `+/-0.043` on the focus slice
([GraphRAG](current/graphrag-backend.md#the-sweep-re-read-against-its-measurement-floor)). The
ranking is reproducible -- `_rank_dedup` breaks ties on `(doc_id, char_start, char_end)` -- but a
document id is not a relevance signal, so every graph-only row is quoted to three decimals it has
not earned. Find out whether the tie blocks are reducible: measure how much of each tie block is
one relevance value versus rounding, and if a finer signal exists (edge weight, hop distance,
mention count, community rank as a continuous term rather than a bucket) score the lane with it and
re-measure the floor.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the scoring in `src/llb/graph/linking.py` / `src/llb/graph/retrieval.py`
  and the floor lane (`compare-graph-fusion --noise-floor`) to read the result.
- User-visible outcome: either graph-only rows whose recall reflects relevance instead of document
  order, or a recorded finding that the lane's evidence is inherently tied and its rows must be
  read at the floor's precision.
- Scope boundary: in scope -- the tie-block census, one finer relevance signal if the census
  supports one, and the before/after floor. Out of scope -- changing the graph schema, the fusion
  mechanics, and the RRF rank basis (fused rows already report a zero band).
- Data and artifact paths: `$DATA_DIR/graph-vector-fusion-multihop/<run>/`.
- Execution path: `make compare-graph-fusion NOISE_FLOOR=1 SPLIT=` before and after, on the CUDA
  host; CI covers the scoring change over the committed graph fixtures.
- Acceptance gates: `make ci` green; the report states the fragile count and floor per graph row
  before and after, and whether any recorded graph-row verdict changes.
- Documentation target: the retrieval-strategies section of
  [GraphRAG](current/graphrag-backend.md) and the floor table in [RAG core](current/rag-core.md).

### chunker-bake-off-under-the-size-cap (optional)

Re-run the seven-strategy chunker bake-off now that `size` is a hard cap on every strategy. The
recorded winner (`sentence`, +0.022 recall@10 over `recursive`) was scored on stores that still
contained oversized units, and the unit-packing strategies are exactly the ones the cap changes:
their chunk counts rise and their long table/heading spans are now split
([RAG core](current/rag-core.md#chunking-strategies)). The ranking may hold, invert, or collapse
into a tie, and the current recommendation cannot say which. Score the same accepted goldset at
the same k and record whether the `sentence` recommendation survives. A second reason to re-run:
those stores also predate exact-duplicate chunk collapse, which changes the chunk counts per
strategy and, on a furniture-heavy corpus, the ranking itself -- it moved the goods rows and drove
that corpus's floor to zero ([RAG core](current/rag-core.md#duplicate-chunk-collapse)).

- Agent status: RUN NEEDED
- Dependencies: use the paired verdict in
  [RAG core](current/rag-core.md#paired-lane-uncertainty-and-verdict), because the recorded
  winner's margin is smaller than one item on the sets involved. Reuse `make compare-retrieval`
  with `NOISE_FLOOR=1` so a changed row can also be read against the corpus's own floor
  ([RAG core](current/rag-core.md#measurement-floor---noise-floor)).
- User-visible outcome: the per-corpus chunker recommendation rests on stores that respect the
  `size` the operator asked for.
- Scope boundary: in scope -- the re-run, the updated table, and an explicit keep-or-change
  verdict on the `sentence` recommendation. Out of scope -- new strategies and tuning `size`.
- Data and artifact paths: the existing per-strategy stores under `$DATA_DIR/llb/rag/<strategy>/`.
- Execution path: `make compare-retrieval CHUNK_STRATEGIES=sentence,recursive,page,heading,late,
  markdown,semantic GOLDSET=<quickstart accepted goldset> NOISE_FLOOR=1` on the CUDA host; no new
  CI coverage.
- Acceptance gates: `make ci` green; the report covers all seven strategies at the recorded k and
  states whether the recorded winner still wins by more than the measurement floor.
- Documentation target: the chunking-strategies evidence in [RAG core](current/rag-core.md).

### multihop-both-hops-ceiling

Every fused row measured so far -- every weight, both depths, both identity policies -- retrieves
BOTH hops for at most 3 of 35 two-hop questions (`all-spans@10` <= 0.086), while single-hop recall
moves freely between 0.686 and 0.800
([GraphRAG](current/graphrag-backend.md#span-identity-evidence)). That ceiling is invariant to
every ranking knob the lane exposes, which means it is probably not a ranking problem: either the
second hop's chunk is not retrievable for the question's own wording (a query problem, addressable
by decomposition), or it is not reachable at k=10 at all (a budget problem). Diagnose which:
measure `all-spans@k` as a function of k (say 10 / 25 / 50) on the same items, and measure the
per-hop retrievability of each labeled span when queried on its own. Record which of the two
explanations the corpus supports, because they lead to opposite fixes.

A third lead is already measured and worth folding into the k sweep: shrinking the CHUNK moves the
ceiling where no ranking knob could, the vector baseline's multi-hop `all-spans@10` running
0.057 -> 0.086 -> 0.114 as the chunking goes from `recursive@800/120` to `sentence@200` to
`recursive@200/30`
([GraphRAG](current/graphrag-backend.md#does-the-pin-survive-a-smaller-chunk-size)) -- which points
at the budget explanation, since k=10 buys more distinct spans when a span is smaller. Overall
recall falls at the same time, so treat it as a diagnostic, not a recommendation.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `all_spans_at_k` / `span_coverage_at_k` in `src/llb/rag/retrieval.py`,
  the sweep lane, and the existing query-decomposition step in
  [RAG core](current/rag-core.md#query-side-processing-uk-query-processing).
- User-visible outcome: the operator learns whether multi-hop evidence coverage is limited by the
  retrieval budget or by the query, instead of tuning ranking knobs that provably cannot move it.
- Scope boundary: in scope -- the k sweep, the per-hop probe, and a written diagnosis. Out of scope
  -- building a new decomposition strategy before the diagnosis names it, and any ranking-policy
  change.
- Data and artifact paths: `$DATA_DIR/graph-vector-fusion-multihop/<run>/`.
- Execution path: `make compare-graph-fusion RAG_K=<k>` per budget plus a per-hop retrieval probe
  on the CUDA host; CI covers the per-hop probe over fake lane stores.
- Acceptance gates: `make ci` green; the report carries `all-spans@k` per budget and the per-hop
  hit rate, and states which explanation the measurement supports.
- Documentation target: the graph-vector fusion evidence section of
  [GraphRAG](current/graphrag-backend.md#graph-vector-fusion-evidence).

### answer-side-span-coverage-metric

The retrieval side distinguishes "carried one hop" from "carried both" (`span_coverage_at_k` /
`all_spans_at_k`, [RAG core](current/rag-core.md#retrieval-metrics)); the ANSWER side has no such
distinction. `objective_score` is reference-answer token F1, so a two-hop answer that states one
fact fluently and omits the other scores roughly half -- the same value a vague answer touching
both facts gets. Every multi-hop answer-quality verdict therefore rests on a metric that cannot
say whether the model used both hops, which is precisely the question the lane exists to ask
([GraphRAG](current/graphrag-backend.md#answer-quality-evidence)). Build the answer-side
counterpart: per gold span, decide whether the ANSWER carries that span's content (lemma and
numeral overlap against the span text, thresholded and Ukrainian-aware, reusing the correctness
tokenizer), then report `answer_span_coverage` and `answer_all_spans` beside the objective and let
the multi-hop verdict read them.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the scoring tokenizer in `src/llb/scoring/correctness.py`, the
  multi-span retrieval metrics in `src/llb/rag/retrieval.py`, and the per-slice comparison in
  `src/llb/eval/answer_quality/`.
- User-visible outcome: the operator learns whether a multi-hop answer actually contains BOTH
  facts, instead of inferring it from a token-overlap score that a half-answer can earn.
- Scope boundary: in scope -- the answer-side coverage metric, its per-case columns, and wiring it
  into the answer-quality slices and verdict as an additive signal. Out of scope -- replacing
  `objective_score` as the leaderboard ranking metric, judge re-calibration, and any change to the
  retrieval metrics.
- Data and artifact paths: additive per-case columns in the standard `$DATA_DIR/run-eval/` bundles;
  comparison under the existing `$DATA_DIR/graph-vector-fusion-multihop/<run>/answer-quality/`.
- Execution path: `make compare-answer-quality` on the drafted multi-hop bundle; CI covers the
  metric on committed two-span fixtures (both facts, one fact, neither, paraphrased).
- Acceptance gates: `make ci` green; on single-span items the answer-side coverage agrees with the
  existing exact/contains signals; the multi-hop re-run reports the new columns with paired
  intervals and states whether they change the recorded verdict.
- Documentation target: [RAG core](current/rag-core.md#scoring) and the answer-quality evidence
  subsection of [GraphRAG](current/graphrag-backend.md#answer-quality-evidence).

### fusion-answer-quality-second-model (optional)

Repeat the end-to-end answer-quality comparison on a second roster model. Whether extra retrieved
evidence converts into a better answer is a property of the MODEL, not only of the retrieval lane:
a measured coverage gain that one model ignores may be exactly what a stronger (or more
instruction-following) model needs, and a single-model result cannot separate "fusion does not
help answers" from "this model does not use the extra hop". The lane, its verdict vocabulary, and
the drafted-grounding rules are current behavior
([GraphRAG](current/graphrag-backend.md#answer-quality-evidence)).

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `compare-answer-quality` as-is with a different `MODEL`; the matched
  stores and drafted bundle already exist.
- User-visible outcome: the operator learns whether the retrieval-only finding is a property of
  the corpus and the fusion lane, or of the one model that was scored.
- Scope boundary: in scope -- one more model, the same lanes/splits/seed, and a two-model
  comparison of the per-slice deltas. Out of scope -- any ranking-policy change, model selection,
  and re-tuning the graph weight per model.
- Data and artifact paths: `$DATA_DIR/graph-vector-fusion-multihop/<run>/answer-quality/`.
- Execution path: `make compare-answer-quality MODEL=<second-roster-model> SPLIT=final,tuning,
  calibration ANSWER_QUALITY_LANES=vector,<best-exact-row>,<best-overlap-row> INCLUDE_DRAFTED=1`
  -- the same three lanes the first model was scored on, so the two models are compared row for
  row; no new CI coverage.
- Acceptance gates: `make ci` green; both models score the identical item set at the same seed;
  the report states whether the two models agree on the verdict per lane, including whether the
  factoid cost of the overlap row reproduces.
- Documentation target: the answer-quality evidence subsection of
  [GraphRAG](current/graphrag-backend.md#answer-quality-evidence).

### retrieved-document-long-context-lane

The measured long-context lane is oracle-grounded -- it reads the item's own gold `doc_id`s, so it
sizes a CEILING and cannot be adopted
([product decisions](current/scope-boundaries.md#context-ablation-lanes-stay-diagnostic)). Add the
shippable sibling: a `retrieved_document` context strategy that takes the top-ranked RETRIEVED
chunk's document (no gold label), lays that whole document into the prompt under the same budget
check and `context_overflow` skip rule, and reports beside the existing lanes. The measured
oracle-versus-rag gap (+0.142 / +0.080 objective on two roster models) then splits into the part
an operator can actually capture by widening the unit of retrieval from a chunk to its document,
and the part that was pure oracle advantage.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the context-source seam, `fits_context_chars`, and the comparison in
  [RAG core](current/rag-core.md#context-ablation-does-rag-pay-for-itself-rag-vs-long-context-ablation).
- User-visible outcome: the operator learns whether "retrieve the chunk, send the document" is a
  real configuration worth shipping, or whether the long-context gain was the gold label all along.
- Scope boundary: in scope -- the strategy, its document-selection rule (top-1 versus top-k
  distinct documents), the budget/skip path, and a four-lane comparison. Out of scope -- changing
  the chunker, the ranking policy, and context-window extension tricks.
- Data and artifact paths: `$DATA_DIR/context-ablation/<run>/`.
- Execution path: `make compare-context-strategies CONTEXT_LANES=closed_book,rag,retrieved_document,long_context`;
  CI covers document selection and the skip rule over fake lane stores.
- Acceptance gates: `make ci` green; the three existing lanes reproduce their current rows exactly;
  a heavy run on both scored roster models reports the four-lane table with paired intervals and an
  explicit adopt-or-reject verdict for the new lane.
- Documentation target: the context-ablation section of [RAG core](current/rag-core.md) and the
  diagnostic-lane boundary in [product decisions](current/scope-boundaries.md).

### closed-book-decoding-stability (optional)

A closed-book score is a noisier measurement than a grounded one: two identical invocations of the
same lane on the same 82 items differed on 11 answers and moved the lane mean 0.160 -> 0.153, while
the `rag` and `long_context` lanes were byte-identical
([RAG core](current/rag-core.md#context-ablation-evidence)). An ungrounded prompt leaves a much
flatter next-token distribution, so kernel-level nondeterminism flips tokens. The drift stayed well
inside the uplift interval and changed no verdict, but a contamination rate quoted to one decimal
place is currently over-stated precision. Measure it: repeat the closed-book lane N times at a
fixed seed, report the between-repeat spread of the lane mean and of the contamination rate, and
either quote the ablation's closed-book numbers with that spread or make the lane reproducible
(pinned sampler / seeded backend options) if the backend allows it.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `compare-context-strategies` with a repeated `closed_book` lane and the
  existing paired-bootstrap reporting.
- User-visible outcome: the operator knows how much of a closed-book delta is measurement noise
  before reading it as parametric knowledge.
- Scope boundary: in scope -- repeat runs, the spread statistic, and whichever of the two remedies
  the measurement supports. Out of scope -- changing the objective metric and swapping backends.
- Data and artifact paths: `$DATA_DIR/context-ablation/<run>/`.
- Execution path: N repeats of the closed-book lane on the committed UA fixture on the CUDA host;
  CI covers the spread statistic over committed fixture rows.
- Acceptance gates: `make ci` green; the report states the between-repeat spread of the lane mean
  and the contamination rate, and the ablation docs quote closed-book numbers accordingly.
- Documentation target: the context-ablation evidence subsection of
  [RAG core](current/rag-core.md).

### context-ablation-question-type-slices (optional)

The context ablation slices by question type, but the committed UA fixture ships no
`needle_items.jsonl` sidecar, so every heavy run so far reported ONE pooled number per lane
([RAG core](current/rag-core.md#context-ablation-evidence)). Pooling hides the question the lane is
most useful for: retrieval almost certainly pays for itself unevenly -- a factoid whose answer is
one span versus a comparative or numeric question whose evidence is scattered. Run the ablation on
a gold set that HAS the sidecar (the quickstart-PDF accepted goldset, or a drafted multi-hop
bundle) so the uplift and the long-context delta are reported per slice, and record which slices
retrieval fails to pay for.

- Agent status: RUN NEEDED
- Dependencies: none. The slicing is already wired; this needs a labeled item set and the run.
  Question-type labels come from the needle sidecar
  ([data prep](current/data-prep.md)).
- User-visible outcome: the operator learns WHICH questions retrieval pays for on their corpus,
  instead of one pooled average over a mixed set.
- Scope boundary: in scope -- the run, the per-slice reading, and a verdict per slice. Out of
  scope -- new metrics, new lanes, and any ranking-policy change.
- Data and artifact paths: `$DATA_DIR/context-ablation/<run>/`.
- Execution path: `make compare-context-strategies GOLDSET=<sidecar-bearing goldset> CORPUS=<dir>`
  on the CUDA host; no new CI coverage.
- Acceptance gates: `make ci` green; the report carries a non-empty slice table with paired
  intervals per slice and states which slices the uplift interval fails to clear zero on.
- Documentation target: the context-ablation evidence subsection of
  [RAG core](current/rag-core.md).

### table-aware-chunking

Add a `table` strategy to `src/llb/rag/chunking/`: chunk boundaries never split a markdown table
row, a table that fits `size` stays one chunk carrying its nearest heading breadcrumb, and an
oversized table splits between row blocks with the header row's offsets recorded as additive
`metadata.table_header_span` -- chunk text stays a verbatim corpus slice with exact offsets.
Non-table text routes through the `recursive` splitter. Extend `compare-retrieval` with a
per-question-type breakdown (joined from `item_provenance.jsonl` when the sidecar exists) so the
numeric and comparative slices -- where tables carry the answers in converted Ukrainian PDF
corpora -- are scored beside the aggregate.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the chunking dispatch seam (`chunk_spans`), the markdown table output
  of the PDF conversion lane ([data prep](current/data-prep.md)), and the question-type taxonomy
  in the draft sidecars.
- User-visible outcome: numeric and comparative questions whose evidence lives in tables stop
  losing recall to mid-table chunk cuts, and the per-type breakdown shows exactly which question
  slice a chunking change helps or hurts.
- Scope boundary: in scope -- the strategy, tuner registration behind `--extended-chunkers`, and
  the per-type `compare-retrieval` breakdown. Out of scope -- cell-level table QA, corpus text
  rewriting, and HTML tables.
- Data and artifact paths: per-strategy stores under the existing comparison layout
  `$DATA_DIR/llb/rag/<strategy>/`; no new roots.
- Execution path: `make build-index CHUNK_STRATEGY=table`; `make compare-retrieval
  CHUNK_STRATEGIES=table,recursive,sentence GOLDSET=<gs>`; CI covers offset round-trips and
  row-boundary alignment on a committed markdown-table fixture.
- Acceptance gates: `make ci` green; every chunk stays offset-exact under `validate-goldset`;
  a heavy comparison over the quickstart accepted goldset reports aggregate plus numeric-slice
  recall@10 / MRR against `recursive` and `sentence`.
- Documentation target: [RAG core](current/rag-core.md) chunking strategies and the
  [data prep](current/data-prep.md) chunking list.

### ua-embedder-domain-finetune

Fine-tune the pinned multilingual E5 embedder on the operator's corpus: export contrastive
(question, gold-chunk) pairs from tuning-split gold items only (positives are chunks overlapping
the item's gold spans; hard negatives come from the BM25 lexical index), train with a
sentence-transformers contrastive objective behind lazy imports, and emit a tuned-embedder
directory whose manifest records the base model, dataset digest, item ids, and split counts. A
split guard refuses pairs from calibration or final ids (the `assert_tuning_only` discipline from
the LoRA hparam search). `compare-embeddings` accepts the tuned directory as a candidate so
uplift is measured by the standard source-span metric on the held-out final split, and the
store/query embedder fingerprint guard keeps a tuned-embedder store from being queried by any
other encoder.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the embedder conventions and bake-off in
  [RAG core](current/rag-core.md#embedder-conventions-and-bake-off), the lexical index for hard
  negatives, and the split-guard pattern in `src/llb/finetune/hparam_search/`.
- User-visible outcome: a corpus-adapted Ukrainian retriever the operator can adopt with measured
  final-split evidence, closing the recall gap on domain terms the general E5 encoder misses.
- Scope boundary: in scope -- pair export, the trainer, the manifest, bake-off integration, and
  the split guard. Out of scope -- cross-encoder (reranker) fine-tuning, generation-model
  fine-tuning (owned by the existing finetune lane), and hosted training.
- Data and artifact paths: pair datasets and tuned models under
  `$DATA_DIR/finetune-embedder/<model-slug>/<timestamp>/`; evaluation through the existing
  `$DATA_DIR/compare-embeddings/` layout.
- Execution path: `make finetune-embedder GOLDSET=<gs> CORPUS=<dir>` then
  `make compare-embeddings` with the tuned directory added as a candidate; CI uses a fake trainer
  plus the hashed-BoW embedder pattern from the curation tests, no GPU.
- Acceptance gates: `make ci` green; the guard refuses a pair set naming calibration/final ids;
  a heavy CUDA run trains on the quickstart tuning split and reports tuned-vs-base recall@10 /
  MRR on the held-out final split, where the adopt-or-keep-base verdict is the bake-off's own
  paired one -- the tuned row must clear zero against the base encoder, not merely outrank it
  ([RAG core](current/rag-core.md#paired-uncertainty-and-the-adopt-or-retain-verdict)).
- Documentation target: [RAG core](current/rag-core.md) embedder section and
  [extended workflows](current/extended-workflows.md) for the trainer lane.

### ua-model-roster-long-run (optional)

Confirm the refreshed-roster ranking at research scale: predeclare a minimum detectable objective
gain and ranking-stability criterion, derive the tuning-screen size from paired power, and run
multi-objective trials until the stability rule or a declared resource budget stops the search.
Score the full held-out final split and add the public Ukrainian screen tracks before making a
default-model adoption decision. Report bootstrap uncertainty and quality/latency Pareto tradeoffs
so a small-sample rank reversal cannot silently change the recommended model.

- Agent status: RUN NEEDED
- Dependencies: use the roster/runtime behavior in
  [platform matrix](current/platform-vector-matrix.md#ukrainian-model-roster-refresh) and the
  bounded baseline in [evaluation rigor](current/rigor-board-judge.md#joint-model--config-search).
- User-visible outcome: a stable refreshed-roster recommendation with uncertainty, public-task
  coverage, and an explicit adopt-or-retain verdict.
- Scope boundary: in scope -- larger private joint search, public-screen lanes, uncertainty, and
  the adoption verdict. Out of scope -- model fine-tuning and hosted/API-only candidates.
- Data and artifact paths: `$DATA_DIR/joint-search/<run>/`, `$DATA_DIR/screen/`, and the matching
  current-doc evidence section.
- Execution path: run `make joint-search` on a CUDA host with the refreshed candidates and full
  final split, then run the public screen for both finalists.
- Acceptance gates: `make ci` green; the search artifact records the effect, power, stability, and
  stopping assumptions plus the derived screen size and consumed trial budget; no final-split
  leakage into tuning; confidence-aware ranking; explicit quality-versus-latency recommendation.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) host evidence.

### normalize-casefold-dense-lane-cost (optional)

Normalization casefolds the whole query, but the dense encoder is case-sensitive: on the 82-item
final split the `normalize`-only lane retrieves WORSE than no mitigation at all under keyboard
noise (0.9268 -> 0.9024 recall@10), even though casefolding is supposed to be the safe half of
the lane ([evaluation rigor](current/rigor-board-judge.md#ukrainian-query-robustness-benchmark)).
The split noise classes sharpen the diagnosis: on the `apostrophe_variant` class the `normalize`
lane loses an item (0.9756 -> 0.9634 recall@10) even though the affected-items table shows all 6
perturbed questions retrieved perfectly in every lane -- the loss is on questions the noise class
never touched, so it is the mitigation step acting on an otherwise clean query, not a failed
repair. Casefolding is a lexical-side convention that the dense side never asked for. Measure
whether the processed query should stay cased on the dense lane while the lexical lane keeps the
folded text -- the `retrieve_queries` seam already carries separate dense and lexical text.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the split dense/lexical query seam in
  `src/llb/rag/query_prep/retrieval.py` and `RagStore.retrieve_queries`.
- User-visible outcome: the operator stops paying dense recall for a normalization step whose
  only job was to make matching safer.
- Scope boundary: in scope -- routing case-preserved text to the dense lane, the A/B, and the
  adopt-or-reject verdict. Out of scope -- changing the embedder, the lexical normalization, and
  the transliteration table.
- Data and artifact paths: `$DATA_DIR/query-robustness/<run>/`.
- Execution path: `make bench-query-robustness` on the CUDA host with and without the change; CI
  covers the dense/lexical text split over a fake store.
- Acceptance gates: `make ci` green; the report shows the `normalize` lane no longer retrieving
  below the `off` lane on any noise class, or records that casefolding is not the cause.
- Documentation target: [RAG core](current/rag-core.md) query-side processing and the robustness
  evidence in [evaluation rigor](current/rigor-board-judge.md).

### restoration-constraint-threshold-sweep (optional)

The restoration constraints ship with three unswept design constants: the surface-compatibility
budget (exact, `SURFACE_MAX_DISTANCE = 0`), the short-token cutoff that locks length and refuses
ties (`AMBIGUOUS_TOKEN_MAX_CHARS = 4`), and the ranking order that puts morphology ahead of local
context ([RAG core](current/rag-core.md#query-side-processing-uk-query-processing)). Each was chosen
to be conservative, and nothing measures what the conservatism costs: a budget of 1 admits a token
that was BOTH transliterated and mistyped, and a cutoff of 3 or 5 moves how many short words stay
untouched. Sweep them on a corpus where the typo lane is not saturated, report retrieval and the
edit-precision audit per setting, and pin each value with evidence or expose it.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `select_restoration` in `src/llb/rag/query_prep/restore.py` and the
  robustness lanes.
- User-visible outcome: the operator knows whether the safe defaults are costing recoverable
  recall, instead of trusting three hand-picked constants.
- Scope boundary: in scope -- the sweep, the per-setting edit audit, and a pin-or-expose verdict
  per constant. Out of scope -- new constraint signals and a learned ranker.
- Data and artifact paths: `$DATA_DIR/query-robustness/<run>/`.
- Execution path: `make bench-query-robustness` per setting on the CUDA host; CI covers each
  setting's selection decisions over committed candidate fixtures.
- Acceptance gates: `make ci` green; the report states recall and the share of corrections a human
  reading of the audit calls wrong, per setting, with an explicit verdict per constant.
- Documentation target: [RAG core](current/rag-core.md) query-side processing.

### conflict-null-model-research

**Research task** -- the answer is not known in advance, and a negative result is a valid outcome
that must be recorded rather than worked around.

Find a defensible independent null for corpus-conflict detection, so the semantic tier can report
a real false-positive rate instead of a rank cutoff. The current calibration measures the
similarity distribution of the corpus's own comparable cross-document pairs, which contains
whatever genuine duplicates the corpus has; with the pair space enumerated exactly the null and
the observed population are the same set, empirical FDR is identically 1.000 at every threshold,
and a budget of `N` returns exactly `N` pairs by construction (measured; see
[data prep](current/data-prep.md#known-limitation-there-is-no-independent-null)). Every downstream
question an operator asks -- "is this pair worth reading?", "did tightening the threshold remove
noise or evidence?", "is this corpus dirtier than that one?" -- currently has no statistical
answer.

Candidate approaches to evaluate, cheapest first; none is known to work:

- **Cross-corpus null.** Score chunks of the target corpus against chunks of an unrelated Ukrainian
  corpus. Pairs across corpus boundaries are unrelated by construction. Risk: a domain/register
  shift makes the null too easy, understating the threshold.
- **Within-document permutation.** Destroy the semantic relationship while preserving the corpus's
  marginal geometry -- shuffle tokens or sentences within a chunk before embedding. Risk: sentence
  encoders are partly bag-of-words, so a shuffled chunk may stay close to its original and the null
  lands too high.
- **Held-out-document null.** Bootstrap over document pairs, using the fact that most DOCUMENT
  pairs share no content, to estimate a per-document-pair rather than per-chunk-pair null. Risk:
  document pairs are few, so the tail is unresolvable on a small corpus -- the same saturation
  problem already measured for chunk-pair sampling.
- **Labelled calibration set.** Use the committed `samples/corpora/conflicts_uk_v1/` planted
  relations as ground truth to fit a threshold with a real measured precision/recall curve, then
  test whether that transfers to the quickstart corpora. Risk: seven planted pairs is a very small
  fit set, and the fixture uses a hashed-BoW fake embedder in CI.

- Agent status: RUN NEEDED
- Dependencies: the calibrated threshold and the enumerated distribution are current behavior
  ([data prep](current/data-prep.md#corpus-calibrated-cosine-threshold---max-candidate-pairs)).
  Reuse `estimate_null_distribution`, `VectorSet.cross_group_similarities`, and the planted-relation
  fixture. The comparable set excludes structurally repeated metadata blocks; use the measured
  post-filter population in [data prep](current/data-prep.md#what-the-semantic-tier-excludes-and-why).
- User-visible outcome: either a null the audit can quote a real false-positive rate against, or a
  recorded finding that cosine over sentence-encoder chunk vectors cannot support one -- which
  would justify moving threshold selection to the claim tier's measured precision instead.
- Scope boundary: in scope -- constructing and comparing candidate nulls, measuring each against
  the planted fixture and both quickstart corpora, and a written verdict per approach. Out of
  scope -- changing the relation vocabulary or the tier order, and shipping any new default before
  a null demonstrably beats the rank cutoff.
- Data and artifact paths: comparison under `$DATA_DIR/corpus-conflicts/null-research/<run>/`;
  no new committed fixtures unless an approach earns one.
- Execution path: a research harness invoked per null model over both quickstart stores plus the
  fixture; CI covers each null constructor deterministically over committed vectors, with the
  heavy corpus comparison run on the CUDA host.
- Acceptance gates: each candidate null is measured on the planted fixture, where the true relation
  labels are known, and reports precision/recall at its resolved threshold; an approach is adopted
  only if it beats the current rank cutoff on the fixture AND its resolved threshold recovers the
  claim-bearing HR swept baseline without flooding goods. If none does, the negative result is
  [product decisions](current/scope-boundaries.md) and the rank-cutoff framing stays.
- Documentation target: the corpus-hygiene known-limitation section of
  [data prep](current/data-prep.md), and [product decisions](current/scope-boundaries.md) for the
  adopt-or-reject verdict.

### typed-rag-answer-envelope

The RAG answer path emits FREE TEXT and every answer-side signal is recovered from that text after
the fact by a heuristic: `classify_response` maps a completion to a status with regex markers,
`is_abstention` reads first-person refusal stems, `parse_citations` scrapes `[i]` markers out of
prose, and groundedness re-segments the answer into "sentence-ish claims" by punctuation
([RAG core](current/rag-core.md#groundedness-and-citation-metrics-groundedness-citation-metrics)).
The repo already validates typed model output with Pydantic -- but only inside the structured-output
BENCHMARK lane, where `build_model` compiles a per-case schema and `is_conformant` reports whether
the completion satisfies it (`src/llb/scoring/structured_schema.py`); nothing on the answer path an
operator would actually ship is typed. Two consequences: the measured citation gap is unreadable
(the durable 3B run scored citation validity 0.000 because the model mostly did not cite at all, a
FORMAT failure scored as a grounding failure), and there is nowhere for a semantic validator to
attach -- checking business rules requires a typed object to check, which is the door this task
builds. Ship an `AnswerEnvelope` at the generation boundary -- the Ukrainian `answer`, `claims[]`
(claim text, the chunk indices it cites, and an optional subject/relation/object triple typed
against the closed 13-type entity vocabulary), an explicit `abstained` flag, and the evidence spans
-- parsed and validated in ONE place, with a completion that does not satisfy it ending in a typed
status rather than being scored as a wrong answer.

- Agent status: RUN NEEDED
- Dependencies: none, and it must land before `ontology-validated-answer-gate` (which validates the
  object this task defines). Reuse `build_model` / `parse_output` / `is_conformant` in
  `src/llb/scoring/structured_schema.py` wholesale (they take a field schema, not a benchmark case),
  the status taxonomy and `format_context` numbering in `src/llb/eval/common.py`, the claim
  segmentation and citation parsing in `src/llb/scoring/groundedness.py`, and the
  `eval.rag.cited_answer` template as the envelope prompt's starting point.
- User-visible outcome: an operator can tell a model that does not KNOW the answer from a model that
  cannot EMIT the answer in the requested shape, and every answer-side metric reads a declared field
  instead of a regex over prose.
- Scope boundary: in scope -- the envelope model, the boundary parse/validate function, the two new
  statuses (`malformed` stays "not JSON at all"; a new `schema_invalid` covers JSON that fails the
  envelope, because the two call for different fixes and today collapse into one number), one
  bounded `repair_once` reprompt carrying the validation error (the same policy shape measured by
  the [agent loop-policy lane](current/extended-workflows.md#agent-loop-policy-recommendation) for
  tool calls), the per-case columns, and a
  roster conformance study. Out of scope -- any SEMANTIC check on the envelope's contents (that is
  `ontology-validated-answer-gate`), changing the headline objective, constrained/grammar decoding
  in the backends, and making the envelope the default before the study supports it.
- Data and artifact paths: additive per-case columns (`envelope_status`, `n_claims`, `repaired`) in
  the standard `$DATA_DIR/run-eval/` bundles; the conformance study under
  `$DATA_DIR/answer-envelope/<run>/`.
- Execution path: `make run-eval ANSWER_FORMAT=envelope MODEL=<model> BACKEND=<backend>` over at
  least two roster models on the CUDA host, since envelope conformance is a property of the MODEL;
  CI drives parse, validate, repair, and each terminal status over the fake completer -- no GPU.
- Acceptance gates: `make ci` green; with the envelope off every recorded bundle reproduces
  bit-identically (the seam adds nothing to the free-text path); the study reports per-model
  conformance, `schema_invalid` rate, and repair rate SEPARATELY from correctness, so a repair gain
  is attributable to formatting rather than to reasoning; each claim's cited indices are validated
  in prompt-layout order, so `reverse_rank` renumbering is respected exactly as citation validity
  already respects it; an envelope answer's objective score matches the free-text score of the same
  `answer` string.
- Documentation target: the scoring and groundedness sections of
  [RAG core](current/rag-core.md#scoring), plus a validation-architecture subsection that names the
  boundary.

### ontology-axiom-layer

The induced ontology is a type INVENTORY, not a set of constraints: `induce_ontology` emits entity
and relation types with counts, confidence, and examples under the `MAX_ENTITY_TYPES` /
`MAX_RELATION_TYPES` caps (`src/llb/prep/ontology/induce.py`, `models.py`), and nothing in that
artifact can be VIOLATED. The graph build accepts whatever the extractor emits -- `add_fact` creates
a lightweight `MISC` fact-only node for an unrecognized endpoint rather than refusing the fact, and
the approved schema states that rule outright ("no grounded fact is dropped",
[graph ontology schema](../design/graph-ontology-schema.md)). So a corpus ledger can assert that one
patent has two different durations, that one work has two exclusive owners, or that one entity is
both `PERSON` and `ORG`, and no stage notices -- after which the drafting pipeline turns those facts
into gold questions and the graph lane retrieves them as evidence. Build the ledger-side half of the
validation architecture: an AXIOM layer over the closed vocabulary and the induced relations --
functional and inverse-functional properties (at most one object per subject, and vice versa),
`domain`/`range` type constraints per relation, disjoint entity-type pairs, symmetry/asymmetry/
irreflexivity, and cardinality bounds -- plus a checker that reads an extraction ledger and reports
every violation with the axiom it breaks and BOTH offending facts' evidence spans.

Serialize the axioms as RDFS/OWL Turtle (`owl:FunctionalProperty`, `owl:InverseFunctionalProperty`,
`owl:disjointWith`, `rdfs:domain`, `rdfs:range`) so the constraint set is standard, diffable, and
reviewable by someone who does not read this codebase -- but keep the SHIPPED checker pure Python
over the existing typed models, so no runtime dependency is added, matching the optional-extras
discipline the rest of the repo keeps. A standard reasoner (`rdflib` + `owlrl` behind an
`[ontology]` extra, marked `heavy_env`) rides along in CI as a CROSS-CHECK only: it must agree with
the in-repo checker on the committed axiom fixture, and a disagreement is a bug in the in-repo
checker. That keeps OWL semantics as the reference without letting a reasoner into the answer path.

- Agent status: RUN NEEDED
- Dependencies: none in code. Reuse the closed vocabulary and `normalize_entity_type` in
  `src/llb/prep/ontology/entity_types.py`, the typed `SROFact` / `Entity` / `OntologyCandidate`
  models in `src/llb/prep/ontology/models.py`, the caps and confidence blend in
  `src/llb/prep/ontology/constants.py`, the fact ingestion seam in `src/llb/graph/build.py`, and the
  violation-report renderer pattern from `src/llb/conflicts/report.py`.
- User-visible outcome: a corpus ledger whose logical inconsistencies are visible and named before
  they become gold questions or retrieved evidence, and a constraint set an operator can read,
  version, and hand to a domain expert.
- Scope boundary: in scope -- the axiom schema and its Turtle serialization, the pure checker, the
  reasoner cross-check, the violation report, the base-rate measurement over the committed corpora,
  and an opt-in `--refuse-violations` build flag. Out of scope -- inferring axioms from corpus
  frequency and shipping them unreviewed (a frequency-induced axiom only restates what the extractor
  emitted; acceptance is `ontology-axiom-signoff`), full OWL DL reasoning, changing the 13-type
  vocabulary or the relation caps, deleting any fact by default, and touching retrieval.
- Data and artifact paths: the candidate axiom set committed at `samples/ontology/axioms_uk_v1.ttl`
  plus its typed JSON form beside it; violation reports under `$DATA_DIR/ontology-validation/<run>/`.
- Execution path: `make validate-ontology-axioms EXTRACTION=<bundle>/extraction.jsonl
  AXIOMS=samples/ontology/axioms_uk_v1.ttl` over the drafted bundles of both quickstart corpora on
  the CUDA host (extraction ledgers already exist; no new inference unless a bundle is missing); CI
  covers each axiom class over a committed fixture carrying one planted violation per class plus the
  reasoner cross-check.
- Acceptance gates: `make ci` green; the pure checker and the `owlrl` reasoner return the identical
  violation set on the committed fixture; the report states the base rate per axiom class on both
  quickstart corpora, and a corpus with ZERO violations is recorded as a measured finding (that
  axiom class buys nothing on that corpus) rather than as a silent pass; the graph build is
  byte-identical unless `--refuse-violations` is passed; every reported violation cites both facts'
  exact spans, so a reviewer can adjudicate without re-reading the corpus.
- Documentation target: the ontology-assisted drafting section of
  [robustness and ontology](current/robustness-ontology-backends.md#ontology-assisted-drafting) and
  a constraints section in [graph ontology schema](../design/graph-ontology-schema.md).

### ontology-validated-answer-gate

Compose the two halves into the shipped two-step gate -- Pydantic at the door, the ontology at the
ledger. Step one is `typed-rag-answer-envelope`: the completion either parses into a typed answer
object or ends in a typed status. Step two is new: the envelope's asserted triples are checked
against the accepted axiom set AND against the corpus ledger the retrieved context came from, so an
answer that violates a functional property, a `domain`/`range` constraint, or a disjointness pair --
or that contradicts a ledger fact whose evidence is IN the retrieved chunks -- ends as
`ontology_violation` or takes one bounded repair instead of being scored as a fluent answer. This is
the step no existing signal covers: groundedness asks whether the answer's tokens appear in a chunk,
which a semantically impossible answer assembled from real chunk tokens passes cleanly.

The measurement has to be read honestly, because the obvious failure mode of any validator is
refusing correct work: report the gate's CATCH rate (violations caught per 100 answers) and its
FALSE-REJECTION rate (answers the gate rejects that the reference scores correct) as separate
numbers, report abstention rate and answered-item count beside the objective, and read the objective
delta on the items the UNGATED lane also answered -- otherwise a gate that improves the mean by
declining the hard items looks like a win.

- Agent status: RUN NEEDED
- Dependencies: `typed-rag-answer-envelope` (the typed object to validate) and
  `ontology-axiom-layer` (the axioms to validate it against); enabling an axiom at answer time also
  needs `ontology-axiom-signoff`, so the unsigned-axiom path must be refused rather than defaulted.
  Reuse the paired verdict machinery in `src/llb/rag/embedding_bakeoff_uncertainty.py` and
  `separates()` in `src/llb/rag/fusion_evidence/stats.py`, the lane-comparison shape of
  `compare-answer-quality`, and the ledger lookup in `src/llb/graph/retrieval.py`.
- User-visible outcome: an operator learns whether semantic validation of RAG answers is worth its
  cost on their corpus -- how many logically impossible answers it stops, how many correct answers
  it wrongly refuses, and what the repair round trip costs in tokens and wall clock.
- Scope boundary: in scope -- the ledger-side check, the `ontology_violation` status, the bounded
  repair, the catch / false-rejection / cost columns, a per-axiom-class adopt-or-reject verdict, and
  a committed violation fixture. Out of scope -- rewriting the answer on the model's behalf beyond
  the one repair, judge-based validation, changing the headline objective, enabling any axiom class
  by default before its measured numbers support it, and inventing a ledger fact the corpus does not
  carry.
- Data and artifact paths: `$DATA_DIR/answer-validation/<run>/` for the lane comparison; the
  fixture at `samples/benchmarks/ontology_violations_uk.json` (the layout its sibling case files
  already use), carrying one planted violating answer per
  axiom class PLUS correct answers a naive checker would reject (a legitimately multi-valued
  relation, a paraphrased entity that normalizes to the same node, an entity typed `MISC` by
  fallback), so the false-rejection number is measured on adversarial cases rather than asserted.
- Execution path: `make compare-answer-validation VALIDATION_LANES=off,pydantic,pydantic+ontology
  MODEL=<model> GOLDSET=<accepted> AXIOMS=<signed-ttl>` over roster-family strata until the declared
  family-coverage and paired-precision targets are reached; CI drives all three lanes, both
  statuses, and the repair path over the fake completer and a fake ledger -- no GPU.
- Acceptance gates: `make ci` green; the `off` lane reproduces the recorded run bundles exactly; the
  fixture's planted violations are caught at 100% per axiom class and the adversarial correct
  answers produce a NAMED false-rejection rate, not a claim of zero; the heavy run reports the
  objective delta against `off` on the commonly-answered items with a paired interval and the
  standard adopt-or-retain verdict, plus abstention rate, answered count, repair rate, and added
  tokens/latency per answer; an axiom class ships enabled only when its catch rate clears its
  false-rejection rate under that verdict, and every class that does not is recorded as measured-and-
  not-adopted; an unsigned axiom file is refused with a named error rather than silently enabled.
- Documentation target: a two-step answer-validation section in
  [RAG core](current/rag-core.md#scoring) beside the groundedness metrics, and the adopt-or-reject
  record per axiom class in [product decisions](current/scope-boundaries.md).

### thinking-suppression-and-answer-language-guard

`qwen3:30b` answers a Ukrainian benchmark prompt with first-person English deliberation ("Okay, I
need to explain...") even though the launcher sends Ollama's native `think: false` on every call,
so the thinking suppression the manifest relies on is not sufficient for that tag. Two scoring
risks follow: reasoning text inflates the generated-token count that throughput and cost are
derived from, and an English answer to a Ukrainian prompt is scored as content rather than caught
as an off-language response. Add a per-response guard that detects a leaked-reasoning prefix and a
dominant-script/language mismatch against the prompt, record both as named per-case flags in the
run bundle beside the existing reliability fields, and decide per model whether suppression needs a
prompt-level instruction on top of the API flag. Evidence for the observation is in the full-roster
throughput baseline in [backend telemetry](current/backend-telemetry.md).

- Agent status: RUN NEEDED
- Dependencies: the throughput protocol in
  [backend telemetry](current/backend-telemetry.md#telemetry-fields) and the correctness/reliability
  fields in [RAG core](current/rag-core.md#scoring).
- User-visible outcome: a run bundle shows how many answers leaked reasoning or answered in the
  wrong language, per model, instead of silently scoring them as ordinary content.
- Scope boundary: in scope -- the detection flags, their manifest fields, and a per-model
  suppression verdict. Out of scope -- rewriting the judge or changing the objective's definition.
- Execution path: re-run the roster throughput protocol capturing generations, then a bounded
  `run-eval` cell per affected tag.
- Acceptance gates: `make ci` green with injected fake generations covering leaked-reasoning,
  off-language, and clean answers; the flags appear in the persisted manifest; every roster tag
  carries an explicit suppression verdict, including the tags where no leak was observed.
- Documentation target: the roster baseline in
  [backend telemetry](current/backend-telemetry.md) and the scoring fields in
  [RAG core](current/rag-core.md#scoring).

### gemma4-gguf-runner-gap (optional)

The host Ollama cannot serve a `gemma4` GGUF at all: 0.20 answers both the curated `gemma4:12b`
tag and the first-party QAT `q4_0` GGUF with `unknown model architecture: 'gemma4'`, so the
`gemma-4-12b-it-w4a16` entry has no Ollama path and only its vLLM checkpoint is measurable here
(see the full-roster throughput baseline in
[backend telemetry](current/backend-telemetry.md)).
Any per-model result that resolves through Ollama therefore silently skips one manifest entry.
Pin the minimum Ollama version that knows `gemma4` in the host setup path, make the resolver report
an architecture-unsupported source as a NAMED skip instead of a generic backend error, and re-measure
the entry on Ollama so the roster is served by one backend end to end.

- Agent status: RUN NEEDED
- Dependencies: the resolver source-selection behavior in
  [platform matrix](current/platform-vector-matrix.md#multi-quant-vllm-resolution).
- User-visible outcome: an unsupported-architecture source fails with a source-specific message
  naming the runtime and the required version, and the roster has no backend-shaped hole.
- Scope boundary: in scope -- the version floor, the named skip, and the re-measured entry. Out of
  scope -- vendoring or building a llama.cpp runner.
- Execution path: raise the host Ollama, re-run the roster throughput protocol, refresh the
  baseline table.
- Acceptance gates: `make ci` green; a source whose architecture the runtime rejects produces the
  named error in a test with an injected fake; the refreshed table carries a measured Ollama row for
  every manifest entry or records the entry as backend-unsupported with its reason.
- Documentation target: the roster baseline in
  [backend telemetry](current/backend-telemetry.md) and the host setup notes in
  [host validation](current/host-validation.md).

## Human-Assisted Tasks

Add new human-gated work here per [Adding Future Tasks](#adding-future-tasks) when acceptance
requires human judgment or authorization.

### embedding-clustered chunk merging (optional)

The measured near-duplicate residue is real but not text-reachable: on the goods corpus 20.7% of
the exact-collapsed chunks have a neighbour at cosine >= 0.99, and the `normalized` collapse tier
merges 26 of those 13105 pairs
([RAG core](current/rag-core.md#near-duplicate-residue-and-the-collapse-tiers)). Only an
embedding-side merge can reach the rest, which the collapse lane deliberately does not do because a
false merge silently deletes a distinct passage from the index. Decide it with a measured
false-merge rate instead of by assumption: cluster the survivors by cosine at several thresholds,
sample the merges for human reading (the residue report's pair sampler already renders them), and
score retrieval per threshold against the corpus's own measurement floor. Adopt only if a threshold
lowers the fragile count without a recall regression AND its sampled false-merge rate is acceptable
on a corpus whose facts differ by one number.

- Agent status: HUMAN-GATED
- Dependencies: none. Reuse `measure_duplicate_residue` in `src/llb/rag/duplicate_residue.py` for
  the clustering and the sampler, and `collapse_duplicate_chunks` for the merge itself. Human step
  that gates completion: a reviewer reads the sampled merges at the candidate threshold and calls
  the false-merge rate acceptable or not.
- User-visible outcome: either an embedding-side merge tier with a measured false-merge rate, or a
  recorded finding that the residue must be left in the index.
- Scope boundary: in scope -- the clustering, the sampled review, the per-threshold retrieval run,
  and the adopt-or-reject verdict. Out of scope -- learned merge policies and corpus rewriting.
- Data and artifact paths: `$DATA_DIR/retrieval-noise-floor/<run>/`.
- Execution path: `make measure-duplicate-residue` per threshold, then `make compare-retrieval
  CHUNK_STRATEGIES=sentence,recursive NOISE_FLOOR=1` per candidate on the CUDA host.
- Acceptance gates: `make ci` green; the report carries the per-threshold recall against the floor,
  the fragile count, and the human's false-merge reading.
- Documentation target:
  [RAG core](current/rag-core.md#near-duplicate-residue-and-the-collapse-tiers).

### goods-fusion-weight-accepted-ledger

Settle the goods-corpus fusion-weight verdict on an item set someone accepted. The recorded
verdict ("the BM25 side costs recall at w=0.5, pin `FUSION_WEIGHT=0.7`") was measured on a
verified 44-item quickstart-PDF accepted goldset that is no longer on disk, and the lexical-row
re-read could not reproduce it: on the SAME corpus at the SAME chunking, the 95-item drafted
goldset inverts it -- fusion ADDS recall at w=0.5 (+0.021, +0.053 with lemmas, against a
+/-0.000 floor) and w=0.7 is the worst of the three weights for the best row
([RAG core](current/rag-core.md#lexical-row-re-read-of-the-fusion-weight-verdict)). The pin is
already withdrawn; what remains is deciding whether the recorded verdict was an artifact of its
item set or of the drafting, which only an accepted ledger over that corpus can answer.

- Agent status: HUMAN-GATED
- Dependencies: none in code -- `make compare-retrieval HYBRID=1 NOISE_FLOOR=1` and the
  verification gate are both current behavior. Human step that gates completion: a reviewer
  accepts (or rejects) enough of the drafted goods questions through
  `make verify-review` / `make verify-accept` to produce an accepted goods ledger.
- User-visible outcome: an operator on a converted-PDF Ukrainian corpus gets a fusion-weight
  recommendation backed by human-reviewed questions, instead of two drafted-versus-vanished item
  sets that disagree.
- Scope boundary: in scope -- worksheet review, the accepted ledger, one re-run per recorded
  weight with the `lexical` row, and a keep-or-change verdict on the recorded table. Out of scope
  -- widening the weight grid, new fusion mechanics, and changing the shipped defaults before the
  accepted run supports it.
- Data and artifact paths: the drafted bundle under
  `$DATA_DIR/graph-vector-fusion-multihop/goods-draft/` plus a new
  `$DATA_DIR/lexical-row-reread/goods-accepted-w<weight>/` per weight.
- Execution path: `make verify-review` then `make verify-accept` over the goods draft, then
  `make compare-retrieval CONFIG=<cfg> GOLDSET=<accepted>/goldset.jsonl SPLIT= HYBRID=1
  NOISE_FLOOR=1` at `FUSION_WEIGHT=0.5,0.6,0.7`.
- Acceptance gates: every worksheet row has a decision; the three weights are scored on the
  identical accepted item set with the `lexical` row and the corpus's floor; the recorded table is
  restated as reproduced, corrected, or retired.
- Documentation target: the hybrid-retrieval evidence section of
  [RAG core](current/rag-core.md#hybrid-retrieval-dense--bm25--rrf).

### fusion-routing-calibration-power (optional)

Increase the sidecar-free routing calibration's statistical power before reconsidering its
production defaults. The first held-out measurement cannot separate its positive retrieval deltas
from zero; see the compact result and frozen-policy diagnostics in
[GraphRAG](current/graphrag-backend.md#sidecar-free-heuristic-calibration). Assemble a larger,
independent multi-span tuning/final ledger, declare its minimum detectable gain and split sizes
before retrieval, then repeat the frozen-policy workflow without widening the threshold grid.

- Agent status: BLOCKED BY HUMAN
- Dependencies: the [shared paired-power contract](current/rag-core.md#paired-power-contract-for-comparison-lanes)
  supplies the predeclared split sizes, and `multihop-ledger-human-acceptance` must provide a
  non-empty accepted multi-span ledger. Human
  step that gates completion: accept enough additional genuinely multi-span questions to meet the
  predeclared split sizes.
- User-visible outcome: operators can distinguish a useful sidecar-free route from a sparse-win
  artifact before changing the fallback defaults.
- Scope boundary: in scope -- a prospective power target, disjoint tuning/final splits, and one
  repeat of the existing deterministic calibration. Out of scope -- widening the threshold grid,
  a learned router, and selecting on final.
- Data and artifact paths: a new `$DATA_DIR/graph-vector-fusion-multihop/<run>/` calibration over
  the accepted ledger.
- Execution path: run `make calibrate-fusion-routing` with the predeclared splits, then run the
  masked `make compare-graph-fusion` reproduction for the frozen policy on each split.
- Acceptance gates: route precision/recall and paired retrieval intervals meet the predeclared
  power target; a threshold changes only if the tuning gain clears zero without single-span
  regression and the untouched final split confirms the same direction.
- Documentation target: the sidecar-free calibration subsections of
  [RAG core](current/rag-core.md) and [GraphRAG](current/graphrag-backend.md).

### calibrate-headline-format-weight

Calibrate the fact/format tradeoff against adversarial context-copy answers and human pairwise
utility labels, then retain or revise the declared weight without changing the decomposition
contract ([current scoring](current/rag-core.md#headline-decomposition-and-declared-ranking-policy)).
Sweep predeclared weights over matched terse, fluent-but-wrong, verbose-supported, and
context-copy cases; measure agreement with the accepted labels and stability across model
families; require a held-out confirmation before changing the default.

- Agent status: BLOCKED BY HUMAN
- Dependencies: a reviewer-accepted pairwise preference ledger covering the four answer shapes.
- User-visible outcome: the explicit format share is grounded in operator utility and resists a
  model that raises recall by repeating retrieved context.
- Scope boundary: in scope -- fixture and worksheet generation, weight sweep, family-stratified
  agreement, uncertainty, and an adopt-or-retain verdict. Out of scope -- prompt rewriting,
  judge-model substitution, and changing the precision/recall/found-rate columns.
- Data and artifact paths: accepted labels under `$DATA_DIR/review/headline-format/`; sweep
  artifacts under `$DATA_DIR/verbosity-sensitivity/<run>/weight-calibration/`.
- Execution path: agent prepares and validates the blinded worksheet; human reviewers accept the
  pairwise labels; agent runs the deterministic sweep and CUDA-host family confirmation.
- Acceptance gates: `make ci` green; all answer-shape strata are represented; held-out agreement
  and confidence intervals are reported per family; any default-weight change names every roster
  rank it changes.
- Documentation target: [RAG core](current/rag-core.md#headline-decomposition-and-declared-ranking-policy)
  and the ranking policy in [evaluation rigor](current/rigor-board-judge.md).

### embedder-decision-on-a-resolvable-item-set

The embedder choice is undecidable on the item sets the repo currently has, and the paired lane
says so precisely: on the accepted converted-PDF ledger 36 of 40 questions are TIED between the
leader and the incumbent, so the 95% paired interval spans `[-0.050, +0.150]` and only a
consistent ~4-question gap could ever clear zero; on the committed fixture the baseline already
retrieves 0.980, leaving 5 questions of headroom for any candidate to win
([RAG core](current/rag-core.md#the-recommendation-re-read-with-paired-uncertainty)). The sub-base
roster addition `intfloat/multilingual-e5-small` is ~3x faster on warm CUDA with flat quality on
n=82 and still RETAIN
([RAG core](current/rag-core.md#blackwell-sub-base-encoder-roster-e5-small)) -- include it when the
enriched ledger re-runs so a cheap CUDA swap can clear an adoption bar if the discordance is there.
Both existing sets are
at their ceiling, which is a property of the QUESTIONS, not of the encoders. Build an item set that
can decide it: predeclare a minimum detectable recall gain and the split size it needs, then
assemble a ledger enriched with questions the incumbent currently MISSES (mine the per-item vectors
in `report.json` for baseline zeros, plus domain-term and morphology-heavy questions the general E5
encoder is expected to fail), accept it through the verification gate, and re-run the bake-off on
it. Record whether any candidate then separates -- a recorded "still undecidable at n=N" is a valid
outcome and is what would justify closing the question. The size the ENRICHMENT has to buy is
already priced: the withdrawn `e5-large` adopt differs on 5 of 250 items, and at that rate the
reporting level needs 300 items, which no committed goldset reaches -- so plain "more questions" is
not the route, raising the discordance rate is (double the rate, halve the floor)
([the re-decision](current/rag-core.md#the-re-decision-what-a-withdrawn-reading-needs)).

- Agent status: BLOCKED BY HUMAN
- Dependencies: the [shared paired-power contract](current/rag-core.md#paired-power-contract-for-comparison-lanes)
  supplies the item count the predeclared gain needs; the paired bake-off lane, the verdict, and
  `report.json` are current behavior
  ([RAG core](current/rag-core.md#paired-uncertainty-and-the-adopt-or-retain-verdict)). Human step
  that gates completion: a reviewer accepts the enriched question set through
  `make verify-review` / `make verify-accept`, since an unaccepted ledger cannot settle a default.
- User-visible outcome: either a measured embedder swap an operator can adopt, or a recorded
  statement of how many questions a decision would need -- instead of a permanently open ranking.
- Scope boundary: in scope -- the power target, the enriched ledger, its acceptance, and one
  re-run per corpus with the verdict restated. Out of scope -- new candidates, fine-tuning (that
  is `ua-embedder-domain-finetune`), and widening the verdict bar beyond recall@k.
- Data and artifact paths: the existing `$DATA_DIR/compare-embeddings/<run>/` layout.
- Execution path: `make compare-embeddings GOLDSET=<accepted-enriched-goldset> NOISE_FLOOR=1` on
  the CUDA host; no new CI coverage.
- Acceptance gates: the predeclared minimum detectable gain and split size are written down BEFORE
  retrieval; every candidate row reports its paired interval on the accepted ledger; the verdict is
  recorded as adopt, retain, or explicitly undecidable at the reached sample size.
- Documentation target: the embedder bake-off evidence in [RAG core](current/rag-core.md) and the
  recommendation line in [platform matrix](current/platform-vector-matrix.md#embedding-bake-off).

### multihop-ledger-human-acceptance

Accept (or reject) the drafted multi-hop retrieval slice through the verification gate, then re-run
BOTH draft-grounded lanes on the accepted ledger -- the fusion sweep and the end-to-end
answer-quality comparison -- so the graph-weight verdict rests on human-reviewed questions instead
of drafted ones. The drafted set, its worksheet, the matched vector/graph stores, and the measured
draft-grounded sweep plus answer-quality comparison are current behavior in
[GraphRAG](current/graphrag-backend.md#graph-vector-fusion-evidence); every drafted multi-hop item
is span-exact and Ukrainian-gated by construction, but only a reviewer can say whether a
shared-bridge question genuinely needs both facts.

- Agent status: HUMAN-GATED
- Dependencies: the [widened handoff](current/graphrag-backend.md#widened-multi-hop-review-handoff)
  supplies the worksheet, while the
  [paired-power contract](current/rag-core.md#paired-power-contract-for-comparison-lanes) derives
  the accepted-ledger requirement from a predeclared minimum detectable retrieval gain, expected
  discordance, confidence, and power. Human step that gates completion: a reviewer decides
  `accept`/`reject` for every worksheet row -- specifically whether the question is answerable ONLY
  with both cited spans -- and signs off on the resulting accepted ledger.
- User-visible outcome: a graph-weight recommendation for multi-hop retrieval backed by a
  human-accepted ledger, or a recorded finding that shared-bridge drafting does not produce
  genuine multi-hop questions and the slice must come from another source.
- Scope boundary: in scope -- worksheet review, `verify-accept`, re-running the sweep over BOTH
  span-identity policies on the accepted ledger, and the adopt-or-reject verdict per knob --
  including whether `graph_fusion_span_identity` flips from `exact` to `overlap` as the shipped
  default, which is currently gated only by the drafted ledger
  ([GraphRAG](current/graphrag-backend.md#span-identity-evidence)). Out of scope -- graph schema
  changes and fusion mechanics (the candidate-depth and span-identity verdicts are current
  behavior in [GraphRAG](current/graphrag-backend.md#candidate-depth-evidence)).
- Data and artifact paths: the widened drafted bundle and worksheet named in
  [the handoff](current/graphrag-backend.md#widened-multi-hop-review-handoff), plus a new
  `$DATA_DIR/graph-vector-fusion-multihop/<run>/` sweep over `accepted/goldset.jsonl` and its
  `answer-quality/` comparison.
- Execution path: start at `make verify-review VERIFY_WS=<widened-multi-hop-worksheet>`, then
  `make verify-accept VERIFY_WS=<widened-multi-hop-worksheet>
  BUNDLE=<widened-multi-hop-bundle>`, then
  `make compare-graph-fusion GOLDSET=<accepted>/goldset.jsonl
  GRAPH_FUSION_CANDIDATES=k,50 GRAPH_FUSION_SPAN_IDENTITY=exact,overlap`,
  then `make compare-answer-quality GOLDSET=<accepted>/goldset.jsonl FUSION_COMPARISON=<that
  sweep>/comparison.json` -- WITHOUT `INCLUDE_DRAFTED`, since an accepted ledger no longer needs
  the drafted-grounding escape.
- Acceptance gates: every worksheet row has a decision; the accepted ledger satisfies its
  predeclared paired-power requirement; retention is reported by relation pair, document mode, and
  source document with uncertainty; the re-run sweep reports the same rows with paired intervals
  and the human records the adopt-or-reject verdict per graph strategy and per span-identity
  policy; the answer-quality comparison re-runs on the accepted ledger with `grounding: verified`.
  If power remains insufficient or rejection failures are statistically concentrated in a
  stratum, run the relation-stratified widening workflow from
  [data prep](current/data-prep.md#widening-a-multi-hop-review-slice) against the latest reviewed
  ledger.
- Documentation target: the graph-vector fusion evidence section of
  [GraphRAG](current/graphrag-backend.md#graph-vector-fusion-evidence).

### corpus-conflict-resolution-review

Review the unresolved semantic conflict candidates through the workbench, then feed the accepted
ledger back into the resolver and repeat the retrieval plus verified answer-quality comparison.
The resolver behavior and the reason semantic candidates have no automatic suppression authority
are current behavior in
[data prep](current/data-prep.md#corpus-conflict-resolution-corpus-conflict-resolution).

- Agent status: HUMAN-GATED
- Dependencies: the resolution lane is current behavior. Human step that gates completion: an
  authorized corpus reviewer chooses `keep_both`, `drop_a`, or `drop_b` for every escalated row
  and signs off on the resulting suppression directives before application.
- User-visible outcome: an accepted or rejected suppression policy backed by reviewed conflict
  labels and a repeatable effect report, instead of adopting semantic similarity candidates as
  deletions.
- Scope boundary: in scope -- workbench review, accepted-ledger application, the same before/after
  metrics, and an adopt-or-revert decision. Out of scope -- changing detector thresholds,
  rewriting source text, or adding the resolver to auto-rag before the reviewed run supports it.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/resolution_review.jsonl`,
  `plan.json`, `conflict_overlay.json`, and `effect.md`; no new artifact root.
- Execution path: `make review-workbench REVIEW_PATH=<resolution-review-jsonl>`, then
  `make resolve-corpus-conflicts FINDINGS=<findings-jsonl> REVIEWED=<resolution-review-jsonl>
  APPLY=1 STORE=<store-dir> GOLDSET=<goldset-jsonl>` and repeat the fixed verified objective run.
- Acceptance gates: every review row has a decision; the regenerated plan has no unresolved
  records; rollback still restores the exact baseline; the human accepts only if retrieval and
  verified objective metrics do not regress.
- Documentation target: the resolution evidence subsection of [data prep](current/data-prep.md)
  and the conflict adapter notes in [review workbench](current/review-workbench.md).

### frontier-judge-authorization

Authorize the frontier scorer lane against real providers. The report tooling is current behavior
([frontier judge agreement and cost report](current/rigor-board-judge.md#frontier-judge-agreement-and-cost-report));
what remains is entirely the human authorization and the judgment it produces.

- Agent status: HUMAN-GATED human_decision: panding
- Command: once a real Anthropic key is in `.env` (~$0.40, 86 items):

  ```bash
  make frontier-judge-agreement \
    FRONTIER_JUDGE_MODELS=anthropic/claude-sonnet-4-5 \
    FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=1.00
  ```

- Dependencies: the agreement lane is current behavior; it runs on the 86-row calibration
  worksheet `calibration/ua_squad_postedited_v1.csv` (every row carries both a human and a local
  judge rating). Human step that gates completion: the operator puts a real Anthropic / OpenAI /
  Google key in `.env` (all three are currently blank placeholders, so no live run is possible),
  approves the per-run spend cap, and signs off on the resulting report.
- User-visible outcome: a decision record stating whether each frontier judge is trusted for
  autonomous gates on Ukrainian data, plus default budget caps derived from measured
  cost-per-item rather than from a guess.
- Scope boundary: in scope -- running the existing lane on the committed UA fixture, reviewing
  the rho and cost tables, recording an accept/reject per provider, and landing the resolved caps
  in the sample configs. Out of scope -- sending any private corpus to a provider, changing the
  headline-ranking policy, and any further report tooling.
- Data and artifact paths: `$DATA_DIR/frontier-judge/<run>/`; fixture is
  `samples/goldsets/ua_squad_postedited_v1/`.
- Execution path: `make frontier-judge-agreement FRONTIER_JUDGE_MODELS=<id>[,<id>...]
  FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=<cap>`; needs live provider access and spend, so it
  stays outside CI entirely.
- Acceptance gates: the report carries a non-`n/a` rho per provider against both references and a
  priced cost-per-item with cap math; the human replaces `human_decision: pending` with an accept
  or reject per provider; the accepted caps land in the sample configs with the decision recorded.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) judge section and
  [product decisions](current/scope-boundaries.md) for the trust decision per provider.

### frontier-judge-retrieved-context-agreement

Optional. Re-measure frontier-vs-human judge agreement with *retrieved* contexts instead of the
gold-span windows the authorization lane uses. The current lane deliberately holds retrieval
constant by grounding each item on a window of its gold source document, which isolates judge
behavior but also hands the judge cleaner evidence than a scored run ever gives it. A judge that
ranks well on oracle context may rank differently when the context contains distractors or misses
the answer entirely -- exactly the cases where an autonomous gate matters most. Add a context
source switch to `load_agreement_items` that pulls each item's top-k retrieved chunks from an
existing store, then report both grounding modes side by side so the gap is visible.

- Agent status: HUMAN-GATED
- Dependencies: blocked by `frontier-judge-authorization` (needs the same provider keys and
  spend). Reuse the agreement lane in `src/llb/scoring/frontier_agreement/` and the store-loading
  seam used by the context-position probe.
- User-visible outcome: evidence for whether frontier-judge trust measured on oracle context
  transfers to the noisy contexts a real scored run produces.
- Scope boundary: in scope -- the retrieved-context source, the two-mode comparison, and the
  delta. Out of scope -- changing the default grounding of the authorization lane before the
  comparison says it should.
- Data and artifact paths: no new roots; a second grounding mode inside the existing
  `$DATA_DIR/frontier-judge/<run>/` bundle.
- Execution path: `make frontier-judge-agreement` with a grounding-mode knob plus a built store;
  CI covers mode selection over a fake store and fake completers.
- Acceptance gates: `make ci` green; the gold-span mode reproduces the current numbers exactly; a
  live run reports rho under both modes with the delta called out.
- Documentation target: the frontier-judge agreement subsection of
  [evaluation rigor](current/rigor-board-judge.md).

### autonomous-vs-assisted-acceptance

Acceptance-test the full upgrade with a human operator: run `auto-rag` on a real Ukrainian corpus
twice -- once fully autonomous, once with human-assisted gates in the review workbench -- and have
the human judge both the reviewer experience and the recommendation quality.

- Agent status: HUMAN-GATED
- Dependencies: the autonomous lane is current behavior ([Auto-RAG](current/auto-rag.md));
  assisted review uses the [review workbench](current/review-workbench.md). Human step that gates
  completion: the operator
  performs both
  runs, reviews gated records in the workbench, measures their own throughput against the legacy
  per-flow sessions, and accepts or rejects the recommendation bundles.
- User-visible outcome: recorded evidence that the autonomous lane produces an acceptable
  recommendation without human action, and that the assisted lane's unified workbench is at least
  as fast and less error-prone than the legacy TUIs.
- Scope boundary: in scope -- the two runs, reviewer-throughput measurement (records per minute,
  correction rate), a comparison of the two recommendation bundles, and the acceptance decision.
  Out of scope -- fixing findings (each finding becomes a new forward task).
- Data and artifact paths: both run bundles under `$DATA_DIR/auto-rag/<run>/`; throughput notes
  and the acceptance record under `$DATA_DIR/auto-rag/<run>/acceptance/`.
- Execution path: `make auto-rag CORPUS=<dir> SCORER_POLICY=auto`, then
  `make auto-rag CORPUS=<dir> SCORER_POLICY=human` with workbench review at each gate; both on
  the CUDA host with a real corpus the operator owns.
- Acceptance gates: the human signs the acceptance record; both bundles are complete and
  reproducible from their manifests; throughput numbers and any usability findings are captured
  as new forward tasks before this item leaves the file.
- Documentation target: `current/auto-rag.md` acceptance evidence and
  [human evaluation guide](../guides/human-tooling/human-in-the-loop-evaluation.md).

### ontology-axiom-signoff

Accept or reject each candidate axiom from `ontology-axiom-layer`, one at a time, before any of them
can gate an answer. An axiom is BUSINESS LOGIC, not a measurement: whether "has owner" admits one
value or many, whether `PERSON` and `ORG` are genuinely disjoint in this domain, and whether a
relation's range is closed are claims about the world that no corpus statistic can settle. Inducing
them from corpus frequency would only restate what the extractor happened to emit -- the same
circularity the conflict tier already hit, where the null and the observed population turned out to
be the same set ([data prep](current/data-prep.md#known-limitation-there-is-no-independent-null)) --
and the cost of a wrong axiom is asymmetric and silent: at the ledger it deletes a true fact from
the report's attention, and at the answer gate it converts correct answers into `ontology_violation`.
The corpus cannot review itself here, which is why this is the one piece of the validation
architecture that sits in this section. The existing signed type-vocabulary review is the
precedent for the form ([graph ontology schema](../design/graph-ontology-schema.md)).

- Agent status: HUMAN-GATED
- Dependencies: `ontology-axiom-layer` supplies the candidate axioms, their Turtle rendering, and
  the per-axiom evidence (supporting facts, contradicting facts, and the measured base rate on both
  quickstart corpora). Reuse the review-workbench ledger pattern
  ([review workbench](current/review-workbench.md)) so the decisions are recorded the same way every
  other review ledger is. Human step that gates completion: a domain reviewer decides `accept` or
  `reject` for EVERY candidate axiom and signs the resulting axiom file.
- User-visible outcome: a signed, dated constraint set an operator can point the answer gate at,
  where every enabled axiom is a decision someone made rather than a statistic the corpus produced.
- Scope boundary: in scope -- the per-axiom review worksheet (each axiom rendered as Turtle plus a
  Ukrainian-language gloss and its supporting/contradicting facts with exact spans), the review
  pass, the signed axiom file, and a recorded reason per rejection. Out of scope -- authoring new
  axiom CLASSES (that is `ontology-axiom-layer`), changing the 13-type vocabulary or the relation
  caps, and enabling any axiom the reviewer did not accept.
- Data and artifact paths: the worksheet under `$DATA_DIR/ontology-validation/<run>/axiom_review.jsonl`;
  the signed set committed at `samples/ontology/axioms_uk_v1.ttl` with the sign-off line in its
  header, mirroring the dated sign-off convention of
  [graph ontology schema](../design/graph-ontology-schema.md).
- Execution path: `make validate-ontology-axioms` to regenerate the candidates and their evidence,
  then `make review-workbench REVIEW_PATH=<axiom-review-jsonl>`; no GPU is required for the review
  itself.
- Acceptance gates: every candidate axiom has a decision and every rejection has a stated reason;
  the committed axiom file contains only accepted axioms and carries the reviewer name and date;
  re-running the checker over both quickstart corpora with the signed set reports its violation base
  rate per accepted axiom; `ontology-validated-answer-gate` refuses to enable an axiom absent from
  the signed file.
- Documentation target: the ontology-assisted drafting section of
  [robustness and ontology](current/robustness-ontology-backends.md#ontology-assisted-drafting), the
  constraints section of [graph ontology schema](../design/graph-ontology-schema.md), and the
  trust boundary in [product decisions](current/scope-boundaries.md).

## Adding Future Tasks

Add a task only when there is concrete forward work with enough detail for an engineer or an
agent to execute without guessing. Use a stable descriptive id such as `platform-matrix-power`
or `prompt-system-tuning`; keep the id only while work remains under it. Place it under
**Agent Implementation Tasks** if it can land to `make ci` green with fixtures/fakes (heavy
deterministic runs on the CUDA host are fine), or under **Human-Assisted Tasks** if a human
review/judgment or authorization gates completion; either way give it a `Dependencies` line and
mark any cross-section block explicitly.

Each task entry must include:

- Dependencies: prerequisite tasks (by number/id), any cross-section block, and -- for
  human-assisted tasks -- the specific human step that gates completion.
- User-visible outcome: what new capability or decision the work should create.
- Scope boundary: what is in scope, what is explicitly out of scope, and which existing modules or
  commands should be reused.
- Data and artifact paths: expected corpus, gold set, config, `$DATA_DIR/<method>/<run>/` outputs,
  and any committed `samples/` outputs.
- Execution path: commands, manual run steps, required local services, and any heavy/dependent steps
  that must stay outside quick CI.
- Acceptance gates: tests, lint/type checks, retrieval thresholds, score comparison method, or manual
  evidence required before the item leaves this file.
- Documentation target: the narrow `docs/impl/current/*.md` topic and any guide that should receive
  the resulting behavior and run notes.

When a task surfaces new future work, add that as a new forward task. Put current behavior and
durable decisions in current docs, never in this plan.
