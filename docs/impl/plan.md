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
[data prep](current/data-prep.md).

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

## Human-Assisted Tasks

Each task's code and unit tests are agent-buildable; the marked **human step** is what gates
completion.

Recommended order for the human steps once the agent has pre-built each task's code: task 1's
coverage-vs-cap review and task 5's derived-case review both consume the same verification gate
over agent-prepared bundles. Task 2's egress consent and spend decision is independent and can
happen whenever the operator is ready to authorize it.

### 1. draft-yield-quality-max -- residual empirical acceptance

- Agent status: **HUMAN-GATED** -- the coverage-vs-cap accept-rate evidence needs a human
  `verify-sample` review pass; the optional multi-hop answer hardening is CLEAR (agent-buildable
  now). The agent can run both draft passes on the current CUDA host so only the review gates.
- Dependencies: none (uses the shipped drafting knobs). Human step: the acceptance evidence below
  needs a local drafter model and a human reviewer and cannot run in CI; the optional multi-hop
  hardening is agent-buildable and unit-tested.
- Context: coverage-target sampling (`--coverage-target`), 2-hop graph-path multi-hop drafting
  (`--multi-hop`), pinned-E5 prior-bundle near-duplicate suppression (`--dedup-against`), and the
  closed question-type + difficulty labels are implementable and unit-covered (module map, report
  fields, and command reference in
  [robust backends and ontology drafting](current/robustness-ontology-backends.md) and
  [data prep](current/data-prep.md)). What remains is the heavy manual acceptance evidence and one
  optional quality hardening, both needing a local drafter model and human review (out of CI):
- Acceptance evidence (human to-do): on the local quickstart PDF corpus, draft once with
  `DRAFT_COVERAGE_TARGET=<n>` and once with the 180-cap default over the same corpus/model, run a
  `make verify-sample VERIFY_N=40` review of each, and record in [data prep](current/data-prep.md)
  whether the coverage-target bundle keeps more citation-valid needles at an equal-or-better accept
  rate, with the retrieval-unique needle fraction per question type. This is the "keeps more needles
  at equal-or-better accept rate" gate that cannot run in CI.
- Optional quality hardening (agent-buildable): multi-hop items ground the two hop-evidence spans
  but leave the reference answer free-text. Require the multi-hop reference answer to be (or
  contain) the verbatim bridge/end-entity span so the answer itself is span-checkable, and extend
  the multi-hop unit tests to assert it -- a free-text answer can drift from the chain even when the
  evidence spans hold.
- Documentation target: [data prep](current/data-prep.md) and
  [robust backends and ontology drafting](current/robustness-ontology-backends.md).

### 2. frontier-ua-draft-lane

- Agent status: **HUMAN-GATED** -- human egress consent + API spend authorization gate the real
  2-document frontier probe; all code and fake-completer tests are CLEAR (agent-buildable now, no
  network).
- Dependencies: none (code reuses `src/llb/prep/frontier.py`). Human step: the real-frontier
  2-document probe requires **human egress consent and API spend** under the recorded egress policy;
  the code and all fake-completer tests are agent-buildable without any network call.
- User-visible outcome: for the best Ukrainian question quality and completeness, an operator
  can opt a draft run into a best-of-breed external API (litellm-routed) for extraction,
  drafting, or both, with an explicit consent gate, a hard budget cap, per-call cost telemetry
  in provenance, and a side-by-side local-vs-frontier yield and quality report over the same
  seeds. Frontier cross-check exists (`make cross-check-goldset CROSS_CHECK_MODEL=`); the draft
  lane does not.
- Scope boundary: in scope -- a frontier endpoint option for the ontology pipeline reusing the
  litellm conventions in `src/llb/prep/frontier.py` behind the same endpoint seam as Ollama and
  vLLM drafting (`src/llb/prep/ontology/endpoint.py`); `--max-usd` and `--max-calls` guards
  that abort cleanly and record the reason; an interactive egress consent prompt naming the
  corpus and destination (policy stays as recorded in
  [product decisions](current/scope-boundaries.md)); a `llb draft-compare` command that drafts
  the same bounded seed subset locally and via frontier and reports kept-yield, gate results,
  and verify-sample accept rate. Out of scope -- making frontier the default, egress for
  scoring or judging, retries of the egress policy discussion.
- Data and artifact paths: `provenance.json` gains `endpoint.cost_usd`, call counts, and
  latency; comparison reports under `$DATA_DIR/draft-compare/<timestamp>/`.
- Execution path:
  `make prepare-goldset-draft DRAFT_ENDPOINT=frontier DRAFT_FRONTIER_MODEL=<litellm-id>
  DRAFT_MAX_USD=<n>`; `llb draft-compare --corpus-root <dir> --seeds <n> --frontier-model <id>
  --local-model <model>`; unit tests use an injected fake litellm completer.
- Acceptance gates: no network call happens without the flag plus consent (unit-tested via the
  injected completer); the budget guard aborts mid-draft and the bundle remains inspectable
  with the abort recorded; a 2-document probe against a real frontier model passes bundle gates
  with parse rate at least matching the local drafter; the comparison report ranks both lanes
  on kept-yield and accept rate.
- Documentation target: [data prep](current/data-prep.md) frontier lane notes;
  [`docs/guides/data-prep/goldset-from-scratch.md`](../guides/data-prep/goldset-from-scratch.md).

### 5. security-corpus-probes

- Agent status: **HUMAN-GATED** -- derived cases must clear the human verify gate before any
  headline/composite use; the generator, unit tests, and the `bench-security` run itself are CLEAR
  / agent-executable on the current CUDA host.
- Dependencies: the shipped ontology artifacts (`ontology.json`, `extraction.jsonl`) from a
  local-drafter bundle. Human step: derived security cases must clear the human
  `verify-sample`/`verify-review`/`verify-accept` gate before any headline/composite use; the
  generator and unit tests are agent-buildable.
- User-visible outcome: the security tier gains corpus-specific cases derived from the target
  corpus itself: prohibited-topic denial-guard probes built from the corpus ontology's
  sensitive topics and entities, benign near-boundary controls that catch over-refusal on
  legitimate corpus questions, and matched-pair bias probes (entity or group swapped, behavior
  fixed) scored for decision consistency -- all flowing through the existing detectors,
  cross-language grouping, and the human verification gate before any headline use.
- Scope boundary: in scope -- `llb derive-security-cases --bundle <draft-bundle>` reading
  `ontology.json` and `extraction.jsonl`, generating cases locally through the same drafter
  endpoint seam; emitted cases reuse the committed case schema (`lang`, `attrs.vector`,
  `xlang_group`) so `bench-security`, `cross_language_consistency`, and refusal-appropriateness
  work unchanged (see [category suite](current/category-benchmark-suite.md) and the Ukrainian
  security adaptation in [evaluation rigor](current/rigor-board-judge.md)); a bias-pair
  consistency metric alongside ASR reusing the matched-group machinery. Out of scope -- new
  detector kinds, safety classifiers, ranking unverified case sets.
- Data and artifact paths: derived cases under `$DATA_DIR/security-derive/<timestamp>/cases.json`
  with per-case grounding spans back to the corpus; a small derived-and-verified sample
  committed under `samples/` for regression.
- Execution path: `llb derive-security-cases --bundle <draft> --out <cases.json>`;
  `make bench-security SECURITY_CASES=<cases.json> MODEL=<m> BACKEND=<b>`; verification through
  the existing `verify-sample`/`verify-review`/`verify-accept` path.
- Acceptance gates: every generated probe cites a corpus topic or entity with an exact span
  (unit-tested); benign controls feed refusal-appropriateness only, never ASR; a full
  `bench-security` run over derived quickstart-corpus cases reports per-family ASR plus
  bias-pair consistency with bootstrap CIs; unverified derived sets are rejected from
  composite/headline paths.
- Documentation target: [category suite](current/category-benchmark-suite.md) security section;
  [`docs/guides/learning-path/learning-path-security.md`](../guides/learning-path/learning-path-security.md).

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
