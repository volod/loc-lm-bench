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

### module-size-soft-limit-refactor

- Agent status: **CLEAR** -- agent-buildable to `make ci` green through mechanical,
  behavior-preserving, test-guarded splits.
- Dependencies: none. Use the package convention and current backlog recorded in
  [Code Organization](current/overview.md#code-organization).
- User-visible outcome: every tracked `.py`/`.sh` file sits at or under ~250 lines unless a single
  cohesive structure reads better whole, so review and comprehension improve.
- Scope boundary: split the remaining over-limit production and test modules along clear
  functional seams; extract helper clusters into intent-named submodules and use sourced sibling
  fragments for shell. Repoint every caller and test to the concrete owner, keep package
  `__init__.py` files docstring-only, and retain `__main__` only for an actual module CLI. Do not
  add compatibility facades or re-export layers. Prioritize largest-first from
  `scripts/code_quality.sh`. Exclude behavior changes and avoid fragmenting a cohesive lookup
  table, dataclass family, or exhaustive match solely to meet the target; keep
  `core/contracts.py` as the justified cohesive exception.
- Acceptance gates: each remaining split must leave its replacement modules under the soft limit,
  keep `make ci` green, and avoid behavior-test changes except import repointing; remove this task
  when the live report contains only explicitly justified cohesive exceptions.
- Documentation target: [overview](current/overview.md#code-organization).

## Human-Assisted Tasks

Add new human-gated work here per [Adding Future Tasks](#adding-future-tasks) when acceptance
requires human judgment or authorization.

### knowledge-cutoff-ua-bilingual-calibration

- Agent status: **BLOCKED BY HUMAN** -- the paired runner and worksheet tooling are agent-buildable,
  but publication-quality Ukrainian translations require bilingual human review.
- Dependencies: use the baseline command, schema, and report contract in
  [knowledge cutoff](current/knowledge-cutoff.md). Human step: review every translated question and
  choice for factual equivalence, answer preservation, fluency, and absence of new temporal clues.
- User-visible outcome: a paired English/Ukrainian cutoff report distinguishes temporal knowledge
  decay from language-comprehension loss for Ukrainian-specialized local models.
- Scope boundary: translate the exact revision-pinned event questions and answer choices without
  adding facts, randomize both lanes with the same source-choice mapping, add paired language-delta
  statistics and a bootstrap interval, and gate the Ukrainian lane on a complete accepted
  worksheet. Do not create new events, translate source articles, or mix rejected/undecided rows
  into a cutoff claim.
- Data and artifact paths: keep translation drafts and review state under
  `$DATA_DIR/knowledge-cutoff-ua/<dataset-revision>/`; write paired model runs under
  `$DATA_DIR/knowledge-cutoff-bilingual/<run_timestamp>/`; if a reviewed translation fixture is
  redistributed, include its CC BY 4.0 attribution, exact upstream revision, and accepted worksheet
  under `samples/verification/knowledge_cutoff_ua/`.
- Execution path: add a Make target that drafts the pinned translation bundle locally, opens the
  shared verification session, freezes accepted rows, runs both language lanes through the same
  local backend, and emits one paired report. Keep drafting and human review separate from the
  model-scoring command so partial review is resumable and cannot be mistaken for accepted data.
- Acceptance gates: every translated row is decided and accepted or excluded; answer keys and
  source-choice identities match mechanically; a bilingual reviewer signs off the worksheet; unit
  tests cover alignment and gate failures; the paired report includes per-month language deltas and
  a seeded bootstrap interval; focused tests, type/lint checks, and `make lint-md` pass.
- Documentation target: extend
  [the knowledge-cutoff guide](../guides/benchmarking/knowledge-cutoff.md) and
  [current behavior](current/knowledge-cutoff.md) with the bilingual workflow and reviewed revision.

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
