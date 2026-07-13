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

- Agent status: **CLEAR** -- agent-buildable to `make ci` green; a mechanical, test-guarded split
  per file. The convention, tooling, CLI exemplar, and the largest ~13 core-path/`rag`/`finetune`
  modules plus the largest shell script already landed (see
  [overview](current/overview.md#module-size--structure) for the current package layout). This
  is a long soft-limit tail, not a single deliverable: it advances module by module and the limit
  is soft, so a genuinely cohesive module may stay whole with a note.
- Dependencies: none. The ~250-line soft limit is documented in AGENTS.md; `scripts/code_quality.sh`
  now surfaces it both as a top-20 longest-code-files report (`.py`/`.sh`/`.mk`/`.awk`/`Makefile`)
  and the full over-limit `.py`/`.sh` list. Already split: the six `cli/*.py` modules; the core-path
  modules (`executor/runner`, `board/miss_analysis`, `finetune/hparam_search`,
  `prep/ontology/pipeline`, `prep/pdf_corpus`, `goldset/verify_session`, `board/recommend`);
  `rag/query_prep`, `rag/chunking`, `scoring/external_rag_session`, `finetune/distill`,
  `finetune/campaign`; and `scripts/quickstart.sh` (now an entrypoint + `scripts/quickstart/*`
  fragments).
- User-visible outcome: every tracked `.py`/`.sh` file sits at or under ~250 lines unless a single
  cohesive structure reads better whole, so review and comprehension improve.
- Scope boundary: in scope -- split the remaining over-limit files (~70 `src/` modules and ~36
  test modules) along clear functional seams (extract helper clusters into intent-named submodules;
  for shell, sourced fragments under a sibling dir). New splits follow the no-shim convention:
  repoint every caller and test at the specific submodule and keep `__init__.py` to just the
  package docstring --
  there is no public release, so a re-export layer is obsolete indirection (a package CLI keeps its
  idiomatic `__main__` entry). Prioritize by size; `scripts/code_quality.sh` lists them
  largest-first. Out of scope -- behavior changes; fragmenting a cohesive lookup table / dataclass
  family / exhaustive match just to hit the count (`core/contracts.py` stays whole as the justified
  exception).
- Acceptance gates: each landed split leaves `scripts/code_quality.sh` with that file no longer over
  the soft limit, `make ci` green, and no behavior test changes except import repointing to the new
  submodules; the task leaves this file when the over-limit list holds only explicitly-justified
  cohesive exceptions.
- Documentation target: [overview](current/overview.md#module-size--structure).

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
