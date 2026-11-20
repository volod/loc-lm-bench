# Agent Workflow Template Sync

## Work contract

Direct request: update this repository's spec and task development flow with the findings of the
upstream `agent-py` template at commit `4f431fa` -- its agent files, its task records, and its
plan tooling -- validate the result, and say whether this repository's
[future research](../future-research.md) register is an approach the template should adopt.

## Implementation and decisions

- **Agent adapters.** [CLAUDE.md](../../../CLAUDE.md) imports the root policy with `@AGENTS.md`
  instead of linking to it: a link is not loaded into a Claude Code session, the import is. The
  `.codex` file and `.cursor/rules/project-rules.mdc` are removed, as upstream did, because Codex
  and Cursor read the root `AGENTS.md` directly. [AGENTS.md](../../../AGENTS.md) gains Start Work
  and Finish Work checklists; its long procedure moved to the
  [planning workflow](../../guides/development/planning-workflow.md), which also absorbed the
  lane, ordering, and task-shape text the plan used to carry in its own introduction.
- **Task records.** This directory, its [template](template.md), and its index are new. A record
  follows the same citation rule as the current docs, so it never cites a host-local run path.
- **`Requires`.** Eight open-task prerequisites moved out of prose `Dependencies` into the new
  field; `make plan-status` skips the tasks still waiting on them.
- **Stricter gate.** `llb.quality.spec_plan_integrity` now also enforces all nine task fields, lane
  statuses, unique well-formed ids, required-before-optional order, a lane never returning to an
  earlier group, `Requires` edges, and forward-only plan wording. The parsers moved to
  `llb.quality.plan_model`, the edge checks to `llb.quality.plan_dependencies`, and the summary is
  `llb.quality.plan_summary`. The `documentation-integrity` spec section and registry row are
  amended in place; the capability stays `shipped`.
- **Two deliberate divergences from upstream.** A field value may wrap onto indented continuation
  lines (upstream reads only the first line, so a value starting on the next line reads as empty),
  and a status may carry a `-- reason`. `make plan-status` schedules in document order rather than
  upstream's required-first order, because this repository's binding rule is that an optional task
  of an earlier capability precedes a required task of a later one.
- **Plan findings the gate surfaced and this change fixed.** Two `RESEARCH` markers folded into a
  status became `Research: yes` fields; a stray annotation in a human-gated status line was removed;
  five tasks gained their missing data-path and execution fields; the forward plan's one history
  word was reworded; and `reranker-end-to-end-crosscheck-on-an-evidence-model`, a `RUN NEEDED` task
  with no human step, moved from the human lane to a new `retrieval-evidence` group in the agent
  lane, at the registry position that capability holds.
- **Not adopted.** Upstream's singular `docs/guide/` directory name: renaming `docs/guides/` here
  would churn every inbound link for no change in the flow.

## Acceptance evidence

Status: accepted. Run on the 12 GiB Blackwell host on 2026-11-19, in a dev-only environment
(`make venv EXTRAS=dev VENV_INSTALL_VLLM=0 SKIP_APT=1`):

- Before the plan was migrated, the stricter gate reported 15 findings against the existing plan,
  every one a real disagreement (listed under the plan findings above); after it, 0.
- `make ci-github` passed every `ci-checks` gate (format, Ruff, mypy over 1497 files, acceptance
  gates, complexity, shell lint, spec-plan, toolchain, roster, artifact contracts) and the GitHub
  suite: 4815 passed, 6 skipped, 161 deselected. The full local `make ci` suite was not run
  because the dev-only environment lacks the optional extras it exercises.
- `make lint-md` passed with 0 broken links and 0 run-path citations, and `git diff --check` was
  clean.
- The new fixtures in `tests/llb/quality/` refuse every shape the gate names and accept a
  cross-lane `Requires`, a wrapped field value, and a status with a reason.

## Handoff

`make plan-status` counts were 41 tasks before and after (no task added or removed). The agent
lane went from 19 to 20 and the human lane from 22 to 21, because the reranker cross-check changed
lane; `retrieval-evidence` now has a group in both lanes. Eight `Requires` edges are open, so 8
tasks are waiting. No registry row changed status; the `documentation-integrity` evaluation was
widened.

The prompt for porting the future-research register upstream is `future-research-prompt.md` at the
repository root, to be moved into the `agent-py` checkout.
