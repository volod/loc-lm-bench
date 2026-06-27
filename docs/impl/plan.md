# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every milestone line in this file must describe work that remains. Current behavior,
operator workflows, run evidence, and decisions live in [`current.md`](current.md) and the topic
files under [`current/`](current/). The product spec lives in [`docs/design/spec.md`](../design/spec.md).

## No Open Work

There are no queued milestone tasks at this time.

## Adding Future Milestones

Add a milestone only when there is concrete forward work with enough detail for an engineer or an
agent to execute without guessing. Use a stable id such as `M8.1` or `M8.1r`; keep the id only while
work remains under it.

Each milestone entry must include:

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

When a milestone surfaces new future work, add that as a new forward task. Put run results,
decisions, and historical notes in current docs, never in this plan.
