# Task Records

A task record preserves a task's work contract and its acceptance evidence after the task leaves
the [forward plan](../plan.md), where its heading and id disappear. Delivered behavior and the
measured results that settle decisions live in the [current docs](../current.md); a record is how a
later reader finds out what was asked, what was accepted, and on what evidence.

- **Planned work.** Copy the [template](template.md) to `<task-id>.md` in this directory before
  editing, and paste the original plan task into it verbatim. Task ids are stable and globally
  unique, so the filename stays a valid reference after the plan heading is gone. Explain any later
  scope change in the record rather than editing the pasted contract.
- **Direct requests.** Use a record when findings, tradeoffs, or verification evidence need a
  durable handoff, under a descriptive filename.
- **Work spanning sessions.** Keep the record open and name the remaining gate and the next action.

A record cites results the same way the current docs do -- no host-local `$DATA_DIR` run path and
no bare run label; see [citing a measured result](../../guides/development/heavy-runs-and-evidence.md).
Link a record from the owning current-docs topic when it explains a durable decision. The full
completion transition is in the
[planning workflow](../../guides/development/planning-workflow.md#completion-transition).

## Records

| Record | Work |
| --- | --- |
| [Agent workflow template sync](agent-workflow-template-sync.md) | Adopting the upstream agent template's adapters, task records, `Requires`, and stricter plan gate |
| [book-ocr-run-ledger-and-resume](book-ocr-run-ledger-and-resume.md) | The resumable `book-ocr` run skeleton: page inventory, stage registry, per-page ledger, content-addressed stage cache |
| [export-redistribution-gate](export-redistribution-gate.md) | The shared redistribution check every export and frontier egress path calls: refuse, never filter, material whose recorded `licence` forbids it |
