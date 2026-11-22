# Task Records

A task record preserves a task's work contract and its acceptance evidence after the task leaves
the [forward plan](../plan.md), where its heading and id disappear. Delivered behavior and the
measured results that settle decisions live in the [current docs](../current.md); a record is how a
later reader finds out what was asked, what was accepted, and on what evidence.

- **Planned work.** Copy the [template](template.md) to `NNNN-<group>-<task-id>.md` in this
  directory before editing, and paste the original plan task into it verbatim. Task ids are stable
  and globally unique, so the filename stays a valid reference after the plan heading is gone.
  Explain any later scope change in the record rather than editing the pasted contract.
- **Direct requests.** Use a record when findings, tradeoffs, or verification evidence need a
  durable handoff, named `NNNN-adhoc-<slug>.md`.
- **Work spanning sessions.** Keep the record open and name the remaining gate and the next action.

A record cites results the same way the current docs do -- no host-local `$DATA_DIR` run path and
no bare run label; see [citing a measured result](../../guides/development/heavy-runs-and-evidence.md).
Link a record from the owning current-docs topic when it explains a durable decision. The full
completion transition is in the
[planning workflow](../../guides/development/planning-workflow.md#completion-transition).

**Naming.** `NNNN` is the next unused four-digit sequence in the table below, `<group>` is the
capability id the task serves (`adhoc` for a direct request), and `<task-id>` is the plan task id;
add the row to the table in the same change. The full rule is
[record file naming](../../guides/development/planning-workflow.md#record-file-naming), and
`make lint-doc-links` refuses a record that breaks it.

## Records

| Sequence | Task | Record | Work |
| --- | --- | --- | --- |
| 0001 | `agent-workflow-template-sync` | [0001-adhoc-agent-workflow-template-sync](0001-adhoc-agent-workflow-template-sync.md) | Adopting the upstream agent template's adapters, task records, `Requires`, and stricter plan gate |
| 0002 | `book-ocr-run-ledger-and-resume` | [0002-scanned-book-recovery-book-ocr-run-ledger-and-resume](0002-scanned-book-recovery-book-ocr-run-ledger-and-resume.md) | The resumable `book-ocr` run skeleton: page inventory, stage registry, per-page ledger, content-addressed stage cache |
| 0003 | `export-redistribution-gate` | [0003-corpus-provenance-export-redistribution-gate](0003-corpus-provenance-export-redistribution-gate.md) | The shared redistribution check every export and frontier egress path calls: refuse, never filter, material whose recorded `licence` forbids it |
| 0004 | `book-recovery-tool-survey` | [0004-scanned-book-recovery-book-recovery-tool-survey](0004-scanned-book-recovery-book-recovery-tool-survey.md) | The recovery-tool survey: candidates per stage installed into isolated environments and smoked on a synthetic and a real page; the per-stage shortlist |
| 0005 | `container-toolkit-and-vllm-candidates` | [0005-adhoc-container-toolkit-and-vllm-candidates](0005-adhoc-container-toolkit-and-vllm-candidates.md) | The NVIDIA Container Toolkit required on CUDA hosts; the vLLM-only recovery candidates served from Docker and added to the selection |
| 0006 | `book-block-ledger-and-engine-adapters` | [0006-scanned-book-recovery-book-block-ledger-and-engine-adapters](0006-scanned-book-recovery-book-block-ledger-and-engine-adapters.md) | The page block ledger and the shortlist engine adapters behind the `book-ocr` stage seam |
