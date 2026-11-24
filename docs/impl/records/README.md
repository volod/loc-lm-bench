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
| 0007 | `book-page-signals-and-variants` | [0007-scanned-book-recovery-book-page-signals-and-variants](0007-scanned-book-recovery-book-page-signals-and-variants.md) | Per-page signals, guarded image variants, and mapping engine boxes back to the original render |
| 0008 | `book-recovery-quality-metrics` | [0008-scanned-book-recovery-book-recovery-quality-metrics](0008-scanned-book-recovery-book-recovery-quality-metrics.md) | The recovery quality harness: reference format, per-class metrics, dangerous substitutions apart from character error |
| 0009 | `book-pilot-page-selection` | [0009-scanned-book-recovery-book-pilot-page-selection](0009-scanned-book-recovery-book-pilot-page-selection.md) | Per-page classes, seeded pilot selection, and engine routing on the class stage |
| 0010 | `book-reading-resolution-and-correction-overlay` | [0010-scanned-book-recovery-book-reading-resolution-and-correction-overlay](0010-scanned-book-recovery-book-reading-resolution-and-correction-overlay.md) | One resolved text per block, a correction overlay anchored to the block, and reviewer decisions in the stage cache key |
| 0011 | `book-uncertainty-markers-and-review-queue` | [0011-scanned-book-recovery-book-uncertainty-markers-and-review-queue](0011-scanned-book-recovery-book-uncertainty-markers-and-review-queue.md) | Uncertainty markers, review crops, the quality manifest, and the workbench queue |
| 0012 | `book-structure-inference` | [0012-scanned-book-recovery-book-structure-inference](0012-scanned-book-recovery-book-structure-inference.md) | Printed-page mapping, the part/chapter/section hierarchy, and the figure, table, and equation inventory |
| 0013 | `book-vault-rendering-and-navigation` | [0013-scanned-book-recovery-book-vault-rendering-and-navigation](0013-scanned-book-recovery-book-vault-rendering-and-navigation.md) | The Obsidian reading vault: chapter files, navigation, anchors, figure crops, and atomic publish |
| 0014 | `book-vault-validation-and-report` | [0014-scanned-book-recovery-book-vault-validation-and-report](0014-scanned-book-recovery-book-vault-validation-and-report.md) | Mechanical checks of a rendered vault, and the processing report of what still needs a person |
| 0015 | `book-cross-reference-linking` | [0015-scanned-book-recovery-book-cross-reference-linking](0015-scanned-book-recovery-book-cross-reference-linking.md) | Conservative cross-reference links, generated lists, and counted misses |
| 0016 | `scanned-book-quality-audit` | [0016-adhoc-scanned-book-quality-audit](0016-adhoc-scanned-book-quality-audit.md) | Restoration fidelity, full-book publication, and subgroup acceptance gates |
| 0017 | `book-corpus-projection` | [0017-scanned-book-recovery-book-corpus-projection](0017-scanned-book-recovery-book-corpus-projection.md) | The retrieval corpus rendered from the book ledger, with page citations and append-only revisions |
| 0018 | `book-resolved-pipeline-scoring` | [0018-scanned-book-recovery-book-resolved-pipeline-scoring](0018-scanned-book-recovery-book-resolved-pipeline-scoring.md) | The resolved pipeline scored beside the raw engine lanes and the existing PDF conversion |
| 0019 | `book-input-and-pilot-audit` | [0019-scanned-book-recovery-book-input-and-pilot-audit](0019-scanned-book-recovery-book-input-and-pilot-audit.md) | Verify the frozen source and pilot, original-frame crops, damage guards, and independent scoring failures |
| 0020 | `book-pilot-pipeline-comparison` | [0020-scanned-book-recovery-book-pilot-pipeline-comparison](0020-scanned-book-recovery-book-pilot-pipeline-comparison.md) | Per-class pipeline comparison against the existing lane, with bootstrap margins, VRAM refusal, and the selected configuration |
