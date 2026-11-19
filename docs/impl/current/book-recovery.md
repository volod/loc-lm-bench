# Scanned Book Recovery

Recovering a scanned technical book -- page photographs with no text layer -- into a page-and-block
ledger that a reading vault and a corpus projection both render from. The contract, including its
boundary, is [Scanned book recovery](../../design/spec.md#scanned-book-recovery); the capability is
`planned` in the [registry](../../design/spec.md#capability-registry), and its remaining work --
tool survey, block ledger, preprocessing, metrics, pilot, rendering, projection -- is in
[the plan](../plan.md).

What exists today is the run skeleton: `llb book-ocr` inventories a scanned PDF, renders each page at
its native resolution, drives a registry of stages page by page with a per-page, per-stage ledger
and a content-addressed stage cache, and resumes after an interruption without repeating a completed
stage. No recognition stage is wired yet. Operator usage is the
[book-ocr guide](../../guides/data-prep/book-ocr.md).

| Page | What it answers |
| --- | --- |
| [Run ledger and resume](book-recovery/pipeline-and-resume.md) | The run directory, the stage seam, cache keys and atomic commits, failure isolation, source policy, the page inventory, and the full-book kill-and-resume evidence |
