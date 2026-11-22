# Scanned Book Recovery

Recovering a scanned technical book -- page photographs with no text layer -- into a page-and-block
ledger that a reading vault and a corpus projection both render from. The contract, including its
boundary, is [Scanned book recovery](../../design/spec.md#scanned-book-recovery); the capability is
`planned` in the [registry](../../design/spec.md#capability-registry), and its remaining work --
preprocessing, structure, the pilot, rendering, and projection -- is in [the plan](../plan.md).

What exists today is the run skeleton, page signals and variants, a class for every page and a
seeded pilot drawn from those classes, the block ledger, the quality harness, and the tool
survey. `llb book-ocr` inventories a scanned PDF, renders each page at its native resolution,
measures that render, and writes only the corrections the measurements call for
([page signals and variants](book-recovery/page-preprocessing.md)). `classes` then assigns every
page the classes it matches, and `pilot-select` draws a seeded pilot from them
([page classes](book-recovery/tool-selection.md#page-classes)). `llb book-ocr` drives a registry
of stages page by page with a per-page, per-stage ledger and a content-addressed stage cache,
and resumes a finished stage instead of repeating it. `layout`, `ocr`, `formula`, and `table` run a
configured engine on the original render or on a named variant. When `by_page_class` is set, the
engine follows the page's class, and the pages of a batch that share an engine run together.
`blocks` merges them into `pages/<pNNNN>/blocks.json`
([block ledger](book-recovery/block-ledger.md)). `llb book-ocr survey`
installs every candidate recovery tool into its own environment and smokes it on one synthetic and
one real page. `llb book-ocr score` compares a ledger, or the existing PDF lane, with a
reference transcription ([quality metrics](book-recovery/quality-metrics.md)). Structure and
rendering are not wired yet.
Operator usage is the
[book-ocr guide](../../guides/data-prep/book-ocr.md).

| Page | What it answers |
| --- | --- |
| [Run ledger and resume](book-recovery/pipeline-and-resume.md) | The run directory, the stage seam, cache keys and atomic commits, failure isolation, source policy, the page inventory, and the full-book kill-and-resume evidence |
| [Page signals and variants](book-recovery/page-preprocessing.md) | The per-page signal vector, the corrections those signals call for, the guards that refuse a damaged page, and the map that puts an engine box back on the original render |
| [Block ledger](book-recovery/block-ledger.md) | The page block contract, the engine adapter seam, reconciliation, and how a stage selects an engine |
| [Tool selection](book-recovery/tool-selection.md) | Which dewarp, layout, OCR, formula, table, converter, and vision-model candidates install and run on this host, and the page classes a pilot is drawn from: the per-stage shortlist with version, licence, VRAM and RAM, install route, and one-page smoke result; what was not installed and why |
| [Quality metrics](book-recovery/quality-metrics.md) | The reference format and the per-page, per-class score: character and word error, dangerous substitutions kept apart, formula fidelity, tables, figures, captions, reading order, and the printed page |
