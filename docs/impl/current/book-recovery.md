# Scanned Book Recovery

Recovering a scanned technical book -- page photographs with no text layer -- into a page-and-block
ledger that a reading vault and a corpus projection both render from. The contract, including its
boundary, is [Scanned book recovery](../../design/spec.md#scanned-book-recovery); the capability is
`planned` in the [registry](../../design/spec.md#capability-registry), and its remaining work
is in [the plan](../plan.md).

What exists today is the run skeleton, page signals and variants, a class for every page and a
seeded pilot drawn from those classes, the block ledger, the resolution of each block into one
text with a correction overlay, the review queue that marks what that resolution could not read,
the quality harness, and the tool survey. `llb book-ocr` inventories a scanned PDF and renders
each page at its native resolution, measures that render, and writes only the corrections the
measurements call for
([page signals and variants](book-recovery/page-preprocessing.md)). `classes` then assigns every
page the classes it matches, and `pilot-select` draws a seeded pilot from them
([page classes](book-recovery/tool-selection.md#page-classes)). `llb book-ocr` drives a registry
of stages page by page with a per-page, per-stage ledger and a content-addressed stage cache,
and resumes a finished stage instead of repeating it. `layout`, `ocr`, `formula`, and `table` run a
configured engine on the original render or on a named variant. When `by_page_class` is set, the
engine follows the page's class, and the pages of a batch that share an engine run together.
`blocks` merges them into `pages/<pNNNN>/blocks.json`
([block ledger](book-recovery/block-ledger.md)). An optional `correct` stage, off unless
the run names it, may propose a substitution the crop supports; the guard refuses a rewrite
([evidence-constrained correction](book-recovery/correction.md)). `resolve` chooses one text
per block and records corrections in an overlay; the raw ledger stays unchanged.
`llb book-ocr review-queue` turns each
unresolved span, empty content block, and low-confidence reading into a marker, a crop, a manifest
row, and a workbench entry
([uncertainty and review](book-recovery/uncertainty-and-review.md)). `llb book-ocr survey`
installs every candidate recovery tool into its own environment and smokes it on one synthetic and
one real page. `llb book-ocr score` compares a raw engine, the resolved pipeline, or the
existing PDF lane with a reference transcription
([quality metrics](book-recovery/quality-metrics.md)).
`llb book-ocr structure` reads the ledgers and writes the printed-page mapping, the hierarchy, and
the numbering inventory. `llb book-ocr render` writes the Obsidian vault from that file and links
each citation the structure names uniquely, and `llb book-ocr validate` checks it
([structure and the reading vault](book-recovery/structure-and-rendering.md)).
`llb book-ocr project` writes the retrieval corpus from the same ledger
([corpus projection](book-recovery/corpus-projection.md)). The generated books in CI exercise
vault publication, recovery from interruption, ingestion, a reviewer revision, and the link
from a retrieval hit to the cited scan region
([full-run rehearsal](book-recovery/full-run-evidence.md)). Operator usage is
the [book-ocr guide](../../guides/data-prep/book-ocr.md).

| Page | What it answers |
| --- | --- |
| [Run ledger and resume](book-recovery/pipeline-and-resume.md) | The run directory, the stage seam, cache keys and atomic commits, failure isolation, source policy, the page inventory, and the full-book kill-and-resume evidence |
| [Page signals and variants](book-recovery/page-preprocessing.md) | The per-page signal vector, the corrections those signals call for, the guards that refuse a damaged page, and the map that puts an engine box back on the original render |
| [Block ledger](book-recovery/block-ledger.md) | The page block contract, reading order, the engine adapter seam, reconciliation, and how a stage selects an engine |
| [Evidence-constrained correction](book-recovery/correction.md) | The optional `correct` stage: a local model proposes substitutions, a mechanical guard admits or refuses them, and a refusal becomes an uncertainty marker |
| [Uncertainty and review](book-recovery/uncertainty-and-review.md) | The resolved text of each block, the correction overlay, what a formula or table publishes while a value is open, the uncertainty markers, and the review queue |
| [Tool selection](book-recovery/tool-selection.md) | Which dewarp, layout, OCR, formula, table, converter, and vision-model candidates install and run on this host, and the page classes a pilot is drawn from: the per-stage shortlist with version, licence, VRAM and RAM, install route, and one-page smoke result; what was not installed and why. The pilot comparison scores accepted pages and writes the book-ocr config; the handbook winner waits on the accepted reference |
| [Quality metrics](book-recovery/quality-metrics.md) | The reference format and the per-page, per-class score of a raw engine, the resolved pipeline, and the existing PDF lane: character and word error, dangerous substitutions kept apart, abstentions, formula fidelity, tables, figures, captions, reading order, and the printed page |
| [Structure and the reading vault](book-recovery/structure-and-rendering.md) | The printed-page mapping, the hierarchy, the running headers, the figure, table, and equation inventory, the Obsidian vault rendered from them, the cross-reference links, and the mechanical check of that vault |
| [Formulas, tables, and figures](book-recovery/formulas-tables-figures.md) | Native formula and table structure on the ledger, LaTeX and equation numbers, Markdown / HTML / image tables, cell disputes, the crops kept beside an uncertain formula, and figure assets with their printed captions |
| [Corpus projection](book-recovery/corpus-projection.md) | The retrieval rendering of the same ledger: content only, page citations, the printed references, stable document ids, append-only revisions, and the operator-declared redistribution class |
| [Full-run rehearsal](book-recovery/full-run-evidence.md) | The generated book that walks every stage from the PDF through a validated vault, an ingested corpus, a killed resume, and one reviewer revision |
