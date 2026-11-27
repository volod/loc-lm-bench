# Uncertainty and review

`resolve` chooses the one text both renderings will read for each block, and records every
correction in an overlay over the raw ledger. The ledger is not rewritten. The review queue reads
that overlay and writes reviewer decisions back into the same file. The optional `correct`
stage writes the model edits this seam reads
([evidence-constrained correction](correction.md)). Operator usage is the
[book-ocr guide](../../../guides/data-prep/book-ocr.md). The ledger they resolve is the
[block ledger](block-ledger.md).

## Where it lives

| Module | Owns |
| --- | --- |
| `src/llb/prep/book/resolve/apply.py` | The page rule: which edits attach, which are orphaned, which reading is primary |
| `src/llb/prep/book/resolve/spans.py` | How one primary reading is tiled into adopted and unresolved spans |
| `src/llb/prep/book/resolve/primary.py` | The configured reading, and the visible fallback when a block lacks it |
| `src/llb/prep/book/resolve/anchor.py` | The block digest an edit must match |
| `src/llb/prep/book/resolve/io.py` | Overlay, resolved text, model corrections, and the run-level reviewer file |
| `src/llb/prep/book/stages/resolve.py` | The `resolve` stage |
| `src/llb/prep/book/correct/` | The optional `correct` stage: proposals, the guard, and the adoption report |
| `src/llb/core/contracts/data_prep/book_resolve.py` | The four resolution families below |
| `src/llb/prep/book/review/` | Markers, crops, the quality manifest, and the review queue |
| `src/llb/review/adapters/book_queue.py` | The workbench adapter that writes reviewer decisions |
| `src/llb/cli/book/review_queue.py` | `llb book-ocr review-queue` |
| `src/llb/core/contracts/data_prep/book_review.py` | The queue and the quality manifest |

Tests: `tests/llb/prep/book/test_resolve.py`, `test_resolve_stage.py`, `test_ledger_reuse.py`,
and `test_review_queue.py`. Model edits are `test_correct_guard.py` and `test_correct_stage.py`.
The adapter is also covered from `tests/llb/review/test_adapters.py`.

## The resolution rule

The run configuration names the primary reading:

```yaml
primary:
  stage: ocr
  engine: <engine>
  variant: original
```

`stage` is `layout`, `ocr`, `formula`, or `table`, and that stage is configured before `resolve`.
`model` is optional; when it is set, the reading's model must match too. `by_page_class` maps a
page class to another primary. When `classes` ran earlier and that map is non-empty, the class
file is an input, so a class change re-resolves the pages it moves, and the page's primary class
wins. A block that does not carry the configured reading uses the first reading in engine,
variant, then model order, and the resolved block sets `fallback`. Falling back does not vote.

One reading is the primary reading: nothing else exists to agree with it. With two or more
readings, each character every other reading shares with the primary is an agreement edit. Its
`raw_text` equals its `replacement`, its evidence is `every reading agrees`, and its id is
`agreement:<block-id>:<start>:<end>`. Two of three agreeing does not outvote the third. An
uncertainty span of the primary stays unresolved unless an overlay edit covers those offsets. A
letter disagreement that is not an uncertainty span stays the primary reading.

The adopted `text` joins the adopted spans only. An unresolved span keeps the primary characters,
with `adopted` false and the uncertainty classes that cover it, and those characters are omitted
from `text`. Span offsets are into the primary reading. A rendering that shows only `text` drops
the unresolved characters; the span is what a later marker reads.

Reviewer edits win over model edits, and both win over agreement. A higher-precedence edit
consumes its whole span. The overlapped edit stays in its source file and is not applied. Between
edits of one author, the earlier start wins, then the edit id. A zero-width edit is an insertion.

## The overlay

An edit carries its author (`agreement`, `model`, or `reviewer`), the raw slice, the replacement,
and the evidence. It is anchored by the block id plus the digest of that block's region and raw
readings, in stored order. Ids follow geometry, so the id alone is not enough: a rebuild can give
a new block the old id.

| Anchor check | Orphan reason |
| --- | --- |
| The block id is gone | `block_missing` |
| The id is still there and the digest differs | `block_changed` |
| The digest matches and the raw slice does not | `raw_mismatch` |

Every orphan has `requeued` true. It is not applied to any other block, including one that kept
the old digest under a new id. Agreement edits are regenerated each run, so they are not orphaned.

Model edits arrive as `pages/<pNNNN>/corrections.json` (`llb.book-ocr-model-corrections`) and are
read only when a `correct` stage is configured before `resolve`. Resolve does not produce
them. The `correct` stage does, and a refusal on that file becomes an unresolved span
([evidence-constrained correction](correction.md)). Reviewer decisions are
`review/decisions.json` (`llb.book-ocr-reviewer-decisions`), a
run-level file outside the stage cache. The resolve key includes that page's decisions and no
other page's. A missing file digests the same as an empty list, so a file that does not edit a
page does not re-run it. An invalid file changes the key and the page fails until the file is
fixed. A new decision therefore re-runs exactly the pages it touches. The stage re-reads the
ledger after writing and refuses to continue if those bytes moved.

## How to run

`resolve` is an ordinary stage. It needs `blocks` before it, and a primary whose stage is also
configured before it:

```yaml
stages:
  - render
  - name: ocr
    params:
      engine: <engine>
      variant: original
  - blocks
  - name: resolve
    params:
      primary:
        stage: ocr
        engine: <engine>
        variant: original
```

The stage writes `overlay.json` and `resolved.json`, and the run publishes the same two files at
`pages/<pNNNN>/`. `pages/<pNNNN>/blocks.json` stays the raw ledger.

## Markers

A span the pipeline cannot read reliably is a short Markdown link, not a gap. The same line is
what a person sees in Obsidian and what a machine parses back:

```text
[!unreadable](source/book.pdf#page=3 "p0003-b0001 2-5") ![[review/p0003-b0001-s0002-0005.png]]
```

The link target is the run-relative stored PDF plus `#page=N`. The title is the block id and the
character offsets into the primary reading. The embed is optional. The span kinds are
`unreadable`, `low-confidence`, and `formula-table`. A page whose recognition failed uses
`page-failed`, described below.

An unresolved character span takes `formula-table` on a formula or table block, `unreadable` when
its text is blank, and `low-confidence` otherwise. An empty content block with no unresolved span
is `unreadable`, or `formula-table` when the block is a formula or table. Content here is body,
heading, caption, footnote, unknown, formula, and table. A figure, an ornament, a running header,
a page number, or neighbour-page bleed is not queued only because its text is empty. A primary
reading whose confidence is present and below 0.9, on non-empty content with no unresolved span,
marks the whole block `low-confidence` and keeps the text. A missing confidence does not route.
The 0.9 cut routes a span to review. It is not a probability of correctness. This engine labels
every block `body`, so a displayed relation is `low-confidence` until a layout stage names it
`formula`.

The passage a reader sees replaces each unresolved span with its marker. An empty passage is the
marker alone. A low-confidence block and an orphaned block prefix the marker and keep the text.
A candidate, when one exists, is the other reading of that span: the highest confidence, then
engine, then variant. For a whole block it is the strongest other reading whose text differs.

## The review queue

`llb book-ocr review-queue` reads a finished resolve and writes three things. `review/queue.json`
(`llb.book-ocr-review-queue` 1.0.0) is the workbench ledger. `metadata/quality_manifest.json`
(`llb.book-ocr-quality-manifest` 1.0.0) lists the same spans, one page object per configured page,
and its `counts` equal the rows. Each page also carries `coverage`: `read`, `blank`, `omission`,
or `failed`. Each crop is `review/<block-id>-s<start>-<end>.png`: the block region padded by 16
pixels, clipped to the page, written as a grey PNG. The same span always gets the same name. An
unread region and a failed page use an id of the form `pNNNN-rMMMM`. A page with no render still
gets its marker and its row, with no crop. A rebuild deletes a crop the new queue does not name.
It leaves `review/decisions.json` in place.

The command runs after resolve. It exits 2 when the path is not a run, or when a configured page
is still unfinished. A page that failed terminally is included. The command prints kind counts and
coverage counts. It does not print the recovered text.

```sh
make book-ocr-review-queue BOOK_RUN=<run-dir>
make review-workbench REVIEW_PATH=<run-dir>/review/queue.json
make book-ocr BOOK_RESUME=<run-dir>
make book-ocr-review-queue BOOK_RUN=<run-dir>
```

A workbench decision becomes a reviewer edit. Accept raw writes the raw slice, so an unresolved
digit reappears. Accept candidate writes the other reading and refuses when there is none. Enter
text writes one non-empty line. Mark illegible writes an `[!unreadable]` marker, so a passage
nobody could read stays visible. Clear removes the edit. The edit id is `reviewer:<entry-id>`,
the author is `reviewer`, and the evidence is the decision token. A decision on a missing block
is refused. The raw ledger is not rewritten. The next `resolve` applies the edit, and a later
queue build drops a span that edit covers. A decision already on disk but not yet in the overlay
keeps its verdict, so reopening the queue resumes at the next pending entry.

An orphan comes back with its old decision shown and an empty verdict. When the block is still
there, the new entry is the block's current text and current anchor, so the next decision can
attach. When the block is gone, the old anchor stays and a new decision is refused.

## Blank pages, unread regions, and failed pages

Coverage is the share of a page's ink that falls inside a recognition box. The boxes are the OCR,
formula, and table readings. When none of those readings was written, the ledger boxes stand in
for them. A layout box marks a window whose ink still has to fall inside one of those boxes.
Confidence is not an input. `account_page` in `src/llb/prep/book/review/coverage.py` is the
check the queue, the vault, and validation all call.

A page with no ink and no blocks is `blank`. The manifest records it. The vault shows

```text
[!blank](source/book.pdf#page=4 "p0004")
```

The corpus token is `{{llb-marker:blank:p0004:0-0}}`. The queue has no row for it, and a gold span
may cover the token.

A text column, or a layout box, is an unread region when less than half of its ink falls inside a
recognition box. The crop is the unread band in the rendered page's pixels, so a stretch of the
same column that was read stays outside it. The queue row is kind `unreadable` and reason
`omission`, and it carries that box. Neighbour blocks stay in the ledger and in both outputs. The
vault shows `[!unreadable]` and the crop. The corpus token is
`{{llb-marker:omission:<region-id>:0-0}}`. A gold span that overlaps it is refused.

A page that has a failed stage, nothing still pending, and no resolved text is one queue row of
kind and reason `page-failed`. The vault shows `[!page-failed]` and, when the render exists, the
page crop. The corpus token is `{{llb-marker:page-failed:<page-id>:0-0}}`. A gold span that
overlaps it is refused. `book-ocr structure` includes the page and gives it no lines.

On a finished run, validation refuses a missing queue with `missing-review-queue` and a queue that
has drifted from the ledger, or whose failed pages disagree, with `stale-review-queue`. A run
that has no page manifest keeps a missing or drifted queue as a warning.

## Contracts

The resolution families, each at `1.0.0` with no older form, live in
`src/llb/core/contracts/data_prep/book_resolve.py`. The queue and the manifest live in
`book_review.py`, also at `1.0.0`.

| Family | File |
| --- | --- |
| `llb.book-ocr-correction-overlay` | `pages/<pNNNN>/overlay.json` |
| `llb.book-ocr-resolved-page` | `pages/<pNNNN>/resolved.json` |
| `llb.book-ocr-model-corrections` | `pages/<pNNNN>/corrections.json`, when a `correct` stage wrote it |
| `llb.book-ocr-reviewer-decisions` | `review/decisions.json` |
| `llb.book-ocr-review-queue` | `review/queue.json` |
| `llb.book-ocr-quality-manifest` | `metadata/quality_manifest.json` |

## Evidence

**What ran.** On 2026-11-22, on the GeForce RTX 5050 laptop (8 GB). This pass did not use the
GPU. Resolution and the ledger rebuild are CPU work. Recognition was the RapidOCR 3.10.0 reading
already recorded on the [block ledger](block-ledger.md#evidence) page (PP-OCRv6 small, ONNX
Runtime, CPU), reused rather than repeated.

- **The rule.** `test_a_digit_span_stays_unresolved_on_the_primary_reading` leaves the digit
  unresolved on the primary reading and adopts the rest.
  `test_agreement_across_every_reading_resolves_and_a_majority_does_not` resolves only
  what every reading shares.
  `test_a_reviewer_edit_overrides_a_model_edit_on_the_same_span` lets the reviewer win.
  `test_a_second_engine_orphans_changed_blocks_and_keeps_the_rest` and
  `test_an_edit_does_not_follow_a_block_whose_id_moved` re-queue a changed block and refuse to
  apply the edit to another id. `test_a_new_reviewer_decision_changes_its_page_alone` re-runs that
  page and leaves the next page's resolved text and both ledgers byte-identical.
  `test_reuse_keeps_each_duplicate_page_its_own_block_ids` keeps each of two identical rasters its
  own block ids.
- **Handbook.** The same primary (`ocr`, `rapidocr`, `original`) on all 1,067 pages of the
  *Space Mission Engineering* scan. Render and recognition were reused (1,067 pages each). The
  ledger stage ran, because its key now includes the PDF page, and wrote ledgers byte-identical
  to the recognition run on every page, including the 56 pairs of identical page images. Resolve
  then ran on all 1,067 pages. None failed. Wall time was 28 s. The ledger has 137,827 blocks.
  Each has one reading, so every block is the primary reading: 137,827 primary spans, no
  agreement, no unresolved span, no fallback, no empty reading, no applied edit, no orphan.
  Adopted text equals the primary reading on every block. PDF page 300 still has 109 blocks and
  the same running header the ledger page cites. Resuming the run cached all 4,268 page-stages
  and ran none.

What would overturn the handbook reading: a second engine that disagrees on a digit, a sign, a
decimal, or a symbol would leave that span unresolved and out of the adopted text. A ledger
rebuild that changes a block's region or its raw readings would orphan an overlay entry rather
than move it. A render whose long side is under 1000 pixels is the ledger page's caveat, not a
property of this rule. The task record is
[0010](../../records/0010-scanned-book-recovery-book-reading-resolution-and-correction-overlay.md).

**Review queue.** On 2026-11-23, on the RTX PRO 3000 Blackwell laptop (12 GB). RapidOCR 3.10.0
(PP-OCRv6 small) on its CPU route; this pass loaded no GPU model. The source is the same
1,067-page *Space Mission Engineering* scan. Render was reused. This pass recognized PDF pages
1-15 and 290-299: 25 pages, 75 stages ran, 25 renders were reused, none failed. Wall time was
75 s, of which the OCR stage recorded 73 s.

The 25 pages hold 2,596 blocks. Every block is `body` and has one reading. There is no unresolved
span, no empty reading, and no missing confidence. On pages 1-15 the 1,481 scores have minimum
0.504, 10th percentile 0.91, and median 0.995. On pages 290-299 the 1,115 scores have minimum
0.503, 5th percentile 0.937, 10th percentile 0.964, and median 0.993. A cut of 0.5 queued none of
the 2,596 lines. The cut is 0.9. It queued 179 lines, 136 of the 1,481 and 43 of the 1,115. All
179 are `low-confidence`, each has a crop, and each keeps its line after the marker. The manifest
has 25 pages and 179 rows, and its counts match. Six pages have no row. PDF pages 3 and 7
contributed 38 and 36 of the rows. A second build left the queue file byte-identical.

A displayed relation on PDF page 299 scored 0.883, and a short decimal on PDF page 295 scored
0.840, so both are queued. A one-character mark on PDF page 3 scored 0.504. Italic prose sits in
the same low band. This engine never named a block `formula`, so the relation's marker is
`low-confidence`.

What would overturn the queue reading: a RapidOCR build whose scores sit on another scale, or a
second engine that disagrees on a digit, a sign, a decimal, or a symbol. That disagreement is an
unresolved span, which the cut does not create. A layout stage that names formula and table blocks
would change the marker kind of those lines. The task record is
[0011](../../records/0011-scanned-book-recovery-book-uncertainty-markers-and-review-queue.md).

**Page coverage.** On 2026-11-26, on host 15301-L-C-3078. This pass loaded no model. It accounted
a 25-page RapidOCR 3.10.0 slice of the *Space Mission Engineering* scan, the same engine the
[block ledger](block-ledger.md#evidence) records, including PDF pages 131 and 135. 23 pages were
`read` and 2 were `omission`. PDF page 131 contributed two unread bands, covering about y
0.08-0.44 and y 0.59-0.89 of the render, so the read middle stayed out of the crops. PDF page 135
contributed two bands on the left column, about y 0.08-0.28 and y 0.64-0.90; the right column
stayed covered. `book-ocr review-queue` wrote those four bands as `unreadable` rows with reason
`omission` and printed `coverage read: 23`, `coverage omission: 2`. The same queue also held 458
`low-confidence` rows from the 0.9 routing cut. Those rows are the cut, not coverage.

What would overturn it: a column with ink and no recognition box marked `read`, or a column whose
boxes already cover at least half its ink marked `omission`. The planted gate is
`tests/llb/prep/book/test_region_accounting.py`. The task record is
[0022](../../records/0022-scanned-book-recovery-book-missing-region-accounting.md).
