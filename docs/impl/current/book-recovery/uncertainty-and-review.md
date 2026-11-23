# Uncertainty and review

`resolve` chooses the one text both renderings will read for each block, and records every
correction in an overlay over the raw ledger. The ledger is not rewritten. Markers, the review
queue, model edits, and both renderings are later stages; they read this seam. Operator usage is
the [book-ocr guide](../../../guides/data-prep/book-ocr.md). The ledger they resolve is the
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
| `src/llb/core/contracts/data_prep/book_resolve.py` | The four families below |

Tests: `tests/llb/prep/book/test_resolve.py`, `test_resolve_stage.py`, and
`test_ledger_reuse.py`.

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
read only when a `correct` stage is configured before `resolve`. This stage does not produce
them. Reviewer decisions are `review/decisions.json` (`llb.book-ocr-reviewer-decisions`), a
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

## Contracts

Four families, each at `1.0.0` with no older form. The models are
`src/llb/core/contracts/data_prep/book_resolve.py`.

| Family | File |
| --- | --- |
| `llb.book-ocr-correction-overlay` | `pages/<pNNNN>/overlay.json` |
| `llb.book-ocr-resolved-page` | `pages/<pNNNN>/resolved.json` |
| `llb.book-ocr-model-corrections` | `pages/<pNNNN>/corrections.json`, when a `correct` stage wrote it |
| `llb.book-ocr-reviewer-decisions` | `review/decisions.json` |

## Evidence

**What ran.** On 2026-11-22, on the GeForce RTX 5050 laptop (8 GB). This pass did not use the GPU. Resolution
and the ledger rebuild are CPU work. Recognition was the RapidOCR 3.10.0 reading already recorded
on the [block ledger](block-ledger.md#evidence) page (PP-OCRv6 small, ONNX Runtime, CPU), reused
rather than repeated.

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
