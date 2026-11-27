# book-missing-region-accounting

## Work contract

```markdown
#### book-missing-region-accounting

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies:
  [reading order](current/book-recovery/block-ledger.md#reading-order-and-paragraph-boundaries)
  records where a line joins its region. Also the
  [review queue](current/book-recovery/uncertainty-and-review.md),
  [validation](current/book-recovery/structure-and-rendering.md#vault-validation), and
  [corpus projection](current/book-recovery/corpus-projection.md).
- User-visible outcome: an operator can distinguish a blank scan, an unread region on an otherwise
  successful page, and a page whose recognition failed, in both outputs and in the review queue.
- Scope boundary: compare original page evidence with layout and recognition coverage. Route
  uncertain omissions to review with original coordinates; preserve confirmed blanks. Permit a
  terminal page failure in a finished queue while refusing an unfinished run.
- Data and artifact paths: `src/llb/prep/book/`, page quality manifests and review crops.
- Execution path: fake an empty response, a missing second column, a legitimate blank and a
  terminal engine failure through the normal run, queue and render commands inside `make ci`.
- Acceptance gates: identify each planted omission even if every returned block is confident;
  account for all four pages, retain readable neighbours, and reject a missing or stale queue
  entry at full-run acceptance. Check the projected machine markers and PDF-page provenance
  against the same ledger.
- Documentation target: `docs/impl/current/book-recovery/uncertainty-and-review.md`.
```

## Implementation and decisions

`account_page` in `src/llb/prep/book/review/coverage.py` compares the rendered page with the OCR,
formula, and table boxes. A layout box shows a region and is not itself coverage. When those
readings were not written, the ledger boxes stand in. Confidence is not an input. An under-covered
column becomes the unread bands, so a read stretch in the middle stays out of the crop.

A blank page is a manifest coverage and a vault and corpus marker, with no queue row. An unread
region is a queue row (`unreadable` / `omission`) carrying the original box and a crop. A terminal
engine failure is a `page-failed` row in a finished queue. An unfinished page is still refused.
`book-ocr structure` includes the failed page and gives it no lines.

The same accounting feeds the vault, the corpus tokens (`blank`, `omission`, `page-failed`), and
validation. On a finished run a missing queue is `missing-review-queue` and a drifted queue, or a
disagreement about which pages failed, is `stale-review-queue`. A run with no page manifest keeps
the previous warnings. `omission` and `page-failed` refuse a gold span. `blank` does not.

The queue and manifest stay schema `1.0.0`. `coverage` was added on each manifest page, and `box`
on a queue entry, in place.

## Acceptance evidence

Status: accepted.

The planted gate is `tests/llb/prep/book/test_region_accounting.py`. One six-page synthetic PDF
goes through `start_run`, then `book-ocr structure`, `review-queue`, `render`, `project`, and
`validate`. The pages are a readable neighbour, an empty engine response, a second column whose
returned blocks are all at confidence 0.99, a blank page, a terminal `RuntimeError`, and a second
neighbour. The manifest coverage is `read`, `omission`, `omission`, `blank`, `failed`, `read`.
Both neighbours and the left-column text stay in the vault and the corpus. The right column's
text is absent. The omission's citation page is 3 and its region starts to the right of the left
block. Deleting the queue fails validation with `missing-review-queue`. Dropping one omission
fails it with `stale-review-queue`. `tests/llb/prep/book/test_page_coverage.py` covers a layout
box that does not hide a missed column, a read stretch kept out of the band, and an unfinished
page refused.

On 2026-11-26, on host 15301-L-C-3078, the same check accounted a 25-page RapidOCR 3.10.0 slice of
the *Space Mission Engineering* scan, including PDF pages 131 and 135. 23 pages were `read` and 2
were `omission`. Page 131's unread ink is two bands (about y 0.08-0.44 and y 0.59-0.89). Page
135's unread ink is two left-column bands (about y 0.08-0.28 and y 0.64-0.90); the right column
stayed covered. `book-ocr review-queue` wrote four `unreadable` / `omission` rows and printed
`coverage read: 23`, `coverage omission: 2`.

What would overturn the scan reading: a column with ink and no recognition box marked `read`, or
a column whose boxes cover at least half its ink marked `omission`.

On 2026-11-26 `make ci` was green: 5,255 passed, 7 skipped, 50 deselected. `make lint-md` was
green on the docs, including the plan after this task left it. The generated schemas for
`llb.book-ocr-review-queue` and `llb.book-ocr-quality-manifest` stay at `1.0.0` and now carry
`coverage`, `box`, and the `omission` and `page-failed` reasons.

`make plan-status` before the task left the plan: 75 tasks, agent lane 50, human lane 25.
After: 74 tasks, agent lane 49, human lane 25. No registry row changed. Removing the id released
it from `book-technical-content-audit` and from `scanned-book-recovery-implementation-audit`.

## Handoff

No residual work under this task. Formula and table recovery, figure assets, and the technical
content audit remain open in the scanned-book group.
