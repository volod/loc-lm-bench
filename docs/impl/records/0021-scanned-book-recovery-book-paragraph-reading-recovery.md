# book-paragraph-reading-recovery

## Work contract

```markdown
#### book-paragraph-reading-recovery

A plausible page box can still hold interleaved lines from two columns, and a paragraph can lose
its continuation when a draft appends stray OCR lines. Recover reading order and usable paragraph
text from explicit page regions, including on folios with bleed, tables, and boxed examples.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- use planted text and captured synthetic engine payloads.
- Dependencies: the [block ledger](current/book-recovery/block-ledger.md), original coordinate
  maps from [page preprocessing](current/book-recovery/page-preprocessing.md), and the
  [pilot worksheet](current/book-recovery/tool-selection.md#pilot-reference-review).
- User-visible outcome: the vault, projected corpus, and worksheet preserve supported reading
  order and paragraph boundaries, with the crop and disagreement visible when placement is unclear.
- Scope boundary: carry the engine's paragraph and reading-order evidence through the ledger.
  Match line reads within their owning region. Join across a column or page only on source
  evidence of continuation. Handle a span across columns, a boxed aside, a caption and its table,
  and a rotated page with neighbouring bleed. Keep unmatched candidates accessible by region.
- Data and artifact paths: `src/llb/prep/book/`, synthetic fixtures under `tests/llb/prep/book/`,
  and page traces under the book run directory.
- Execution path: run the normal stages and render, project, and prepare the worksheet over a
  planted scan with competing structured and line readings inside `make ci`.
- Acceptance gates: planted paragraphs retain their text and order with no cross-column
  interleaving, duplicates, or unsupported joins; an unmatched line cannot become trailing body
  text without placement evidence; a title, caption, cell, or boxed passage cannot be folded
  into another paragraph; page variants return each region to its source pixels. The fixture
  detects order, paragraph text, and table-caption association failures individually.
- Documentation target: `docs/impl/current/book-recovery/block-ledger.md`.
```

## Implementation and decisions

A page with both a structural stage (`layout`, `formula`, or `table`) and a line reading is
ordered in `src/llb/prep/book/ledger/reading.py`. A line joins the tightest region that covers
at least half of the smaller box. A body or `unknown` box that only wraps two or more tighter
regions is dropped. A line two regions of similar area claim about equally stays its own block.
A body region that spans the gutter, with at least two lines on each side that do not themselves
cross it, is replaced by those lines. A line with at least half its width inside a region stays
there. A heading, caption, table, formula, figure, footnote, or piece of furniture inside a body
region stays its own block. A boxed aside is `unknown`. A bare integer in the top or bottom 12
percent of the page, owned by no region, is a `page-number`. Pages with only one stage keep the
existing overlap cluster.

`paragraph` on the block is true only when the block kept a structural region. The field is
optional on `llb.book-ocr-block-ledger` 1.0.0 and defaults to false. The renderer joins
consecutive body lines in one column, and joins a finished region only when its text ends in a
hyphen. The tail is the other column, or the first unit of the next page in the same column.
The corpus keeps one fragment per block. The worksheet places an unmatched line at its crop
("No region owns this line.") and lists the rival regions when placement is unclear
("Placement is unclear.").

Code: `src/llb/prep/book/ledger/reading.py`, `src/llb/prep/book/ledger/assemble.py`,
`src/llb/prep/book/pilot_review/merge.py`, `src/llb/prep/book/render/join.py`,
`src/llb/core/contracts/data_prep/book_blocks.py`.
Detail and the measured pages:
[block ledger](../current/book-recovery/block-ledger.md#reading-order-and-paragraph-boundaries).

## Acceptance evidence

Status: accepted.

- **Planted order, paragraph text, and table-caption association.**
  `tests/llb/prep/book/test_paragraph_reading.py` and
  `tests/llb/prep/book/test_paragraph_pipeline.py` run the normal stages over a planted
  two-column scan with a page-sized interleaved box, a stray line, a boxed aside, a table and
  its caption, a hyphen across a column and onto the next page, and a rotated page with
  neighbouring bleed. The ledger order matches the planted order. The stray line and the line
  two regions claim equally are their own blocks. The cell stays on the table. The caption and
  the table are neighbours and each keeps its own text. The vault joins the hyphen tails. The
  corpus keeps one fragment per block. The worksheet flags the stray line "No region owns this
  line." and the rival line "Placement is unclear.", with the two regions as the crop. The
  rotated body region maps back to the source blob, and the bleed text stays out of that
  reading.
- **Pilot pages.** On 2026-11-26, on the RTX PRO 3000 Blackwell laptop (12 GB), the saved Docling
  layout readings and RapidOCR line readings of the 25 pilot pages of the 1,067-page
  *Space Mission Engineering* scan were merged again. The merge read those saved readings on the
  CPU. Each rendered page is 1241 by 1755 pixels. The merge produced 885 blocks: 769 with
  `paragraph` true, 116 line-only, 19 tables, and 4 page numbers. Fourteen layout strings appear
  on two regions, matching two proposals from the layout engine. On the three-column index page,
  31 line readings inside a table whose centers fall across the gutter stay on that table.
- **Gates.** On 2026-11-26 `make ci` was green: 5,249 passed, 7 skipped, 50 deselected.
  `make lint-md` was green on the docs, including the plan after this task left it.

## Handoff

The next recovery tasks read the ledger in this order. A structural region large enough to
contain two other regions is dropped even when it is one real paragraph. A cell with less than
half its width inside its table stays a separate line.
