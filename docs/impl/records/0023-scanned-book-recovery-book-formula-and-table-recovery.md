# book-formula-and-table-recovery

## Work contract

```markdown
#### book-formula-and-table-recovery

Formulas and tables carry the book's technical content, and they are where generic OCR fails worst:
indices, Greek letters, minus signs, and equation numbers in one, merged cells and numeric columns in
the other. Preserve native formula and table structure through the ledger and both
exports before scoring the candidates against a human reference.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- planted formulas and table structures through the normal stages.
- Dependencies:
  [reading order](current/book-recovery/block-ledger.md#reading-order-and-paragraph-boundaries)
  records where a line joins its region. The vault renders formula and table text
  ([reading vault](current/book-recovery/structure-and-rendering.md#reading-vault)).
- User-visible outcome: formulas render as valid LaTeX with the book's equation number kept as a
  tag, and an uncertain formula keeps its crop beside the LaTeX; tables render as Markdown when one
  header row and no merged cells suffice, as HTML otherwise, and with the original image retained
  when even HTML cannot hold the structure faithfully.
- Scope boundary: carry native structured engine output, including table caption associations,
  row and column spans, empty cells, and formula number and position; inline versus display math;
  local LaTeX parse validation compatible with Obsidian; Markdown/HTML/image fallback; route
  disputed table cells through dangerous-substitution checks and preserve their source evidence.
  Formula derivation, unit conversion, and calculation stay outside recovery.
- Data and artifact paths: `src/llb/prep/book/engines/` and `src/llb/prep/book/render/`; formula and
  table crops under `assets/` and `review/` in the book directory.
- Execution path: synthetic formula and table fixtures inside `make ci`.
- Acceptance gates: `make ci` green; the emitted formula parses and keeps its equation
  number, each formula below the validity or agreement bar carries its crop, and the fixture
  catches an altered cell value, collapsed header, lost empty cell, and wrong caption separately.
- Documentation target: `docs/impl/current/book-recovery/formulas-tables-figures.md`.
```

## Implementation and decisions

The ledger stays `llb.book-ocr-block-ledger` 1.0.0. A raw reading gained optional `formula` and
`table`. A block gained `cell_disputes`. Missing fields still load.

A formula reading is LaTeX, display or inline, the printed equation number, and the side it
sits on. A table reading is the cell grid, header-row count, caption, and `faithful`. Docling,
MinerU, Markdown, and a raw block file attach that structure. The vault and the corpus render
it. Display math uses `\tag` for the number. A formula the local parser rejects, or that the
readings do not agree on, keeps `review/<block-id>-formula.png`. A bare alignment `&` is wrapped
in `aligned` when the body parses without it.

A table with one header row and no merged cells is Markdown, and an empty cell stays empty.
Spans and extra header rows are HTML. An unfaithful grid is the page image and is not also
printed. A cell dispute keeps both engines' texts and a review crop. The corpus has no assets,
so an unfaithful table is its caption only, and a formula crop is not embedded.

A printed number the engine spaced, or wrote as `\text{(10-5)}`, is the equation number. In the
number slot, `\cdot` and `\div` count as the dash. An exponent and a product outside that slot
do not. Two stages of one engine that wrote the same text are stored once.

`altered-cell`, `collapsed-header`, `lost-empty-cell`, and `wrong-caption` are separate faults.
The raw score uses a native grid when the reading has one. The resolved score omits the grid
when the export is the image.

## Acceptance evidence

Status: accepted.

`tests/llb/prep/book/test_formula_table.py` is the planted gate: a parsed formula keeps
`\tag{10-28}`; a formula below the validity bar or the agreement bar embeds its crop; Markdown
keeps an empty cell; a span is HTML; an unfaithful grid is an image and is not scored as a
published grid on the resolved lane; `5` against `S` is one cell dispute with substitution
`5/S` and both sources; each of the four table faults fires alone; Docling and Markdown keep
the caption, the span, the empty cell, and the equation number.

On 2026-11-26, on host 15301-L-C-3078, Docling 2.150.1 read PDF pages 258 and 768 of the
*Space Mission Engineering* scan. Page 258: 9 formulas, six equation numbers (`10-2a` through
`10-6`), six parsed, three review crops. Page 768: 4 formulas, all parsed, numbers `24-11` and
`24-12`, and one 12-by-4 table with 6 empty cells, no spans, its caption, rendered as Markdown.
Both pages were `read`. The review queue had no entries. Vault validation passed. One printed
page used the structure fallback, which is the two-page slice. The detail is in
[formulas and tables](../current/book-recovery/formulas-tables-figures.md#evidence).

What would overturn the scan reading: a `\tag` that is not the printed number, a parsed formula
that still embeds a crop, a rejected formula with no crop, or one of those empty cells missing
from the Markdown.

On 2026-11-26 `make ci` was green: 5,266 passed, 7 skipped, 50 deselected. `make lint-md` was
green on the docs, including the plan after this task left it. The generated schema for
`llb.book-ocr-block-ledger` stays at `1.0.0` and now carries `formula`, `table`, and
`cell_disputes`.

`make plan-status` before this task left the plan: 74 tasks, agent lane 49, human lane 25.
After: 73 tasks, agent lane 48, human lane 25. No registry row changed. Removing the id released
it from `book-technical-content-audit`, `book-synthetic-book-end-to-end`, and
`scanned-book-recovery-implementation-audit`.

## Handoff

No residual work under this task. Figure assets and the technical-content audit remain open in
the scanned-book group.
