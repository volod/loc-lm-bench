# Formulas and tables

A formula or a table keeps the structure the engine returned. The ledger stores it on the raw
reading. The vault and the corpus both render from that reading, and the score uses the same
grid. Nothing here derives a formula, converts a unit, or calculates a value.

Figure crops stay the tight assets the vault already writes. This page does not add a figure
extractor. See [structure and the reading vault](structure-and-rendering.md#reading-vault).

The contract is [Scanned book recovery](../../../design/spec.md#scanned-book-recovery). The
task record is
[0023](../../records/0023-scanned-book-recovery-book-formula-and-table-recovery.md).

## What a reading carries

`llb.book-ocr-block-ledger` stays at `1.0.0`. A raw reading may carry a formula or a table.
Older ledgers that omit the fields still load.

A formula is LaTeX, display or inline, the printed equation number without the parentheses
that only delimited it, and whether that number sits on the left or the right. A display
formula is `$$` or `\[...\]`. A whole line of `$...$` or `\(...\)` is inline. Math inside a
paragraph stays in that paragraph.

A table is rows of cells. An empty string is a cell the engine saw and left blank. `colspan`
and `rowspan` default to 1. `header_rows` is how many leading rows are headers. `caption` is
the engine's caption for that table. `faithful` is false when the cells overlap or a cell
nests another table, which HTML cannot hold.

A block may also carry cell disputes. Two engines that disagree on a cell run the same
dangerous-substitution check the character spans use. A digit, sign, decimal, or symbol
difference, or a named substitution such as `5/S`, is a dispute. Each dispute keeps both
engines' cell texts. A letter-only difference stays on the raw readings and is not a dispute.

Docling and MinerU attach this structure from their own JSON. A Markdown reading does too:
a pipe grid or an HTML table, and a preceding `Table` line stored as the caption without
being copied into the grid text. A raw `raw_blocks.json` that already contains `formula` or
`table` keeps them.

When the formula stage and the table stage of one engine write the same text for one block,
the ledger keeps that text once.

## How an export is chosen

Display math is `$$...$$`. The equation number is the engine's printed number when it has
one, otherwise the structure inventory's number, written as `\tag{...}` on the right. Inline
math is `$...$`, with a parenthesized number after it when the engine recorded one. The
number is not placed inside the math as a second copy of the parentheses.

The local parser is the one the quality harness uses. A bare alignment `&` outside an
environment is wrapped in `\begin{aligned}...\end{aligned}` when the body parses with those
ampersands removed, which is the form Obsidian's MathJax renders. The export keeps the
wrapped form. A body that still does not parse is emitted as the engine wrote it.

A formula keeps a crop beside that math when it is below the validity bar or the agreement
bar. Validity is the parser accepting the body. Agreement fails when the reading's confidence
is under the review cut (0.9), the block has an uncertainty span, or two formula readings
disagree on the canonical LaTeX or the equation number. The crop is
`review/<block-id>-formula.png`, padded like the other review crops. The corpus has no page
assets, so it does not embed that crop.

A printed number the engine spaced out, or wrapped as `\text{(10-5)}`, is still the equation
number. A trailing parenthetical in the equation-number slot (after `\quad` or `&`) may use
`\cdot` or `\div` where the dash was; those two commands count as the dash there. A product
that is not in that slot is left in the formula. An exponent such as `e^{(24-9)}` is not
read as a number.

A table with one header row, a rectangular grid, and no span other than 1 is Markdown, and
an empty cell stays an empty cell. Any other faithful grid is HTML. An unfaithful grid is
the page image under `assets/<chapter>/`, and the export does not also print the grid. A
cell dispute keeps `review/<block-id>-table.png` beside the grid that was published. The
corpus has no assets: an unfaithful table contributes its caption only.

The caption is printed above the grid when no other block already has that text. A `Table`
line that is its own block is not repeated.

## Separate table faults

Table similarity still scores the grid. Four failures are also named on their own, so a
fixture can tell them apart: `altered-cell` (same shape, one cell text changed),
`collapsed-header` (a header row or a header cell dropped, body rows unchanged),
`lost-empty-cell` (an empty cell removed and the other cells kept), and `wrong-caption`.
Each planted case returns that one name.

## Where it lives

| Piece | Module |
| --- | --- |
| Formula and table from engine text or a Docling grid | `src/llb/prep/book/grid.py` |
| Docling JSON, including captions and flat cells | `src/llb/prep/book/engines/parse_docling.py` |
| MinerU content lists and raw JSON | `src/llb/prep/book/engines/parse_common.py` |
| Markdown, dots.ocr, and grounding text | `src/llb/prep/book/engines/parse_text.py` |
| Cell disputes | `src/llb/prep/book/ledger/cells.py` |
| Ledger merge | `src/llb/prep/book/ledger/assemble.py` |
| Parse check, Markdown / HTML / image | `src/llb/prep/book/render/mathform.py` |
| Validity and agreement bars | `src/llb/prep/book/render/bars.py` |
| Formula and table crops | `src/llb/prep/book/render/technical_assets.py` |
| Vault prose | `src/llb/prep/book/render/prose.py` |
| Corpus text | `src/llb/prep/book/project/text.py` |
| Raw and resolved scoring | `src/llb/prep/book/metrics/submission.py`, `src/llb/prep/book/metrics/resolved.py` |
| Named table faults | `src/llb/prep/book/metrics/table_faults.py` |
| Fixture | `tests/llb/prep/book/test_formula_table.py` |

The raw score uses the native grid whenever the reading has one. The resolved score uses
that grid when the export is Markdown or HTML, and submits no grid when the export is the
page image. A reading with no structure still falls back to the text.

## Evidence

On 2026-11-26, on host 15301-L-C-3078, Docling 2.150.1 read two pages of the *Space Mission
Engineering* scan: PDF page 258 (a dense-formula page) and PDF page 768 (a large-table page).
The formula and table stages both ran. Render stayed cached on the reparse. Both pages were
`read`. The review queue had no entries. Vault validation passed. Structure's printed-page
mapping used its fallback for one of the two pages, which is the slice, not a formula result.

Page 258 produced 9 formula blocks. Six carry a right-hand equation number: `10-2a`, `10-2b`,
`10-3`, `10-4`, `10-5`, and `10-6`. Six of the nine parse. The vault embeds a review crop for
the other three, including the formula numbered `10-6`, whose body still does not parse. The
two numbers Docling wrote with `\cdot` and `\div` in the number slot are `10-2b` and `10-3`.

Page 768 produced 4 formula blocks, all parsed. Two carry `24-11` and `24-12` on the right.
The other two have no printed number in the engine text. It also produced one table: 12 rows
by 4 columns, one header row, no merged cells, 6 empty cells, caption attached, rendered as
Markdown. The empty cells are empty Markdown cells. The caption's `109` is the engine's
reading and was not rewritten.

What would overturn this reading: a display formula whose `\tag` is not the number printed
on that formula, a formula the local parser accepts that still embeds a crop, a formula the
parser rejects with no crop, or one of those 6 empty cells missing from the Markdown.
