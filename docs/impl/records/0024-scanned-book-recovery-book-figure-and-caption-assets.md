# `book-figure-and-caption-assets`

## Work contract

```markdown
#### book-figure-and-caption-assets

Figures, graphs, and drawings carry labels, legends, and scales that a tight crop loses, and a
scanned page also carries ornaments and noise that a loose detector extracts as figures. Extract
them conservatively, keep each with its printed caption, and name them so they sort as they appear.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- synthetic figure regions and captions.
- Dependencies:
  [reading order](current/book-recovery/block-ledger.md#reading-order-and-paragraph-boundaries)
  records where a line joins its region. The vault already names a figure crop from its PDF page,
  ordinal, and kind
  ([reading vault](current/book-recovery/structure-and-rendering.md#reading-vault)).
- User-visible outcome: each figure is one asset named `assets/<chapter-slug>/p<NNNN>_<ord>_<kind>`
  plus a slug only when one derives reliably, padded to keep labels, legends, scale markings, and the
  caption, embedded at its reading position with its printed caption and source-page link.
- Scope boundary: in scope -- grouping of multi-region figures into one asset with every source
  region recorded; padding policy; caption association by proximity and printed number; rejection
  of ornaments and badges that repeat across pages and of isolated noise; crops taken from the best
  geometric variant while recording the original region. Out of scope -- redrawing or vectorizing
  figures, and any generated description or alt text beyond the printed caption.
- Data and artifact paths: `src/llb/prep/book/assets/`; `assets/` in the book directory.
- Execution path: synthetic fixtures inside `make ci`.
- Acceptance gates: `make ci` green; asset names are stable across reruns and sort in reading
  order; a repeated ornament never becomes a figure; a figure split across two regions appears
  once; a wrong or ambiguous caption cannot replace the printed association.
- Documentation target: `docs/impl/current/book-recovery/formulas-tables-figures.md`.
```

## Implementation and decisions

The vault calls `llb.prep.book.assets`. The ledger schema stays `llb.book-ocr-block-ledger`
1.0.0. Source regions, the variant, and rejections live in `metadata/figure_assets.json`.

The ordinal in the asset name is the primary region's reading order, so names sort as the
figures appear. A number slug is written only for a unique caption-evidence inventory row, or
for exactly one caption whose text opens as a figure number. A page-footer inventory row with
no such caption is not a slug. The slice showed a real caption the inventory had called a
footer; the caption text is what associates it. A caption that opens as a table or an equation
is not a proximity guess. Neighbor pairing in the vault is unchanged; the associated caption is
suppressed so its text appears once, under the embed.

Pixels come from dewarp, else deskew, else a quarter turn, else the original render. The
recorded boxes stay in the original frame. Repeated badges and isolated specks are relabelled
`ornament` in the note. The corpus projection does not receive the new render fields; its
defaults leave captions in place and write no figure assets.

## Acceptance evidence

Status: `accepted`.

Synthetic gates, in `tests/llb/prep/book/test_figure_assets.py` and the existing vault tests:
a split figure is one asset and the names follow reading order and stay byte-identical on a
second render; a badge repeated on two pages is `repeated-ornament` and is not embedded; an
isolated speck is `isolated-noise`; a nearer caption that prints a different number does not
replace the printed one; two captions for one number, and two unnumbered captions at a similar
gap, stay unattached; a footer number with no caption is not a slug; a caption the inventory
calls a footer still supplies the number when its text prints it; a table caption is not
attached; the crop pixels come from the kept dewarp variant while the recorded region stays on
the original render.

On 2026-11-27, on host vola-ASUS-V16-V3607VH1 (ASUS V16), `make book-ocr-render` rendered the
25-page configured slice of the *Space Mission Engineering* scan. This pass did not re-OCR.
17 figure-kind regions on 12 pages became 16 assets. 11 carry a printed number, including one
caption the inventory had recorded as a page footer. One asset joins two regions. None were
rejected. The slice has no kept geometric variants, so every crop is the original render.
Vault validation reported 0 findings. The numbers and what would overturn them are in
[figure assets](../current/book-recovery/formulas-tables-figures.md#figure-assets).

`make ci` on 2026-11-27: 5289 passed, 2 skipped, 50 deselected. `make lint-md` reported
no broken links and no citation or record-name findings. `make format` left the tree
unchanged.

## Handoff

`make plan-status` before the edit: 72 tasks, agent lane 47, human lane 25, 31 waiting on open
Requires, priority agent task `book-figure-and-caption-assets`. After the edit: 71 tasks,
agent lane 46, human lane 25, 31 still waiting, priority agent task
`book-evidence-constrained-correction`. CLEAR went from 33 to 32.

Removing this task drops its id from `book-technical-content-audit`,
`book-synthetic-book-end-to-end`, and `scanned-book-recovery-implementation-audit`. Each of
those still requires other open tasks, so the waiting count did not fall. No capability
registry row moved. `scanned-book-recovery` still has open work, and the next agent task is
`book-evidence-constrained-correction`.
