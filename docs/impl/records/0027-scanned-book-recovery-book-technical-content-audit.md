# book-technical-content-audit

## Work contract

```markdown
#### book-technical-content-audit

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: the [block ledger](current/book-recovery/block-ledger.md) and
  [review semantics](current/book-recovery/uncertainty-and-review.md).
  Admitted and refused model edits are
  [evidence-constrained correction](current/book-recovery/correction.md).
  Figure crops and caption association are
  [figure assets](current/book-recovery/formulas-tables-figures.md#figure-assets).
- User-visible outcome: a reader can trace each technical value or uncertainty marker through the
  chosen recognition route and overlay to the pixels that support it.
- Scope boundary: exercise an ambiguous index and sign, a merged or empty cell, a diagram with
  labels but no caption, competing nearby captions, and an omitted column. Include an engine
  fallback and a stale reviewer edit after rebuilding the ledger.
- Data and artifact paths: `tests/llb/prep/book/` and the audit record.
- Execution path: replay deterministic synthetic analogues in `make ci`.
- Acceptance gates: verify raw readings stay immutable; every edit names its evidence; an
  unsupported digit, caption or cell cannot enter the resolved reading. Pin each exercised routing
  loss to a regression or an explicit abstention. Score specialist output using the emitted formula,
  grid and caption association.
- Documentation target: `docs/impl/current/book-recovery/uncertainty-and-review.md`.
```

## Implementation and decisions

The audit drove a planted four-page scan through the real pipeline: two scripted engines on the
structural `formula` and `table` stages, then `blocks`, `resolve`, structure, the review queue,
the vault, the corpus projection, validation, and the resolved score. Each case in the scope
boundary is one region of that scan (`tests/llb/prep/book/technical_fixture.py`). Run against
the code before this change, the run showed six seams where a technical value the scan does
not settle reached a rendering, or a marker was lost. Each is fixed and pinned:

- **Formula and table structure bypassed resolution.** The vault, the corpus, and the resolved
  score printed the primary engine's LaTeX or grid regardless of the overlay, so an open digit
  was published as math, a reviewer or model edit never reached either output, and a prefix
  marker on a formula was dropped. `src/llb/prep/book/render/emitted.py` now decides once what a
  block publishes: the engine structure only while every span is primary or agreement, the
  structure re-read from the corrected text after an overlay edit, and no structure while a span
  is open. `src/llb/prep/book/render/structured.py` writes either form for the vault and the
  corpus, with leading markers kept above the structure. An open formula is its passage beside
  its crop; an open table is its passage laid out as a grid, so a disputed cell is a marker in
  its cell.
- **Disputed cells, contested captions, and formula letters never opened a span.** A cell
  dispute only added a crop beside the published grid, and an engine caption was never compared
  at all. `src/llb/prep/book/resolve/technical.py` carries each to the primary text: a located
  cell is held, a blank, missing, or unlocatable primary cell holds the whole table, two captions
  that differ dangerously hold the whole table, and a letter two formula readings write
  differently is held as `symbol`. The resolve identity is `llb-book-resolve 2`.
- **Cells were compared by list index.** A merged cell shifted the rest of its row and an
  omitted column was never compared. `src/llb/prep/book/ledger/cells.py` compares the expanded
  grids (`occupancy` in `src/llb/prep/book/grid.py`), treats a merge whose text matched as no
  dispute, and compares a position one grid lacks with a blank. The blocks stage version is 5.
- **A value the primary shortened was adopted.** A character only another reading has was a
  zero-width slice and held nothing, so `0.5` against `0.05`, or a dropped minus, was adopted.
  `src/llb/prep/book/resolve/masks.py` holds the characters on either side of the point, and
  `candidate_text` composes the other reading over the held characters (`.05`).
- **Numbers were parsed across a dropped span.** The adopted text joins the characters around
  an open span, so `Figure 12-3` with an open `3` became figure `12` and named an asset
  `_12`, and a folio could read as another page. `src/llb/prep/book/resolve/numbers.py` masks
  every number an open span touches; structure lines and figure numbering read that text, and a
  caption whose number is unread is not a proximity guess. The cross-reference boundary in
  `src/llb/prep/book/render/grammar.py` refuses a number that runs into a marker or a token.
- **Captions.** The caption under a figure was printed from the gap-joined text, which dropped
  its marker; it is now the caption block's passage. A sentence opening `Figure 3-1 shows` was
  inventoried as figure 3-1's caption and anchored the figure on prose; a numbered opening that
  runs on into a lowercase word is no longer a caption (`src/llb/prep/book/structure/inventory.py`).
  The resolved score now reads the figure-asset association the vault prints
  (`select_figures` and `caption_links` in `src/llb/prep/book/assets/extract.py`, read by
  `src/llb/prep/book/metrics/resolved_captions.py`) instead of the reading-order neighbour.

Shared helpers replaced copies: one `resolved_reading` in `src/llb/prep/book/resolve/primary.py`
serves the renderer, the score, and the queue (the score's copy ignored the model), and
`table_markdown` moved to `src/llb/prep/book/render/mathform.py`. `HypBlock.held_open` marks a
block the lane abstained on, so an open formula is not counted as a rendered formula that fails
to parse.

`test_wrong_caption_and_folio_lose_credit_when_a_guess_would_pass` asserted the score credited
the reading-order neighbour `Fig. 9-9. Wrong.`; the vault has printed `Fig. 3-2. Right.` there
since the figure-asset task, by its printed number. That test is now
`test_the_printed_caption_is_credited_and_a_wrong_folio_is_not`, and its intent -- a guess the
vault does not print earns nothing -- is `test_competing_captions_earn_nothing_when_a_nearest_guess_would_pass`.

## Acceptance evidence

Status: accepted.

On 2026-11-27, on host vola-ASUS-V16-V3607VH1 (ASUS V16), CPU only, no model.

- **Raw readings stay immutable.** The ledger bytes of all four pages are identical before and
  after the queue, structure, vault, and corpus passes, and each block keeps every engine's raw
  text (`test_raw_readings_stay_immutable_and_every_edit_names_its_evidence`). The rebuild keeps
  the primary reading byte-identical and adds the new engine beside it.
- **Every edit names its evidence.** Every applied edit on every page has non-empty evidence and
  source; agreement edits carry `every reading agrees`; the three reviewer edits carry
  `accept-raw`, `accept-raw`, and `enter-text` (`test_technical_review_rebuild.py`).
- **No unsupported digit, caption, or cell enters the resolved reading.** Neither output holds
  `0.5 s`, `0.05 s`, `x = -3`, the raw formula `v_1 - v_i`, `Valve | 35`, `Valve | 36`, the
  omitted-column grid, `Table 2-1`, `Table 2-7`, `Figure 12-3`, or `Figure 12-8`. The three
  figure assets carry no number and no caption, and the inventory holds no figure
  (`test_technical_content_audit.py`). After the rebuild the stale `v_k` is in neither output.
- **Each exercised routing loss is pinned.** Regressions: formula and table structure, the
  disputed cell, the insertion hold, the caption number, the caption marker, the citing
  sentence, the citation boundary, and the score's caption association. Explicit abstentions:
  the formula held as its passage and crop, the whole table held for a blank primary cell and an
  omitted column, the whole table held for contested captions, the caption with an unread
  number, and two competing captions attached to neither. The engine fallback and the stale
  reviewer edit are pinned as recorded routes (`fallback` true with engine `beta` in both
  outputs and the score; three `block_changed` orphans requeued on the open span).
- **Specialist output is scored as emitted.** The open formula submits no math and is
  `held_open`; the agreed formula submits `E = mc^2 \tag{2-5}`; the merged header submits a
  colspan of 2; the whole-held table submits no grid; no figure is credited a caption
  (`test_the_score_reads_what_the_vault_emits`).
- **Validation.** The planted vault validates with no finding and 9 queue entries.

The same two integration modules were run once against the code before this change: 6 of their
8 tests failed. Validation of that vault reported 4 `marker-queue-mismatch` findings (formula
spans 2-3 and 4-5, the table cell, the caption digit) and 4 unreferenced crops. The figure asset
was `p0003_0008_figure_12` with the disputed caption attached. The vault and corpus printed the
raw formula, `Valve | 35`, and `Table 2-1`, and the vault anchored `^fig-3-1` on the citing
sentence. The two tests that passed pin properties that already held: raw-reading immutability
and the fallback route.

Gates: `make format`; the book, review, and goldset suites; `make ci`; `make lint-md`;
`git diff --check`. Results are in the handoff.

## Handoff

No new work was routed. All six seams were fixed in place. Two things a later task will see:
the target run's queue will grow wherever two table engines write the same grid in different
text forms (HTML against pipes), because the text rule already holds every differing symbol
open; and `src/llb/prep/book/grid.py` was already past the soft size limit, which the closing
audit's module-split scope covers.
