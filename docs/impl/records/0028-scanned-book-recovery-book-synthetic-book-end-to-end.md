# book-synthetic-book-end-to-end

## Work contract

```markdown
#### book-synthetic-book-end-to-end

Every stage is proven on its own fixtures, and the first place they all meet is a multi-day GPU run
over 1,067 pages -- the most expensive place to learn that two stages disagree about a coordinate
frame, a block id, an anchor, or a cache key. Rehearse the whole chain in CI first, on a generated
book small enough to run in seconds and built so that every seam carries a case.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- fake engines over a generated scan; no model and no GPU.
- Dependencies: the synthetic scan factory the run tests already use
  ([run ledger and resume](current/book-recovery/pipeline-and-resume.md#where-it-lives)).
  What a formula or table publishes while a value is open is
  [technical content](current/book-recovery/uncertainty-and-review.md#technical-content).
  The projection command is
  [corpus projection](current/book-recovery/corpus-projection.md).
  The vault check is
  [vault validation](current/book-recovery/structure-and-rendering.md#vault-validation).
  Figure crops and caption association are
  [figure assets](current/book-recovery/formulas-tables-figures.md#figure-assets).
  The optional correction stage is
  [evidence-constrained correction](current/book-recovery/correction.md).
  Cross-reference linking is
  [cross-references](current/book-recovery/structure-and-rendering.md#cross-references).
- User-visible outcome: one CI test takes a generated multi-chapter scan from the PDF to a validated
  vault and an ingested corpus projection, and fails naming the stage whose output broke the next.
- Scope boundary: in scope -- a generated scan with a contents page, numbered headings, printed
  numbers offset from PDF pages with one gap, a page placed upside down, a two-column page, a formula
  with an equation number, a ruled table, a captioned figure, a cross-reference and an index entry,
  an unreadable span, and one page whose recognition fails; every stage through the `book-ocr`
  runner with fake engines whose readings disagree on one digit; render, link, validate, project,
  the reference sidecar checked against the vault's links, `ingest-corpus`, and a
  `validate-goldset` refusal of a span over the open marker; a planted kill
  mid-run and a resume; one reviewer decision followed by a re-render and a re-projection. Out of
  scope -- recognition quality (the pilot measures it) and any real page.
- Data and artifact paths: the generator beside the existing scan factory under
  `tests/llb/prep/book/`; outputs in the test's temporary directory only.
- Execution path: inside `make ci`, CPU-only.
- Acceptance gates: `make ci` green; every full-run clause of the registry evaluation holds on the
  synthetic book -- every page accounted for, every file reachable from the contents, previous/next
  links reciprocal, every link, anchor, and embed resolving, no uncertain span without its marker and
  queue entry, a byte-identical re-render after the resume -- plus page provenance on every ingested
  chunk and the gold-span refusal; the reviewer decision changes the rendered and projected text of
  its page alone and the re-projection ingests as a revision; the failed page shows as its marker in
  both renderings.
- Documentation target: `docs/impl/current/book-recovery/full-run-evidence.md`.
```

## Implementation and decisions

The rehearsal is tests only. No production module changed.

`tests/llb/prep/book/synthetic_pages.py` builds the eleven-page scan. `synthetic_book.py` runs it
through render, signals, variants, classes, layout, OCR, formula, table, blocks, and resolve, with
four fake engines. The reader and the layout engine disagree on one digit (`3.2` against `3.3`).
Correction stays off. The reader raises on PDF page 1 and dies once, via `SimulatedKill`, on the
formula page's OCR. `test_synthetic_book.py` walks the seams through structure and names the seam
that failed. `synthetic_publish.py` renders, validates, projects, checks the reference sidecar,
ingests, refuses a gold span over the open marker, and applies one `enter-text` decision.

Three fixture facts the first runs got wrong:

- A folio inside the contents-density band is one more contents-like line. On a short page that
  continues the contents and drops the heading. Folios now sit at y 0.965, still in the footer,
  below that band. The contents page carries roman `i`, so the arabic series does not extend
  backward onto it, and the textless first page stays front matter. The only gap is PDF page 5.
- `clean_page()` paints glyph rows on the whole page. Those rows are ink no text box covers, so
  coverage marked almost every page unread. Pages are flat grey except where a case needs ink.
  The upside-down page is one bottom-heavy band rotated 180 degrees into its reader box, so the
  signal reads 180 and the ink is covered. Column ink sits inside the two reader boxes. The
  unread span is thin lines in the figure box: a solid block is absorbed by the local-median ink
  mask and never becomes a region.
- `metadata/render.json` records the absolute run directory. The byte compare of the resumed vault
  and the uninterrupted vault drops that field. The notes match.

Ingest uses a one-character floor. The synthetic documents are shorter than the default floor of
500, and the check is page provenance, not the floor. The detail is
[full-run rehearsal](../current/book-recovery/full-run-evidence.md).

## Acceptance evidence

Status: accepted.

On 2026-11-27, on host vola-ASUS-V16-V3607VH1 (ASUS V16), CPU only, no model and no GPU. The scan
is generated in the test temporary directory. The target handbook was not opened.

Command: `uv run python -m pytest tests/llb/prep/book/test_synthetic_book.py`. Result: 1 passed.
The same test runs inside `make ci`.

What the test requires, and what that means:

- **Every page is accounted for.** The manifest has 11 pages. PDF page 1 is front matter with no
  folio and no ledger, and it is a terminal failure. The contents page is roman `i`. Printed 10
  is observed on PDF page 3. Printed 12 is the one gap. Printed 13 is observed on the column page.
- **The cases survive their stages.** The rotated page reads 180 degrees and the formula page
  reads 0. The column page is `multi_column`, and the left column stays before the right. The
  dispute opens a digit span whose adopted text is not the raw `3.2`. The formula's equation
  number is `2-1`. The table block carries a grid. The failed page has no `blocks.json`.
- **Resume does not repeat finished renders.** The kill leaves `.ocr.partial` and no `ocr`
  directory on the formula page. After resume the page images match the uninterrupted run.
- **The vault validates, and the two renders match.** `validate_book` reports ok, which is the
  check for contents reachability, reciprocal previous/next links, and resolving links, anchors,
  embeds, and uncertain spans. The vaults match once `run_dir` is removed from `render.json`.
  The formula carries its tag, the table grid is present, the figure is embedded, and both the
  page-failed marker and an unreadable marker are in the notes. The corpus uses marker tokens;
  the vault does not.
- **The sidecar matches the vault.** A resolved citation's anchor is in the vault. An unresolved
  citation has no target and a reason of ambiguous, gap, or absent. The index term `transfer`
  points at printed 11 on the dispute page. `Figure 9-9` is still in the corpus as a citation the
  book does not satisfy.
- **Ingest and the gold gate.** Every chunk has page provenance. A gold span over the unresolved
  marker is an error containing `unresolved marker`.
- **One reviewer decision, one page.** `enter-text:2` changes one content file and that page's
  review crop. The re-projection has one revision. The new document's `revision_of` is the
  ingested original, the new file contains `3.2`, and ingest keeps both files. The page-failed
  marker is still in both renderings.

## Handoff

`make plan-status` before the removal: 68 tasks, agent lane 43, human lane 25, priority agent
task `book-synthetic-book-end-to-end`. After the removal: 67 tasks, agent lane 42, human lane 25.
The priority agent task is `book-publication-and-resume-audit`, which no longer requires this
task; its dependency line points at the rehearsal page.
`scanned-book-recovery-implementation-audit` drops this id and still requires the publication
audit and the remaining recovery tasks.

No new work was routed. The publication audit still owns the long generated book, and the target
handbook stays with `book-full-run-on-the-target-scan`. The operator's target scan is present under
`$DATA_DIR/books/` (1,067 pages, A4 image-only PDF); nothing in this change opens it. The area
index, `current.md` page count, and the book-ocr guide now point at the rehearsal page.
