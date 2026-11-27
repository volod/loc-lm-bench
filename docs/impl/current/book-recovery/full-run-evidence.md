# Book recovery: full-run rehearsal

The registry's full-run clauses, rehearsed on a generated book before a handbook is committed.
Recognition quality is the pilot's measurement. This page is whether the stages agree on page
coordinates, block ids, anchors, and cache keys when they run as one chain. The target handbook
is not an input.

The contract is the full-run sentence of
[scanned book recovery](../../../design/spec.md#scanned-book-recovery). The task record is
[0028](../../records/0028-scanned-book-recovery-book-synthetic-book-end-to-end.md).

## Where it lives

| Module | Owns |
| --- | --- |
| `tests/llb/prep/book/synthetic_pages.py` | The eleven-page scan: pixels and the text the engines return |
| `tests/llb/prep/book/synthetic_book.py` | The fake engines, the planted kill, and the run start |
| `tests/llb/prep/book/test_synthetic_book.py` | The seam runner. A failure names the stage whose output the next stage could not use |
| `tests/llb/prep/book/synthetic_publish.py` | The vault, the projection, ingest, the gold-span refusal, and the reviewer revision |

The run itself is the [book-ocr runner](pipeline-and-resume.md#where-it-lives). Correction stays
off: the open digit is a disagreement between two engines, not a model edit
([technical content](uncertainty-and-review.md#technical-content)). The vault check is
[vault validation](structure-and-rendering.md#vault-validation). The projection is
[corpus projection](corpus-projection.md).

## The generated book

Eleven pages, 420 by 540. Printed numbers start at 10 on PDF page 3, so the offset is 7, and PDF
page 5 has no folio: printed 11 is followed by printed 13, so printed 12 is the one gap. The
contents page carries the roman folio `i`, which stops the arabic series from extending onto it.
The failed page has no text and is front matter.

| PDF | Printed | What the page carries |
| --- | --- | --- |
| 1 | none | The reader raises. No ledger. The vault and the corpus show the page-failed marker |
| 2 | i | Contents: two chapters, one section, and the index |
| 3 | 10 | Heading `1 Mission Design` |
| 4 | 11 | Heading `1.1 Orbit Window`. The reader says `3.2` and the layout engine says `3.3` |
| 5 | 12, gap | A body line and no folio |
| 6 | 13 | Two columns. The left column stays before the right |
| 7 | 14 | Heading `2 Propulsion`, display formula `E = mc^2` numbered `(2-1)`. The planted kill dies once on this page's OCR |
| 8 | 15 | Ruled table `Table 2-1. Stage loads.` with the rows the table engine returns |
| 9 | 16 | Captioned figure `Figure 2-1. Transfer geometry.` and the one unread drawing |
| 10 | 17 | Index. `transfer` points at printed 11 |
| 11 | 18 | A bottom-heavy band rotated 180 degrees, so the signal reads 180. The band lands inside the reader box |

Folios sit below the contents-density band. A folio inside that band is one more contents-like
line, and a short page then continues the contents and loses its heading. The unread drawing is
thin lines with paper between them, inside the figure box and outside every text box. A solid
block is absorbed by the local-median ink mask and never becomes an unread region. The citation
`Figure 9-9` names nothing in the book.

## What the chain checks

`test_synthetic_book_reaches_a_validated_corpus` runs two copies. One is interrupted by a
`SimulatedKill` during OCR of the formula page, leaving that page's `.ocr.partial` and no `ocr`
directory. Resume commits that stage and does not repeat the pages already rendered: the page
images match the uninterrupted run. The failed page stays a terminal failure.

Structure then has eleven pages, one gap, the four headings, the contents page, and no conflicts.
The vault renders the formula with its equation tag, the table grid, the figure embed, the
page-failed marker, and an unreadable marker for the drawing. Citation numbers are links, so a
caption survives as the words on either side of the number. `validate_book` is the check that
every content file is reachable, previous and next links are reciprocal, and every link, anchor,
embed, and uncertain span resolves. The two vaults match once `metadata/render.json` drops
`run_dir`, which is the absolute path of that run.

The projection keeps the open digit and the failed page as corpus markers. The reference sidecar
agrees with the vault: a resolved citation's anchor is in the vault, an unresolved citation names
why, and the index line for `transfer` is printed 11 on PDF page 4. Ingest with a one-character
floor (the synthetic documents are shorter than the default floor) puts page provenance on every
chunk. A gold span over the unresolved marker is refused.

The reviewer enters the disputed digit. Resume, re-render, and re-validate change one content
file, the section that holds that page, and that page's review crop. The re-projection writes one
new document whose `revision_of` is the ingested original, and ingest keeps both. The failed
page's marker is still in the vault and in the corpus.

## How to run

The test is part of `make ci`. It uses fake engines and does not load a model or a GPU.

## What would overturn it

A resumed vault whose notes differ from the uninterrupted run once `run_dir` is set aside. A
reviewer edit that changes a second content file. A gold span over the open marker that
validates. The failed page rendering as prose. A second gap, or a heading that the contents
continuation swallows because a folio was counted as a contents line.
