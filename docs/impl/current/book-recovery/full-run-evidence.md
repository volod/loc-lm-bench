# Book recovery: full-run rehearsal

The registry's full-run clauses are rehearsed on generated books before handbook publication.
The rehearsals check whether recovery and publication agree on page coordinates, block IDs,
anchors, and cache keys. A diagnostic on the target scan also exercises rendering and provenance.
The pilot measures recognition quality.

The contract is the full-run sentence of
[scanned book recovery](../../../design/spec.md#scanned-book-recovery). The task records cover
the [eleven-page rehearsal](../../records/0028-scanned-book-recovery-book-synthetic-book-end-to-end.md)
and the [publication audit](../../records/0029-scanned-book-recovery-book-publication-and-resume-audit.md).
The [target-run record](../../records/0032-scanned-book-recovery-book-full-run-on-the-target-scan.md)
tracks the provisional handbook profile and its acceptance gates.

## Where it lives

| Module | Owns |
| --- | --- |
| `tests/llb/prep/book/synthetic_pages.py` | The eleven-page scan: pixels and the text the engines return |
| `tests/llb/prep/book/synthetic_book.py` | The fake engines, the planted kill, and the run start |
| `tests/llb/prep/book/test_synthetic_book.py` | The seam runner. A failure names the stage whose output the next stage could not use |
| `tests/llb/prep/book/synthetic_publish.py` | The vault, the projection, ingest, the gold-span refusal, and the reviewer revision |
| `tests/llb/prep/book/publication_fixture.py` | The 64-page extension, command driver, and decoded-image measurement |
| `tests/llb/prep/book/test_publication_audit.py` | Interruption and retry, fallback pixels, a repeated folio, and a retrieval hit traced to its page region |
| `tests/llb/prep/book/test_publication_edges.py` | Rollback on filesystems without directory exchange and exact projection of split paragraphs |

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

## Publication and resume audit

On 2026-11-27 on an ASUS V16 V3607VH with an Intel Core 5 210H, the normal commands ran the 11-page
control and a 64-page generated book with injected engines and a 1,000-character section cap.
The larger run decoded over four times as many pages during render and validation. Each run
held at most two full decoded render arrays at a time. This bounds live decoded page memory
by page size. The run still keeps page paths and text metadata for the whole book.

The fixture interrupts OCR and then publication of both outputs. Successful OCR-stage records
survive resume unchanged. The vault retry restores the same bytes at the same destination. The
clean and resumed vault and corpus trees also match after accounting for the run directory and
the acquisition run ID and capture time. All 64 fallback PNGs match the original render files
byte for byte, including the failed page and the RGB page. Direct PDF rasterization independently
matches the pixels on three pages. The original PDF copy also matches byte for byte.

The repeated printed folio 18 remains plain text in the vault and is marked ambiguous in the
reference sidecar. Deleting the queue entries fails vault validation. Gold validation refuses a
span over the open marker. A lexical retrieval hit on PDF page 64 leads from the staged document
offset to its block, the continuation note's anchor, and the colored rectangle inside the cited
pixel region. The reviewer edit creates one new document revision. The old document stays
readable and one incoming reference updates its target to the new revision.

## Target scan diagnostic

On the same host and date, `make book-ocr` with the `render` stage processed all 1,067 pages of
*Space Mission Engineering: The New SMAD* at the scan's native resolution. The run took about
69 seconds in 43 batches of up to 25 pages. Resume reported 1,067 cache hits and zero stage
executions. The source SHA-256 was
`89101813d4c695e9f992d540e751c4e2f4b80c6550f24d5d23e1a795383cb7c8`.

For PDF pages 1, 30, 31, and 1,067, a separate CPU run used RapidOCR 3.10.0 and ONNX Runtime
1.32.0 on the original render, with signals, routing classes, block assembly, and resolution.
Resume reused all 24 stage results. The fallback PNGs matched the full scan renders byte for byte
and matched direct PyMuPDF rasterization pixel for pixel. Validation accounted for four completed
pages and 1,063 outside the selection, with zero findings and one structure-conflict warning.
The queue held three low-confidence entries. The vault linked ten printed references and left six
unresolved.

The local-only projection ingested all three documents, with scan provenance on all 60 fixed
180-character chunks. All 296 citation rows led to existing vault anchors after the repair.
Previously, 54 rows from PDF page 30 pointed at a missing anchor in its second note. At caps of
200, 500, 1,000, and 12,000 characters, the vault and corpus selected the same 60, 26, 13, and 3
note paths respectively. At the smallest cap, the pre-fix layouts selected 66 and 64 paths.
A repeat at the default cap reproduced hashes and symlink targets for all 21 vault members
and 12 corpus files.
The diagnostic exercises these publication and provenance paths before full-book recognition.

## Provisional MinerU profile

The [MinerU Standard profile](../../../../samples/book_ocr/space-mission-engineering-the-new-smad/provisional-mineru-standard.yaml)
selects 25-page batches and the original native-resolution render. MinerU 4.2.3 reads each page
once into middle JSON, retaining block coordinates for equations, tables, figures, and text.
The `layout` reading supplies `blocks` and `resolve`. Signals and primary classes describe
the pages and the review load. The profile follows the
[unscored handbook pilot](tool-selection.md#handbook-pilot-reading), whose 25 warm recognition
calls averaged 11.115 seconds per page, or 3.3 hours projected for 1,067 pages. The run keeps
the scan and projection local-only. Its reference accuracy and release profile remain open.

On 2026-11-28, on an HP ZBook Fury G1i with an RTX PRO 3000 Blackwell laptop GPU (12,227 MiB),
`make book-ocr` selected all 1,067 pages of the same SHA-256 above. The interruption occurred
after 22.3 seconds, with 78 committed stage records including three MinerU readings. Resume
picked up at PDF page 4. The committed records retained their SHA-256 digests, and the resume log
contained no new execution of any of those stages.

The first 100 pages were published from a separate cache-only run that reused all 600 stage
results. Vault validation accounted for 100 completed pages and 967 outside that selection,
with zero findings. Two renders matched across 277 files and symlinks, and the 100 fallback
PNGs matched the original render bytes. The vault resolved 295 printed references, left 184
ambiguous and 406 unresolved, and reported nine structure conflicts and seven fallback folios.
Of the 107 notes, one continuation held a figure and page furniture with no text to project.
Ingest retained all 106 projected documents. All 2,611 fixed 180-character chunks carried
source-PDF page provenance covering all 100 selected pages, and all 1,243 citation rows pointed
to existing vault anchors. The review queue held zero entries. MinerU supplied no confidence
score for any of the 1,627 block readings, so the queue measures flagged issues within this
profile's available signals.

### Full provisional run

The full run completed 43 batches and all 1,067 pages through render, signals, classes,
layout, blocks, and resolve. It reused 3,201 artifacts for the render, signals, and class stages
from the same scan. There were zero failed stages. The two invocations used 7,659 seconds
(2.13 hours) of wall time, including preflight and excluding the pause between invocations.
The elapsed time including the pause was 7,679 seconds. The 1,067 successful recognition calls
averaged 7.162 seconds. The earlier warm-pilot projection was 11,860 seconds (3.29 hours) on an
RTX 5050 laptop.

The vault has 1,218 content notes. Validation accounted for every page with zero findings.
It reported 42 structure conflicts and 650 fallback folio mappings. Of the printed references,
5,800 resolved, 2,548 remained ambiguous, and 3,397 remained unresolved. Of the unresolved,
445 were printed-page references in a mapping gap. The report preserves these limitations for
operator review.

The resumed run and a fresh render from the same cache matched across 3,174 files and
symlinks. All 1,067 fallback PNGs matched the cached render bytes, and the vault's PDF matched
the source SHA-256. Ingest retained all 1,216 local-only projected documents. All 32,062
fixed 180-character chunks carried source-PDF provenance, together covering every input page.
All 13,791 citation rows resolved to existing vault anchors. Two vault continuation notes had
no projected text. Other notes supplied citations for both pages.

The review queue was empty across every primary class and all 72 inferred chapter or
outside-chapter groups. The primary-class counts were 38 dense formulas, 56 grey background,
34 gutter or corner text, 120 labelled diagrams, 31 landscape, 58 large tables, 425 multi-column,
223 normal, 26 severe curvature, and 56 small footnotes. MinerU supplied no confidence score
for any of the 18,044 block readings. The queue is a count of issues detected by the available
signals. Accuracy and the inferred chapter and printed-page mappings await
[release review](../../plan.md#human-assisted-tasks).

## How to run

The generated books run in `make ci` using CPU fixtures. The
[book-ocr guide](../../../guides/data-prep/book-ocr.md) gives the commands for a local scan.

## What would overturn it

A resumed vault whose notes differ from the uninterrupted run once `run_dir` is set aside. A
reviewer edit that changes a second content file. A gold span over the open marker that
validates. The failed page rendering as prose. A second gap, or a heading that the contents
continuation swallows because a folio was counted as a contents line. Any growth in live decoded
pages beyond the bound, a retry that loses the prior vault, a linked repeated folio, or a
citation row whose note, anchor, and source pixels disagree would reopen the publication audit.
Any repeat of a committed stage on resume, differing vault bytes from the same cache, or a
chapter or class count that disagrees with the queue would reopen full-run qualification.
