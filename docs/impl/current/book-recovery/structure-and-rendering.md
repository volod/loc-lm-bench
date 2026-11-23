# Structure

The printed page, the hierarchy, and the figure, table, and equation numbers of one scanned book,
read from the block ledger after `resolve`. The vault that renders this file is still open work
([book vault rendering](../../plan.md#book-vault-rendering-and-navigation)).

`llb book-ocr structure` is a command, not a stage. It reads every configured page's ledger, prefers
the resolved text when that text is non-empty, and writes one book-level file. A block's kind is
not an input: a line engine labels every block `body`, and the same pages with every block
relabelled `body` produce the same mapping and the same running headers.

```sh
make book-ocr-structure BOOK_RUN=<run-dir>
```

Exit code 2 is a run that is missing a ledger. The command prints counts. It does not print the
recovered text. A second run over the same ledgers writes the same bytes.

## Where it lives

| Module | Role |
| --- | --- |
| `src/llb/prep/book/structure/` | The inference: lines, page numbers, running headers, contents, headings, reconciliation, inventory |
| `src/llb/core/contracts/data_prep/book_structure.py` | The `llb.book-ocr-document-structure` 1.0.0 model |
| `src/llb/cli/book/structure.py` | `llb book-ocr structure` |
| `make/data-prep/book.mk` | `make book-ocr-structure` |

Tests: `tests/llb/prep/book/test_structure.py`, over synthetic ledgers. No page of a real book
enters git.

The file is `metadata/document_structure.json`. It holds one row per configured PDF page
(`observed`, `gap`, `unnumbered`, or `front-matter`), the running headers, the hierarchy, the
numbering inventory, every conflict, and the PDF pages read as contents.

## How a node is earned

A printed number is a short numeral in the outer corner. The fit keeps a step when the printed
number advances by at most as many pages as the PDF did. An interior page with no numeral is a gap
when the printed numbers kept pace, and an unnumbered plate when they did not. Pages before either
series are front matter. The fit does not invent a number past the series except the single page
touching each end of a pure offset. A hole in the configured pages does not extend a chapter
across the pages that were not read.

A running header is the other short line in the top band, kept when the same text repeats. A
section number there that disagrees with a numbered heading covering the page is a
`header-mismatch`. It does not become a section of its own.

Contents pages are the page that says so and a following page that is still mostly contents
lines. An entry is evidence. It is never a node. A heading is a numbered line that is words rather
than a table cell, a bare chapter number opening a page, or a front-matter name (preface, index,
and the same short list). A taller line is not a heading. Reconciliation pairs an entry to one
heading by number, title, and printed page, and records every entry with no heading, every heading
with no entry, and every title or page that disagrees. The node stays on the page where the heading
was seen.

With no headings, the book falls back to page-range children split where the running header
changes. `fallback` is true on a page or a node the evidence did not settle.

Figure, table, and equation numbers come from a caption line, a footer line, or a display number
`(n)`. A page that is a list of those captions does not inventory them as captions on that page.

## Evidence

**What ran.** On 2026-11-23, on the RTX PRO 3000 Blackwell laptop host, CPU RapidOCR 3.10.0, no
GPU. The source is the 1,067-page target scan. The slice is PDF pages 1-40 and 290-299 (50 pages).
`structure` then read those ledgers.

**Printed pages.** 42 numerals were read, 5 leading PDF pages are front matter with no numeral, and
3 numerals the engine did not read were filled by the fit: PDF 6 as v, PDF 18 as 1, PDF 23 as 6.
In the body the observed offset is 17 (PDF 19 is printed 2, PDF 40 is printed 23, PDF 290 is
printed 273). In the roman front matter the offset is 1 (PDF 7 is vi, PDF 17 is xvi).

**Hierarchy.** The slice yields the preface on PDF 15-17, chapter 1 on PDF 18-40 with sections
1.1-1.5 and the subsections under 1.2, 1.3, and 1.4, then, after the unread span, subsections
10.6.3 and 10.6.4 and section 10.7 with 10.7.1 and 10.7.2. Section 10.7 carries both a contents
match and the running header. 173 contents entries have no heading on these pages; each is a
conflict, not a node. One contents title and one running-header section number disagree with the
heading on the page, and both disagreements are recorded. The inventory holds 97 figure, table,
and equation numbers.

**What would overturn it.** An observed body numeral on this scan that is not 17 behind its PDF
page. A heading on these pages that the file omits, or a node whose text is not on that page. A
second run over the same ledgers that is not byte-identical.
