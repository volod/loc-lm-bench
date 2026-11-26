# Structure and the reading vault

The printed page, the hierarchy, and the figure, table, and equation numbers of one scanned book,
read from the block ledger after `resolve`, and the Obsidian vault rendered from that file.

`llb book-ocr structure` is a command, not a stage. It reads every configured page's ledger, prefers
the resolved text when that text is non-empty, and writes one book-level file. A block's kind is
not an input: a line engine labels every block `body`, and the same pages with every block
relabelled `body` produce the same mapping and the same running headers.

```sh
make book-ocr-structure BOOK_RUN=<run-dir>
```

Exit code 2 is a page whose ledger is still unfinished. A page that failed terminally is included
and contributes no lines. The command prints counts. It does not print the recovered text. A
second run over the same ledgers writes the same bytes.

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
conflict, not a node. 22 headings on these pages have no contents entry. One contents title
and one running-header section number disagree with the heading on the page, and both
disagreements are recorded. The inventory holds 97 figure, table, and equation numbers.

**What would overturn it.** An observed body numeral on this scan that is not 17 behind its PDF
page. A heading on these pages that the file omits, or a node whose text is not on that page. A
second run over the same ledgers that is not byte-identical.

## Reading vault

`llb book-ocr render` reads the structure file and the resolved text and writes an Obsidian
vault. It is a command, not a stage. A page with no resolved text becomes one marker and a link
to that PDF page. The command prints the vault path. It does not print recovered text. It refuses
a missing structure, a source checksum that disagrees, or a page list that disagrees.

```sh
make book-ocr-render BOOK_RUN=<run-dir>
```

The default vault is `book/<book-slug>/` inside the run. `--output` publishes elsewhere, replaces
the directory in one step, and drops files the new vault does not contain. Exit code 2 is a
missing structure, a bad path, or a destination inside the scan's directory. After publish,
render runs [vault validation](#vault-validation) and exits 1 when a check fails.

| Module | Role |
| --- | --- |
| `src/llb/prep/book/render/` | Names, line joining, section files, Markdown, cross-reference links, and publish |
| `src/llb/cli/book/render.py` | `llb book-ocr render` |
| `make/data-prep/book.mk` | `make book-ocr-render` |

Tests: `tests/llb/prep/book/test_render.py`, over synthetic ledgers. No page of a real book
enters git.

The vault renderer verifies the source PDF checksum before publishing. It keeps page paths
and crop coordinates in memory. During publication it copies fallback PNGs byte for byte,
streams a source PDF under the copy policy, and decodes a page on demand for crops with a
one-page cache. Figures and review crops retain the source render's colour channels.
`test_render.py` checks colour crops, exact fallback bytes and refusal of a substituted PDF.

### What a reader opens

The vault holds `00_Table_of_Contents.md`, the figure, table, and equation lists, `content/`,
`assets/`, `review/`, `metadata/`, and `source/`. A content note is one structure node, or one
page-range part when the node's resolved text passes 12,000 characters. The cut falls on a
block boundary. A formula, a table, and a
figure with its caption stay whole. A paragraph splits only when that paragraph alone passes
the cap.

A node that resumes after one of its children becomes a later file. Previous and next reach it.
The contents lists the node's first span, including the parts the character cap made. Each note
opens and closes with the same links: the contents, the neighbouring notes, and the PDF page
the note starts on. The page link is a relative path plus `#page=<n>`, counted from 1.

Front matter holds `book`, `node_id`, `pdf_start`, `pdf_end`, `printed_start`, `printed_end`,
and `review_count`. Printed numbers stay strings, so a roman numeral is not read as an integer.

A numbered target's anchor is its printed number: `^ch-1`, `^sec-1-1`, `^sub-1-2-2`,
`^fig-1-1`, `^tbl-2-1`, `^eq-1-1`. The first block of a PDF page carries `^p0007`. Changing
block ordinals leaves those anchors in place. A block whose ledger kind is `figure` is a tight
crop at `assets/<chapter-slug>/p<NNNN>_<ord>_<kind>`, plus the printed number when the inventory
has one. Review crops keep the queue's 16-pixel padding. A line engine that labels every block
`body` yields the anchors and no figure crops.

A formula is display math. The inventory's equation number becomes a `\tag` when the text has
none. A `](` inside that math is LaTeX, not a relative link. An image or a wikilink in the
same math is still a link. A table stays Markdown when every line contains a pipe, HTML when
the text is already a table, and one column otherwise. An unresolved span keeps its review
marker. A running header or a printed page number that structure identified stays in the note,
labelled.

Consecutive body lines in one column join into a paragraph. A block whose ledger `paragraph`
flag is true joins the next body block only when its text ends in a hyphen. A footnote, a
column break, an indent after a sentence, and the column's `Table ... Fig. ... Eq. ...` index
line each start a new paragraph. A word split by a hyphen at the column edge is joined, and so
is a hyphen whose tail is the first unit of the next page in the same column. The rest of the
right column stays in reading order, after the left column's footnote and index line. The
corpus keeps one fragment per block, so the joined word is in the vault and the block that
held the tail still cites the remainder.

`source/` follows the run's three policies. `symlink` is the default. `copy` writes a read-only
PDF. `reference` writes `source/reference.txt` and points the page links at the PDF where it
already is. `BOOK_PAGE_FALLBACK=1` adds `source/pages/<page-id>.png` and a link on each note.
The fallback and the source policy leave content file names unchanged.

`metadata/render.json` records the slug, the content files, the source checksum, the section
cap, the source policy, the page-fallback switch, `run_dir`, `obsidian_version` `1.15`, and
`pdf_page_fragment` `page`. `run_dir` is how a vault published outside the run finds that run.
Obsidian 0.10.8, released 2021-01-11, introduced
`[[file.pdf#page=3]]`. The 1.15 help for PDF files and for embeds still documents that fragment.
Opening the vault in the app is a later human task.

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_RUN` | A book-ocr run that already has a document structure | required |
| `BOOK_OUTPUT` | Vault directory | `book/<slug>/` inside the run |
| `BOOK_SOURCE_POLICY` | `copy`, `symlink`, or `reference` | `symlink` |
| `BOOK_PAGE_FALLBACK` | Set to `1` to write a rendered image of each page | off |
| `BOOK_SECTION_CAP` | Resolved characters in one note. The floor is 200 | `12000` |
| `BOOK_SLUG` | Lowercase ASCII vault name | the PDF file name |
| `BOOK_REFERENCE_GRAMMAR` | YAML grammar that replaces the handbook cues | the handbook grammar |

Citations are [cross-references](#cross-references).

### Evidence

**What ran.** On 2026-11-23, on the RTX PRO 3000 Blackwell laptop host, render read the 50-page
RapidOCR slice of the 1,067-page target scan: PDF pages 1-40 and 290-299, CPU RapidOCR 3.10.0,
no GPU. The render itself does no recognition and uses no GPU.

**The vault.** 52 content notes and no failed page. 238 review crops. No figure crops, because
the line engine labeled every block `body` and a crop is written only for kind `figure`. The
inventory numbers are anchors on the lines structure named. Section 1.1 is one note, PDF 18-20,
printed 1-3. The untitled front matter is one span in five parts under the 12,000-character cap.
On PDF 19 the column break inside "necessary" is one word again, and the footnote and the
Table/Fig/Eq index line are their own paragraphs. A second render of the same ledgers matched
the first byte for byte, symlink target included.

**What would overturn it.** A second render of these ledgers that differs by a byte. A block of
kind `figure` on this slice with no crop. A numbered anchor that changes when only the block
ordinals change. An Obsidian release that drops `#page=`: the recorded version and the link
form change together.

## Cross-references

Render links a citation only when one target carries that number. A miss stays the words the
page printed, and the report counts it. The command prints the totals. It does not print the
sentence.

`BOOK_REFERENCE_GRAMMAR` is a YAML file and replaces the handbook grammar entirely. A missing
file, an unknown key, or a grammar with no page cues or no index titles is exit code 2.

The handbook cues are `Fig.` / `Figs.` / `Figure` / `Figures`, `Table` / `Tables`, `Eq.` /
`Eqs.` / `Equation`, `Sec.` / `Section`, `Chapter` / `Chap.` / `Ch.`, and `p.` / `pp.` /
`page`. A comma in place of the period (`Fig,`, `Eq,`) is the scanned form of the figure and
equation cues. Parentheses around an equation number are optional. A comma, `and`, or a dash
joins further numbers, and each number resolves on its own. `Chaps. 3-6` cites chapters 3 and
6. `Chapter 22.1.5` does not cite chapter 22: a dotted tail is not a range. `Fig. 1-1` is one
figure number, because the figure pattern already contains the hyphen.

A figure, table, or equation is unique when one caption or one display number carries it. A
page-footer echo of that number is the running index line, not a second item. Two captions
with one number stay plain text, and so do several footers with no caption. The report names
those keys. A section or a chapter is unique when one hierarchy node carries that number.

A printed page links when one observed page carries that label. The link lands on the anchor
of the first block of that PDF page, so a page split across two notes opens where it starts.
A mapping gap stays plain, and so does a label that is both a gap and an observed page. The
report counts a gap citation under unresolved and under gaps. The citation is a warning. It
does not fail the check.

An index note is one whose title matches an index title in the grammar. The handbook title is
`index`. On those notes, a comma list or a dot leader at the end of a line cites a printed
page. The same tails in any other note stay text. This 50-page slice has no index; the index
rule is pinned by the fixture.

The link keeps the cue and wraps the number. `Fig. 1-1` points at `note.md#^fig-1-1`,
`Eq. (1-1)` points at `note.md#^eq-1-1`, and `p. 4` points at `note.md#^p0004`. From a
content note the path is that note's file name, including when the target sits in the same
file. From a list at the vault root the path starts with `content/`. A fenced block, display
math, an existing link, inline code, and an anchor line are not citations.

The vault root holds `01_List_of_Figures.md`, `02_List_of_Tables.md`, and
`03_List_of_Equations.md`. The contents links them. They sit outside the previous/next chain.
A row is the number, a link to that item's own anchor, the printed page when the inventory
has one, and the PDF page. The row does not copy the caption. Two items that share a number
both appear. An empty kind says `None.`

`metadata/reference_links.json` records the grammar name, the counts by kind, the ambiguous
keys, and the gap count. It quotes no sentence and names no run. The vault report repeats the
counts under `## Cross-references`, and `validation.json` carries the same object. A missing
report is a warning.

| Module | Role |
| --- | --- |
| `src/llb/prep/book/render/grammar.py` | The handbook grammar, and a file that replaces it |
| `src/llb/prep/book/render/catalog.py` | Which numbers are unique, ambiguous, or a mapping gap |
| `src/llb/prep/book/render/link.py`, `index_links.py` | Citation spans, and index tails |
| `src/llb/prep/book/render/lists.py`, `references.py` | The three lists and the count file |
| `src/llb/prep/book/validate/references.py` | The report section and the gap warning |

Tests: `tests/llb/prep/book/test_reference_links.py`. Every reference type resolves on a
fixture. Two figures numbered `1-1` stay plain, and the key is reported. A printed page
inside a mapping gap stays plain. A page split across two notes links to the note where that
page starts. A grammar of one cue replaces the handbook cues.

### Evidence

**What ran.** On 2026-11-23, on the RTX PRO 3000 Blackwell laptop host (12 GB), CPU only.
Render read the document structure and the resolved text of the 50-page RapidOCR 3.10.0 slice
of the 1,067-page target scan: PDF pages 1-40 and 290-299. Recognition was not repeated. The
slice has no index.

**Counts.** Resolved 128: chapter 2, equation 10, figure 61, section 19, table 36. Ambiguous
52: equation 23, figure 6, table 23. The ambiguous keys are `equation 1-0`, `figure 1-4`,
`figure 10-314`, `table 1-0`, `table 1-2`, and `table 10-26`. Unresolved 251: chapter 48,
equation 3, figure 21, page 3, section 126, table 50. Mapping gaps left unlinked: 0. The
three unresolved page citations name printed pages this slice does not contain. Most
unresolved section and chapter citations name nodes outside these 50 pages. Five cues of the
form `Chapter` followed by a dotted number stayed plain. No figure cue was linked as a single
digit. The lists hold 26 figure rows, 18 table rows, and 4 equation rows.

**The check.** `book-ocr validate` reported 0 findings and the same two warnings the structure
file already accounts for: 197 structure conflicts, and 3 printed pages that used a fallback.
A second render wrote the same reference report and the same validation file.

**What would overturn it.** A second render of these ledgers whose reference counts differ.
One of those five dotted chapter cues becoming a link. A hyphenated figure number linked as
two numbers. One of the ambiguous keys above linked from prose. A gap citation on a later run
that is neither plain text nor a warning.

## Vault validation

`llb book-ocr validate` checks one rendered vault and writes the report into that vault.
`render` runs the same check after it publishes. `run` and `resume` run it for each vault
already under `book/` that has `metadata/render.json`. A run with no vault is unchanged, and a
dry run does not check one. The check reads files. It uses no GPU.

```sh
make book-ocr-validate BOOK_DIR=<book-dir>
```

Set `BOOK_RUN` when the vault was published outside its run. A vault inside the run, or one
whose `metadata/render.json` records `run_dir`, finds the run itself.

Exit code `0` means every check passed. Exit code `1` means a check failed, and the command
names the file. Exit code `2` means the directory is not a vault, or the run cannot be read.
The command does not print recovered text.

The report is `metadata/processing_report.md` in the vault. `metadata/validation.json` is the
same outcome, with one record per PDF page. The run directory keeps a separate
`metadata/processing_report.md` for its stages. The two files share a name and live in
different directories.

Every PDF page is completed, failed, or skipped. A selected page with a `^pNNNN` anchor is
completed. A page whose note carries `[!page-failed]` is failed, and that status wins over an
anchor. A page outside the configured range is skipped. Consecutive skipped pages share one
row. `validation.json` lists every page. A selected page with neither an anchor nor a failure
marker fails the check.

Every content note is reachable from the contents. Previous and next name each other, and
walking Next from the first note visits every content note once. The contents Next starts
that chain.

Every relative link, wikilink, embed, and block anchor resolves. A heading fragment uses the
same slug as `make lint-doc-links`. A PDF fragment is `#page=<n>` with `n` inside the source
page count. An image under `assets/`, `review/`, or `source/pages/` is linked from a note, and
every linked image exists. A note whose front matter does not name its PDF pages fails the
check.

When `review/queue.json` exists, each open span marker matches one open queue entry. The match is
the marker kind, the block id, the offsets, and the PDF page. A failed page is matched as a page,
apart from those spans. On a finished run a missing queue is `missing-review-queue` and a queue
that has drifted from the ledger, or whose failed pages disagree, is `stale-review-queue`. Both
fail the check. A run with no page manifest keeps a missing or drifted queue as a warning, and
the markers are then matched to the ledger. Orphaned overlay entries are counted. The count is
not itself a failure. Structure conflicts and printed-page fallbacks are warnings.
The report's Cross-references section repeats the counts render wrote
([cross-references](#cross-references)). A printed-page citation that falls in a mapping gap
is a warning and does not fail the check.

The report lists the configured engines, models, and parameters, the block readings, and the
confidence buckets. A bucket is diagnostic routing evidence, not a probability that the
characters are correct. It also lists the open queue by marker kind and uncertainty class, and
the pages and content types still open for a person.

| Module | Role |
| --- | --- |
| `src/llb/prep/book/validate/` | The checks and the vault report |
| `src/llb/cli/book/validate.py` | `llb book-ocr validate` |
| `make/data-prep/book.mk` | `make book-ocr-validate` |

Tests: `tests/llb/prep/book/test_vault_validate.py`. The clean fixture passes. Each failure class is
a broken copy of that fixture, and the finding names the file.

### Evidence

**What ran.** On 2026-11-23, on the RTX PRO 3000 Blackwell laptop host (12 GB), CPU only. The
vault is the render of the 50-page RapidOCR 3.10.0 slice of the 1,067-page target scan: PDF
pages 1-40 and 290-299. The review queue was built from those ledgers, the vault was rendered
again, and `book-ocr validate` checked it.

**Accounting.** All 1,067 PDF pages are accounted for: 50 completed, 0 failed, and 1,017
skipped because they sit outside `1-40,290-299`. The skipped pages collapse to two rows,
41-289 and 300-1067. The vault has 52 content notes and 0 findings. The manifest records
render, ocr, blocks, and resolve as done on all 50 pages, and 25 of each stage were reused.

**What still needs a person.** The queue holds 238 open `low-confidence` entries, all on
`body` blocks, and no `unreadable` or `formula-table` entry. No uncertainty class is set on
those entries. Orphaned overlays: 0. The report warns that 197 structure conflicts remain
(173 contents entries with no heading, 22 headings with no contents entry, one title
mismatch, and one header mismatch) and that 3 printed pages used a fallback. Those warnings
match the structure file. They do not fail the check.

**Readings.** 5,278 RapidOCR readings, model `3.10.0, PP-OCRv6 small`, variant `original`.
Every reading has a score. The minimum is 0.501 and the maximum is 1.000. The buckets are 0
below 0.50, 131 in 0.50-0.80, 107 in 0.80-0.90, 120 in 0.90-0.95, and 4,920 in 0.95-1.00.
The two buckets below 0.90 sum to 238, the same count as the open queue. The routing cut is
0.9, so a score in 0.90-0.95 is counted and not queued.

**What would overturn it.** A page in this vault with no anchor and no failure marker. A
content note the contents cannot reach. A queue entry with no marker, or a marker with no
queue entry. A second check of this vault that reports a finding. A confidence-bucket total
other than 5,278.
