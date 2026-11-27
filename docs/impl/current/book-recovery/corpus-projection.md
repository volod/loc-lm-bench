# Corpus projection

The retrieval rendering of a scanned book. It reads the same ledger and the same document
structure as the [reading vault](structure-and-rendering.md#reading-vault), and it writes the
corpus shape `ingest-corpus` already reads. The vault keeps navigation, front matter, and
Obsidian markers for a person. This rendering keeps the content, and it cites the PDF page of
every character.

`llb book-ocr project` is a command, not a stage. It needs a run that already has a document
structure. The redistribution class has no default: a missing or unknown class is exit code 2.
The command prints how many documents it wrote and how many of them revise an earlier file. It
does not print the recovered text.

```sh
make book-ocr-project BOOK_RUN=<run-dir> BOOK_CORPUS=<corpus-dir> BOOK_LICENCE=local-only
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `BOOK_RUN` | A book-ocr run that already has a document structure | required |
| `BOOK_CORPUS` | The corpus directory to write | required |
| `BOOK_LICENCE` | `redistributable` or `local-only`, declared by the operator | required |
| `BOOK_LANGUAGE` | Language recorded on every document | detected from the text |
| `BOOK_SECTION_CAP` | Packed source characters in one document. The floor is 200 | `12000` |
| `BOOK_SLUG` | Lowercase ASCII book name used in vault file paths | the PDF file name |
| `BOOK_REFERENCE_GRAMMAR` | YAML grammar that replaces the handbook cues | the handbook grammar |

The output refuses to replace the run directory, and it refuses a directory that overlaps the
source PDF. A second projection of an unchanged ledger writes the same bytes.

## Where it lives

| Module | Role |
| --- | --- |
| `src/llb/prep/book/project/text.py` | Content text: formulas as LaTeX, tables as Markdown, markers as tokens |
| `src/llb/prep/book/project/spans.py` | Page spans that partition the document, and the block citation on each span |
| `src/llb/prep/book/project/identity.py` | The document id, and `revision_of` when the text changes |
| `src/llb/prep/book/project/write.py` | The Markdown file and its citation, reference, and metadata sidecars |
| `src/llb/prep/book/project/references.py` | Citation spans, index lines, and the hierarchy place |
| `src/llb/prep/book/project/build.py` | `project_corpus` |
| `src/llb/prep/book/render/index_segments.py` | One printed line per cell of a glued index column |
| `src/llb/core/contracts/data_prep/book_references.py` | The `llb.book-references` contract |
| `src/llb/prep/corpus/sidecars.py` | The `.references.json` suffix ingest and the bundle share |
| `src/llb/cli/book/project.py` | `llb book-ocr project` |
| `src/llb/goldset/markers.py` | The refusal of a gold span that overlaps an unresolved marker |
| `make/data-prep/book.mk` | `make book-ocr-project` |

Tests: `tests/llb/prep/book/test_project.py`, `tests/llb/prep/book/test_reference_links.py`,
and `tests/llb/prep/book/test_index_terms.py`, on synthetic books. No page of a real book
enters git. Ingest copies a sibling citation sidecar and a sibling reference sidecar when a
text document has them (`src/llb/prep/corpus/ingest_text.py`), and a stale document takes
both with it.

## What a document contains

One content file of the vault becomes one corpus document. Furniture stays in the ledger and
out of the text: running headers, page numbers, ornaments, and neighbour-page bleed. The
generated contents, the previous and next links, and the YAML front matter are not written.
A formula is the same LaTeX the vault emits, including the equation-number `\tag`, and it
does not embed a page crop. A table is Markdown or HTML when the grid is faithful. An
unfaithful table contributes its caption only, because the corpus has no page assets. The
choice is in [formulas and tables](formulas-tables-figures.md). A figure keeps the caption
the page printed.

An unresolved span, an empty block, a blank page, an unread region, a failed page, and a
low-confidence or orphaned reading are a machine token, not a callout:

```text
{{llb-marker:<reason>:<block-id>:<start>-<end>}}
```

`unresolved`, `empty`, `omission`, and `page-failed` are the reasons `validate-goldset` refuses.
A span that overlaps one of those tokens is not evidence. `blank`, `low-confidence`, and
`orphaned` stay visible and do not refuse a span. A blank page's token names the page. An
omission token names the unread region. A failed page's token names the page.

The citation sidecar is the existing `llb.pdf-citations` contract, parser `book-ocr`. Its
page spans partition the document, including the gap before a block and the trailing newline.
Each block records its id, its region, the character range of its text, the vault file, the
vault anchor, and any marker's ledger offsets and document offsets. A block with no numbered
anchor cites the page anchor. The offsets are indexes into the document text.

The metadata sidecar is `<document>.md.metadata.json`. `source_system` is `book-ocr`, so the
document takes the acquired lane. `payload_digest` is the source PDF checksum. `language` is
the operator's value or the detected language. `licence` is the class the operator declared.
`book_logical_key` is the part's identity and is ignored by ingest.

## Printed references

`llb book-ocr project` writes `<document>.md.references.json` beside each document, under
the `llb.book-references` contract at version `1.0.0`. The scan is the vault's catalog and
resolvers, and `BOOK_REFERENCE_GRAMMAR` replaces the handbook cues for both. A resolved
citation records the cited token's offsets in this document and the target block's offsets
in the document that holds that block. An ambiguous key, a mapping gap, and a number with
no target block in the projection stay unresolved and name no target. The reason is
`ambiguous`, `gap`, or `absent`.

An index line records the term's offsets and the PDF pages its printed pages map to. A
printed label this projection does not observe stays on the line as unresolved. A range
covers every printed page it spans, up to 40 pages, and a longer range keeps its endpoints.
A line of only page numbers continues the previous term. A sub-entry records its parent's
term offsets when the line is indented under one. A recognizer that glues an index column
into cells of the form `[{'type': '...', 'content': '...'}]` is read as one printed line per
cell, and the vault linker uses the same split. A line that also contains other text stays
one line. The hierarchy path runs from the book root to the document's node and stores each
node's id, level, number, and title. The sidecar stores offsets, so a later report can name
the span and leave the sentence in the document.

`ingest-corpus` copies the sidecar with the document and deletes it when the document goes
stale. The document fingerprint folds the sidecar in when the file is present
([mixed-corpus ingestion](../data-prep/ingestion-corpora.md)). A corpus bundle lists each
sidecar as a member. A `local-only` export refuses the document before it writes the output
directory, so the sidecar stays with the corpus.

## Identity and revisions

The logical key is the source checksum, the structure node, and the part index of that node
inside the layout. The document id is `book-` plus the first 16 hex characters of the key's
SHA-256, plus the first 16 hex characters of the text's SHA-256. An unchanged text keeps its
filename. `refresh-index` fingerprints the document and its sidecars, so it can refresh changed
provenance or references under the same document ID.

A changed text -- a resolved marker, or an admitted correction -- is a new file. `revision_of`
names the previous file, which stays on disk. A gold span accepted against the earlier text
still validates. Re-projecting the new text again keeps that link; it does not add another
file. The output keeps every version it has written.

Projection writes a new document's sidecars before publishing the Markdown file. A retry fills
in interrupted output before ingestion can discover that document. When a printed reference
points to a revised document, its sidecar can update the target while the citing document keeps
its text and ID. Both outputs share section boundaries and the corpus records the local anchor
when a cited page continues in a second note. A split paragraph contributes its part to each
document once, with character offsets local to that document.

## Redistribution

The class is an argument. `local-only` ingests and retrieves on this host. The fine-tune
export refuses it and names the document and the class. An unknown class is refused at
projection time, before any file is written.

## Evidence

**What ran.** On 2026-11-24, on the RTX PRO 3000 Blackwell laptop host (12 GB), CPU only.
Recognition was not repeated. `book-ocr project` read the document structure and the resolved
text of the 50-page RapidOCR 3.10.0 slice of the 1,067-page target scan: PDF pages 1-40 and
290-299. The operator declared `local-only`. The section cap was the default 12,000.

**The projection.** 50 documents, 0 revisions, 212,609 characters. The shortest file is 84
characters and the longest is 12,020. The cap counts packed source text, so a rendered file
can run past it by the heading and the trailing newline; one file does, by 20 characters.
The hierarchy on this slice is 2 chapters, 6 sections, and 20 subsections. Every citation
span partitioned its document. 5,177 blocks were cited, each with a vault file and an anchor.
The structure marks 42 blocks as page numbers and 63 as running headers; none of those 105
block ids appeared. No document opened with YAML front matter, and none contained a contents,
previous, or next link, or an Obsidian callout. Every ledger block on this slice has kind
`body`, so the slice did not exercise formula or table rendering. 232 markers were written,
all `low-confidence`. The detected language was `en`. `source_system` was `book-ocr`, and
`payload_digest` was the source PDF checksum. A second projection reported 0 revisions and
left the bytes unchanged.

**Ingest.** `ingest-corpus` with the default 500-character floor admitted 44 of the 50
documents and skipped 6. It copied 44 citation sidecars. A second ingest left the document
fingerprints unchanged. Fixed chunks of 180 characters produced 1,194 chunks. Every chunk
carried a PDF page range, and those ranges cover all 50 PDF pages of the slice.

**What would overturn it.** A second projection of these ledgers whose bytes differ. A
furniture block id in a sidecar. A citation span that does not partition its document. An
ingested chunk with no page range, or a page of this slice outside every chunk range. A
`local-only` export of this projection that writes a training file.

The fixture, not this slice, pins formula LaTeX, a rendered table, an unresolved marker, the
new file after one admitted correction, the gold span that still validates on the earlier
text, the gold span that is refused on the marker, a missing licence, and the export refusal.

**Printed references.** On 2026-11-27, on the same host, CPU only. Recognition was not
repeated. `book-ocr project` read the document structure of a 25-page pilot slice of the
1,067-page scan. The operator declared `local-only`, the language was `en`, and the section
cap was the default 12,000. The slice's printed-page map is 8 observed labels, 4 gaps, 11
unnumbered pages, and 2 front-matter pages. The projection wrote 29 documents and 29
reference sidecars, with 0 revisions.

The sidecars record 534 citations: 31 resolved, 5 ambiguous, 2 mapping gaps, and 496 with no
target in this slice. By kind: figure 25, table 24, equation 5, section 26, chapter 16, and
index 438. Both gap citations name printed page 409, which the structure marks as a gap.
Every cue span and every target span is the document text at those offsets. Three documents
are index notes and hold 444 index entries. None records a parent: the recognized lines do
not indent a sub-entry. Fifteen printed pages map, on labels 36 (1), 299 (10), 333 (1), and
456 (3). Those four are among the eight observed labels. The index lines in this slice do
not name the other four (85, 137, 241, 334). 723 printed labels stay unmapped because the
slice does not observe them. A second projection wrote the same bytes.

**What would overturn the reference reading.** A cue or target span that is not the document
text at those offsets. On the synthetic book, a resolved citation whose target the vault
does not link, or a vault link the sidecar omits. A second projection of this ledger whose
bytes differ. A parent recorded on a line that is not indented, or an indented sub-entry
with no parent. A glued column left as one entry, or a line that only mentions a cell split
into cells. The fixture pins the vault agreement, the grammar swap, the indented parent
inside a glued column, the printed range, the gap, the ambiguous key, the ingest copy, the
stale removal, the stable fingerprint, and the `local-only` export refusal.
