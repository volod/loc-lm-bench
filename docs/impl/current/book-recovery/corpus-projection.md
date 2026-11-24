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

The output refuses to replace the run directory, and it refuses a directory that overlaps the
source PDF. A second projection of an unchanged ledger writes the same bytes.

## Where it lives

| Module | Role |
| --- | --- |
| `src/llb/prep/book/project/text.py` | Content text: formulas as LaTeX, tables as Markdown, markers as tokens |
| `src/llb/prep/book/project/spans.py` | Page spans that partition the document, and the block citation on each span |
| `src/llb/prep/book/project/identity.py` | The document id, and `revision_of` when the text changes |
| `src/llb/prep/book/project/write.py` | The Markdown file, the citation sidecar, and the metadata sidecar |
| `src/llb/prep/book/project/build.py` | `project_corpus` |
| `src/llb/cli/book/project.py` | `llb book-ocr project` |
| `src/llb/goldset/markers.py` | The refusal of a gold span that overlaps an unresolved marker |
| `make/data-prep/book.mk` | `make book-ocr-project` |

Tests: `tests/llb/prep/book/test_project.py`, over a synthetic three-page book. No page of a
real book enters git. Ingest copies a sibling `*.citations.json` when a text document has one
(`src/llb/prep/corpus/ingest_text.py`), and a stale document takes its sidecar with it.

## What a document contains

One content file of the vault becomes one corpus document. Furniture stays in the ledger and
out of the text: running headers, page numbers, ornaments, and neighbour-page bleed. The
generated contents, the previous and next links, and the YAML front matter are not written.
A formula is display LaTeX, with the inventory's equation number as a `\tag` when the text
has none. A table is Markdown. A figure keeps the caption the page printed.

An unresolved span, an empty block, a failed page, and a low-confidence or orphaned reading
are a machine token, not a callout:

```text
{{llb-marker:<reason>:<block-id>:<start>-<end>}}
```

`unresolved` and `empty` are the reasons `validate-goldset` refuses. A span that overlaps
one of those tokens is not evidence. `low-confidence` and `orphaned` stay visible and do not
refuse a span that covers only the adopted text.

The citation sidecar is the existing `llb.pdf-citations` contract, parser `book-ocr`. Its
page spans partition the document, including the gap before a block and the trailing newline.
Each block records its id, its region, the character range of its text, the vault file, the
vault anchor, and any marker's ledger offsets and document offsets. A block with no numbered
anchor cites the page anchor. The offsets are indexes into the document text.

The metadata sidecar is `<document>.md.metadata.json`. `source_system` is `book-ocr`, so the
document takes the acquired lane. `payload_digest` is the source PDF checksum. `language` is
the operator's value or the detected language. `licence` is the class the operator declared.
`book_logical_key` is the part's identity and is ignored by ingest.

## Identity and revisions

The logical key is the source checksum, the structure node, and the part index of that node
inside the layout. The document id is `book-` plus the first 16 hex characters of the key's
SHA-256, plus the first 16 hex characters of the text's SHA-256. An unchanged text keeps its
filename. `refresh-index` then sees the same id and the same fingerprint.

A changed text -- a resolved marker, or an admitted correction -- is a new file. `revision_of`
names the previous file, which stays on disk. A gold span accepted against the earlier text
still validates. Re-projecting the new text again keeps that link; it does not add another
file. The output keeps every version it has written.

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
