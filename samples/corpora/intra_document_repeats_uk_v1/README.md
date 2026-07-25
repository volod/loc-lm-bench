# intra_document_repeats_uk_v1 -- planted intra-document repeat fixture

A converted-PDF-shaped Ukrainian manual whose boilerplate repeats INSIDE the one document, plus a
short second document that shares one block with it. This is the shape that
[duplicate chunk collapse](../../../docs/impl/current/rag-core.md#duplicate-chunk-collapse) hides
but cannot fix at the source, and that the conversion-side handling in `llb.prep.pdf.repeats`
acts on. The sibling `duplicate_chunks_uk_v1/` fixture plants the opposite shape (the same page
furniture repeated ACROSS documents).

| document | unique content | repeats |
| --- | --- | --- |
| `nastanova-oblik.md` | three sections (оприбуткування / переміщення / списання) | the save-procedure block 3x, the support block 3x, the table header 2x |
| `dovidka-oblik.md` | one summary section | the support block (its only copy) |

What the fixture plants, and what `tests/llb/prep/test_pdf_repeats.py` asserts against it:

- Block census (`min_repeats=3`): 2 repeated groups, both inside `nastanova-oblik.md`, largest
  group 3 copies, over 18 blocks. Both are prose and therefore eligible; the repeated table
  header is NOT, because dropping it from every copy but the first would corrupt the tables.
- `drop` removes the 4 later copies (2 groups x 2), taking the corpus from 1957 to 1440 chars.
- `anchor` keeps all 6 copies and prefixes each with its section breadcrumb (1957 -> 2137 chars),
  so no two copies are byte-identical any more.
- Chunk census at `sentence@200/30`: 3 duplicate groups over 15 chunks -- 2 intra-document and 1
  cross-document (the support block also appears in `dovidka-oblik.md`), which is the split
  `duplicate_stats` reports and `compare-retrieval` prints per lane.

The numbers above are the assertion, not documentation of an incidental fact: changing these
files changes what the repeat tests measure.
