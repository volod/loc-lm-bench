# Block ledger

The page block ledger is the only thing later recovery stages read. Each PDF page gets one file of
blocks -- a kind, a region on the rendered page, a reading-order index, and every engine's raw text
for each image variant. Nothing in the file is a chosen or corrected transcription: a disagreement
on a digit, a sign, a decimal point, or a symbol is an uncertainty span that names every reading's
slice and picks none of them.

The contract is `llb.book-ocr-block-ledger` 1.0.0
([data-prep contracts](../artifact-contracts/data-prep-contracts.md#scanned-book-run-families)).
The run that writes the file is the [book-ocr guide](../../../guides/data-prep/book-ocr.md).

## What a block is

Kinds are `body`, `heading`, `formula`, `table`, `figure`, `caption`, `footnote`,
`running-header`, `page-number`, `ornament`, `neighbour-page-bleed`, and `unknown`. A label the
ledger does not know stays `unknown`. A figure keeps its region and loses its text, including a
paragraph an engine invented to describe the picture.

The region is an axis-aligned box in pixels of `pages/<pNNNN>/render/page.png`, origin at the top
left. An engine that was asked for a different image variant still reports its boxes in that
frame: the stage maps them through the variant's stored transform before the merge
([page signals and variants](page-preprocessing.md#where-an-engine-reads)). A block without a PDF
page and a region fails validation. The block id is
`<page-id>-b<ordinal>` with the ordinal zero-padded to four digits (`p0300-b0000`). Ids are
assigned after a sort on geometry and engine, so the order the engines finished in does not change
them.

One block can carry several raw readings, keyed by engine, model, and image variant. With only
one stage's readings, a line that sits inside another engine's block shares that block and keeps
its own reading. With a layout region and a line reading both present, the line joins the region
that owns it, in the [reading order](#reading-order-and-paragraph-boundaries) below. Two boxes
from the same engine and the same stage join only when they are nearly the same rectangle, and
then in that engine's order. Neighbouring lines stay separate. Readings from different engines or
variants are aligned
character by character. A letter-only difference stays on the raw readings. A difference that
touches a digit, a sign (hyphen, plus, minus, en dash, em dash), a decimal point (a dot, or a
comma between digits), or a non-alphanumeric symbol becomes a span. Two readings that agree do
not outvote a third.

## Engines

`layout`, `ocr`, `formula`, and `table` are stages. Each one's `engine` parameter names an
adapter. `by_page_class` overrides that name for one class, and `variant_by_page_class` overrides
`variant` the same way. When the `classes` stage is configured earlier and either map is
non-empty, the engine stage reads that page's `classes.json` and routes on its primary class.
The class file is an input, so its digest is in the cache key. The digested parameters are the
engine and the variant that actually run. `ocr_language`, when the stage sets one, is part of
that key. A non-original variant also records the box-map recipe. `vram_mb` is not in the key:
it is the dry-run budget, a map of engine to megabytes, and a host that cannot hold a named
engine refuses by that engine's name before a page is read. A class change re-runs only the
stages that route on it. A run that sets `page_class` and does not configure `classes` still
routes every page on that one class. An id the class stage does not declare is not refused.

Inside a batch the stage runs pages grouped by the engine they will open, in the order that
engine is first asked for, and in page order within a group, so each engine opens at most once
per batch. The adapter is opened from the stage, and only on a cache miss. The cache key uses
the engine that actually ran. `blocks` must be configured after `render` and at least one of
those four. It reads whichever of them the run configured and writes the ledger. The class
vocabulary and the cuts are [page classes](tool-selection.md#page-classes).

An adapter that is not installed raises `EngineUnavailable` with a reason and no traceback, at
dry-run and at execution. An adapter that runs but writes nothing usable, or that silently falls
back to a weaker mode, raises `EngineRejected`. A process that crashes after writing its output
file is accepted. Heavy libraries stay in the survey's own environment or in a vLLM container;
importing the engine package does not import them. A server an engine needs is started once per
batch and stopped before the next stage.

The registered engines are the survey shortlist plus the `fake` engine CI uses:

| Engine | Stages it serves |
| --- | --- |
| `rapidocr` | ocr |
| `mineru-basic`, `mineru-standard` | layout, ocr, formula, table |
| `docling`, `docling-vlm-mineru2-pro`, `docling-vlm-surya-ocr2-5` | layout, ocr, formula, table |
| `paddleocr-vl` | layout, ocr, formula, table |
| `marker-balanced` | layout, ocr, formula, table |
| `dots-ocr` | layout, ocr, formula, table |
| `deepseek-ocr-2`, `olmocr-2` | ocr, formula, table |
| `ollama-glm-ocr-latest`, `ollama-deepseek-ocr-3b` | ocr, formula, table |
| `fake` | layout, ocr, formula, table |

Marker fast, PP-StructureV3, granite-docling, Chandra, and LightOnOCR are not registered. Qwen is
a correction model and is out of this stage. Boxes are interpreted once per page: 0..1 when every
box is in that range, thousandths of the page when every box is in 0..1000 on a larger image, and
pixels otherwise. A line in the top of a tall page is not rescaled on its own.

RapidOCR returns lines in detection order. Its adapter orders a page that has text on both sides
of the gutter as the left column, then the right, split by a line that crosses the gutter. A
layout engine's own order is kept. A line engine's reading is stored as it came, including a
misread of display lettering; correcting it is a later stage.

## Where it lives

| Piece | Path |
| --- | --- |
| Contract | `src/llb/core/contracts/data_prep/book_blocks.py` |
| Assembly and reconciliation | `src/llb/prep/book/ledger/` |
| Adapters, parsers, runners | `src/llb/prep/book/engines/` |
| Stages | `src/llb/prep/book/stages/recognize.py` (one engine stage), `stages/blocks.py` (the merge), `stages/routing.py` (the class route and the batch order) |
| Tests | `tests/llb/prep/book/test_block_ledger.py`, `tests/llb/prep/book/test_paragraph_reading.py`, `tests/llb/prep/book/test_paragraph_pipeline.py`, `tests/llb/prep/book/test_engine_adapters.py` |
| List what can run here | `make book-ocr-engines` |

`reading.json` inside an engine stage is that engine's proposals before the merge. The published
ledger is `pages/<pNNNN>/blocks.json`, copied from the committed `blocks` stage only after that
stage succeeds. The one-page PDF an engine reads is a working file and is not kept.

## How to run

```yaml
stages:
  - render
  - name: ocr
    params:
      engine: rapidocr
      page_class: formula
      by_page_class: {formula: mineru-standard}
  - blocks
```

`make book-ocr BOOK_PDF=<scan.pdf> BOOK_CONFIG=<file> BOOK_PAGES=300`. An engine that is not
installed refuses the dry run by name (`EngineUnavailable`), before any page is read. To route
on the class stage, configure `classes` before the engine stage:

```yaml
stages:
  - render
  - signals
  - classes
  - name: variants
    params:
      dewarp: uvdoc
  - name: ocr
    params:
      engine: docling
      ocr_language: eng
      vram_mb: {docling: 2970, mineru-standard: 2867}
      variant_by_page_class: {severe_curvature: dewarp}
      by_page_class: {dense_formulas: mineru-standard}
  - blocks
```

A non-original `variant_by_page_class` value needs the `variants` stage earlier in the same
list. `vram_mb` names every engine the stage may run. The numbers above are the survey peaks
in MiB, not a selected handbook config.

## Reading order and paragraph boundaries

A page that has both a structural reading (`layout`, `formula`, or `table`) and a line reading
is ordered by region, not by the overlap cluster. Each line joins the tightest region that
covers at least half of the smaller box. A body or `unknown` box that only wraps two or more
tighter regions is dropped, and those regions are the page. A line that two regions of similar
area claim about equally stays its own block. The worksheet marks that block "Placement is
unclear." and lists the rival regions. A line with no region stays at its own crop, marked "No
region owns this line.", in the column its box falls in.

A body region that spans the gutter, with at least two lines on each side that do not
themselves cross the gutter, is replaced by those lines, each in column order (left, then
right). A line that only clips a region from the other column stays out of that region. A line
with at least half its width inside the region stays there, including a table cell whose center
has crossed the gutter. A heading, caption, table, formula, figure, footnote, or piece of
furniture inside a body region stays its own block. A boxed aside
(`aside`, `sidebar`, `text-box`, `boxed`, `boxed-text`) is `unknown`. A bare integer in the top
or bottom 12 percent of the page, owned by no region, is a `page-number`.

`paragraph` is false unless the block kept a structural region. The field is optional on
contract 1.0.0 and defaults to false, so a ledger written before it still validates. The
renderer joins consecutive body lines in one column. A block with `paragraph` true joins the
next body block only when its text ends in a hyphen. That tail is taken from the other column,
or from the next page when the next page's first unit is the tail and starts in the same
column. The corpus keeps one fragment per block: the vault shows the joined word, and the
block that held the tail still cites the remainder. Furniture, including neighbouring-page
bleed and a page number, stays out of the corpus.

## Evidence

**What ran.** On 2026-11-22, on the GeForce RTX 5050 laptop (8 GB), with RapidOCR 3.10.0
(PP-OCRv6 small, ONNX Runtime, CPU) from the survey `light` environment. The GPU was idle; this
engine does not use it. The other shortlisted engines were not smoked here: their environments
are not installed, Ollama is not installed, and no vLLM image is present. On the survey host
those engines need more than this card has.

- **Synthetic page.** `render`, `ocr` with `engine: rapidocr`, and `blocks` on the survey's
  one-page synthetic scan. The ledger has 30 blocks. The planted `0.05 km/s` is present as raw
  text. Block ids are `p0001-b0000` onward. No uncertainty spans, because one engine ran.
- **Handbook page.** The same stages on PDF page 300 of the 1,067-page *Space Mission Engineering*
  scan (the page whose running header is "Design of Interplanetary Orbits"). RapidOCR returned
  109 lines and the ledger kept all 109, the header as its own block at the top, then the left
  column and then the right. Ids run `p0300-b0000` through `p0300-b0108`. The page took 4.5
  seconds, 4.4 of them in recognition.
- **Handbook.** The same stages on all 1,067 pages. Every stage finished: 3,201 page-stages, none
  failed, in 82 minutes (about 4.6 s per page, almost all of it recognition). The ledger has
  137,827 blocks, all `body` because this engine does not label a kind, mean 129 a page (5 on the
  dedication page, 842 on a parameter-table page). No uncertainty spans, because one engine ran.
  748 blocks contain a line break the detector returned inside one box. Pages the inventory
  records as a landscape image placed onto the portrait page, including PDF pages 126, 145, and
  181, rendered upright and were read as ordinary lines. Display lettering on the dedication page
  was misread and kept as the raw reading. An earlier merge, which chained any two boxes whose
  smaller one was half inside the other, glued neighbouring lines together. Same-engine boxes now
  join only when the rectangles nearly coincide. Rebuilding all 1,067 ledgers from the saved
  readings took 19 s; render and recognition were not repeated.
- **Pilot pages, layout plus lines.** On 2026-11-26, on the RTX PRO 3000 Blackwell laptop
  (12 GB), the saved Docling layout readings and RapidOCR line readings of the 25 pilot pages of
  the same 1,067-page scan were merged again. The merge read those saved readings on the CPU.
  Each rendered page is 1241 by 1755 pixels. The merge produced 885 blocks: 769 kept a structural
  region (`paragraph` true), 116 are line-only blocks placed by their boxes, 19 are tables, and 4
  are page numbers. Fourteen layout strings appear on two regions, matching two proposals from
  the layout engine. On the three-column index page, 31 line readings that sit
  inside a table and whose centers fall across the gutter stay on that table (0 line-only body
  blocks remain inside a table there). On the contents page, 9 line-only blocks sit in column
  order and the bottom folio is a `page-number`.
- **Gates that do not need a GPU.** The ledger round-trips its model and the generated JSON
  Schema. A block missing its page or its region fails both. `0.05` against `O.05` is a digit
  span naming both slices, including when two of three readings agree. A hyphen against an em
  dash is a sign span. An empty data directory refuses `rapidocr` as `EngineUnavailable` with
  no traceback. Two runs of the fake engine write byte-identical ledgers.

What would overturn the page-300 reading: a render whose long side is under 1000 pixels (the
coordinate rule would then be untested on this page), or a RapidOCR build that stops returning
quads in pixel space. What would overturn the pilot-page merge: a structural region that contains
two tighter regions and is itself one paragraph (the container rule would drop it), or a cell
whose width is less than half inside its table (it would stay a separate line). The task records
are
[0006](../../records/0006-scanned-book-recovery-book-block-ledger-and-engine-adapters.md)
and
[0021](../../records/0021-scanned-book-recovery-book-paragraph-reading-recovery.md).
