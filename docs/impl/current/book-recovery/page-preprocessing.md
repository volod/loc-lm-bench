# Page signals and variants

Each rendered page gets a measured signal vector and only the corrections those signals call for.
The original render is always kept and never rewritten. A geometric correction stores the point
map that sends a variant pixel back to that render, and an engine stage applies the map before
its boxes reach the block ledger, so every region stays in one frame.

The contracts are `llb.book-ocr-page-signals` and `llb.book-ocr-variant-manifest`, both `1.0.0`
with no older form
([data-prep contracts](../artifact-contracts/data-prep-contracts.md#scanned-book-run-families)).
How a run commits the stages is [run ledger and resume](pipeline-and-resume.md). The shortlist
the corrections were chosen from is
[tool selection](tool-selection.md#page-preprocessing-dewarp-deskew-illumination).

## What is measured

`signals` reads `render/page.png` and writes `signals.json`. The runner publishes the same file
to `pages/<pNNNN>/signals.json`. The vector is measurements, not decisions:

| Signal | Reading |
| --- | --- |
| `orientation_deg` | Quarter turns clockwise the content is from upright: 0, 90, 180, or 270. This is the printed content, not the PDF placement rotation the inventory records |
| `skew_deg` | Residual tilt after that quarter turn, in degrees. Positive is counter-clockwise. Letter baselines decide when the page has enough long lines; a projection-profile search is used for a page of full-width lines |
| `line_curvature_px` | How far text lines bow, in pixels of the upright page |
| `background_unevenness` | Spread of the page's slow background, 0 on flat paper |
| `contrast` | Separation of ink from paper, 0 when the two meet |
| `blur` | Sharpness. Recorded so a later stage can see a soft page. Nothing here corrects it |
| `noise` | High-frequency grain. Denoising is gated on it |
| `gutter_side`, `gutter_width_px` | Which edge is darker than the page center, and how far that darkness reaches. `none` and 0 when no edge is |
| `bleed` | A neighbour-page strip as a rectangle of the original render, or null. It is not cropped |

A page of full-width lines is sideways when turning it 90 degrees multiplies the row-ink variance
by six or more. That is the synthetic case; a figure or a ruled table on a real page does not
reach it. Anything else is read as horizontal, and then as upright or upside down. A component
counts as a letter only when it is about the size of one and the ring around it is clearly
lighter than the ink, so a photograph does not vote. Letters whose centers stay within a few
pixels of one line's own center are one line -- membership is against that center, not the
previous letter, because a run of slightly lower letters would otherwise walk down the page.
Eight or more such lines and a negative median vote mean the bottoms do not share a baseline.
That correction is applied only when the scan was placed at 90 or 270 degrees, because on a
portrait page the same vote also goes negative for an upright table or a bibliography. A page
with fewer lines than that, which is how the synthetic fixtures are built, is decided by which
facing has the darker lower half of each text band.

## What is generated

`variants` reads the render and the signal vector and writes
`pages/<pNNNN>/variants/`. That directory is the stage directory: `manifest.json`, and
`<name>/page.png` for each kept correction. The original is a manifest entry with no image file
and an identity map; the pixels stay the render PNG.

A correction is generated only when its signal crosses the threshold below. A clean page keeps
the original and nothing else. There is no global binarization.

| Variant | Generated when | What it does |
| --- | --- | --- |
| `rotated` | `orientation_deg` is not 0 | The quarter turn that makes the page upright. Exact pixel rotation, not a resample |
| `deskew` | absolute `skew_deg` is at least 1.5 | That quarter turn, then the residual tilt. The canvas grows so the page still fits |
| `dewarp` | `line_curvature_px` is at least 3 | The upright, deskewed page, then an unbend |
| `illumination` | `background_unevenness` is at least 0.04 | Divide by a wide median background. Applied to the original, not to a warp |
| `contrast` | `contrast` is at most 0.25 | Local contrast (CLAHE) on the original |
| `gutter` | a dark edge at least 12 px wide and no more than 12 percent of that side | Lift only that edge. A black panel or a figure that darkens a third of the page is left alone |
| `denoise` | `noise` is at least 8 | A small bilateral filter on the original. A median blur is not used: it erases a one-pixel minus |

`deskew` is one composed affine back to the original, not a chain a later reader has to replay.
`dewarp` stores a grid, sampled every 16 pixels with the last row and column kept, of the dense
map that built the image. The OpenCV unbend moves a pixel by the bow of the text line there, the
quadratic fit minus the straight fit, and by nothing outside that line's span. A line bowed by
more than 40 pixels, or more than three times the page's typical line, is left out of the remap:
that fit is a figure or a column edge. Photometric variants store the identity affine. Every map
sends a variant pixel to an original pixel.

The `dewarp` parameter chooses the unbend. The default `opencv` is the deterministic remap above
and the one `make ci` runs. `page-dewarp` and `uvdoc` run in the survey environments
(`book-light` and `book-paddle`); a result with no point map is refused, because a geometric
variant that does not cover the page is a crop, and the stage still succeeds so the refusal is
cached. `auto` tries `page-dewarp` and falls back to `opencv`, recording which one ran. An
explicit `page-dewarp` or `uvdoc` that fails is refused, not swapped.

## What is refused

Two guards run before a variant is kept. Either one records the variant as `refused` with a
reason and writes no image.

The stroke guard looks for thin strokes and small dots at least 24 grey levels darker than the
local median background, including faint minuses and decimal points. It samples an ink pixel
from each selected component and ignores dot-sized specks on dark surfaces. The variant must
keep at least 70 percent of the sampled marks. A photometric variant checks each mark at the
same pixel. A geometric variant allows a three-pixel neighbourhood for interpolation.

The content guard runs on `rotated`, `deskew`, and `dewarp`. A text line is sampled at a real ink
pixel, not at the middle of a full-width band, so a two-column page whose gutter is empty does
not look like a lost line. Each run of ink along the four outer strips gets its own sample,
so retaining one edge fragment cannot stand in for a separate clipped gutter line. A map that
does not cover the page is the same refusal.

## Where an engine reads

`layout`, `ocr`, `formula`, and `table` take a `variant` parameter. The default is `original`,
which reads the render. Any other name reads that variant's PNG, and only when `variants` is
configured earlier in the same run; a dry run rejects an unknown name and a missing stage. At
execution, a name that was not kept for that page raises `VariantUnavailable`. The stage fails.
There is no silent fallback to the original.

Each proposed box is mapped through the stored transform before the reading is written, so
`blocks` merges rectangles that already share the render's pixel frame. The mapper converts from
box edges to the stored pixel-center frame and back. A quarter-turn box therefore recovers the
same original pixels. A grid-based box also encloses mapped grid intersections within the box,
which can extend beyond the mapped outer corners. A block read on a rotated and deskewed variant
of a synthetic page lands within 4 pixels of the same block on the original. See the
[pilot input audit](quality-metrics.md#pilot-input-audit) for the exact quarter-turn crop test.

## Where it lives

| Piece | Path |
| --- | --- |
| Contracts | `src/llb/core/contracts/data_prep/book_preprocess.py` |
| Measurement, corrections, guards, maps | `src/llb/prep/book/preprocess/` |
| Stages | `src/llb/prep/book/stages/preprocess.py` |
| Engine image and the map back | `src/llb/prep/book/preprocess/frames.py`, called from `stages/recognize.py` |
| Tests | `tests/llb/prep/book/test_page_signals.py`, `test_page_variants.py`, over pages from `_page_factory.py` |

OpenCV is imported when a stage runs, not when the registry is built. The stage's tool identity
is the OpenCV version. The decision thresholds are part of the cache key, so changing one re-runs
the stage. PNG bytes use a fixed compression and no ancillary chunks; the JSON is written with
sorted keys. Two OpenCV runs of the same page are byte-identical. A GPU dewarp is not recomputed
to prove that: the stage cache returns the refused or kept record as it was written.

Run both stages with the existing command. There is no separate target:

```sh
make book-ocr BOOK_PDF=<scan.pdf> BOOK_STAGES=render,signals,variants
```

## Evidence

**What ran.** On 2026-11-22, on the GeForce RTX 5050 laptop (8 GB), CPU only: OpenCV 5.1.0, no model
loaded and no GPU memory taken for this work. The source is the 1,067-page handbook scan (163.7 MB;
every page an image, no embedded text). Stages were `render`, `signals`, and `variants`, in batches
of 25, with the default OpenCV unbend.

The renders were already committed. Remeasuring skew from letter baselines re-ran the signal stage
on all 1,067 pages and rebuilt the variants that the new vector called for: 2,078 stages ran, 1,123
were taken from cache, none failed, in about 12 minutes. Tightening the unbend so it moves a pixel
by the bow only, and not by a line's slope or a fit that runs off the page, then rebuilt variants
alone: 1,067 variant stages ran, the 2,134 render and signal stages came from cache, none failed,
in about 3.5 minutes. The counts below are that finished state.

- **Orientation.** 1,049 pages are upright. PDF pages 126-143, eighteen pages, are upside down, and
  each kept a `rotated` variant. They are the landscape-placed scans whose content is inverted; the
  other landscape placements stay at 0. Page 126's render reads upside down, and the rotated image
  reads upright ("6.1 The Requirements Definition Process"). A portrait page was not turned.
- **Skew.** Letter baselines, where the page has enough long lines. The largest absolute tilt is
  0.7 degrees and the median is 0.1, so nothing reaches the 1.5 degree deskew gate and no deskew
  variant was written. A level page is left level. The projection search still measures a synthetic
  page of full-width lines (a planted 2 degrees reads -2.0) and is not what decided these pages.
- **Curvature.** 26 pages bow by at least 3 pixels (3.00 to 5.51; the largest is PDF page 765).
  24 kept an OpenCV unbend. On those pages the vertical move, once a quarter-turn is set aside, is
  3.4 to 9.5 pixels: the bow, not a slope and not a spike at the margin. Two pages, 611 at 3.0
  pixels and 648 at 3.1, were refused because the unbend lost a text line or a page-edge line.
- **Illumination, contrast, noise, gutter.** 1,066 pages are uneven enough to try illumination
  (0.04 or more). 951 were kept and 115 were refused by the stroke guard. The cover, unevenness
  0.348, was refused for erasing 12 of 13 thin strokes in the artwork. Only PDF page 2 is faint
  enough for local contrast (0.078), and that one variant was kept. Seven pages measure noise of
  8.00 to 8.63 and were denoised; the cover is 1.8 and a text page such as page 11 is 6.7, under
  the gate. A dark edge is measured on 1,044 pages, and 54 of them are at least 12 pixels and at
  most 12 percent of that side (all on the left, 12 to 142 pixels wide). Those 54 were lifted. The
  cover's 554-pixel black panel, 45 percent of the width, was not.
- **Bleed and blur.** 32 pages record a neighbour-page band, and none of it is cropped. Blur is
  recorded and not corrected: 163 to 2,560, median 1,309.
- **The original.** All 1,067 pages keep an original entry with no image of its own. Rebuilding
  signals and variants did not run the render stage again.
- **UVDoc.** On the same host and date, UVDoc through paddleocr 3.7.0 on the Transformers engine
  (torch 2.15.0) was run on PDF page 765, the page with the largest measured bow. The model
  returned the unwarped image and no point map, so the variant was refused and no image was kept.
  Resuming that one-page run took the render, the signals, and the refusal from the stage cache
  and did not run the model again. The GPU was idle afterwards, at the same 41 MiB it held before
  the smoke.

**What would overturn it.** A portrait page, or any page outside 126-143, reported upside down, or
126-143 reported upright. A deskew variant on a page whose letter baselines are under 1.5 degrees.
A kept unbend whose vertical move is tens of pixels on a page bowed by only a few, or a kept unbend
that drops an edge line. Illumination or denoising kept where it erases a real minus or decimal, or
a black panel treated as a gutter. A render stage running again when only the signal or variant
recipe changed. A UVDoc result that carries a point map covering the page and is still refused, or
a resume that runs the model again under the same key.
