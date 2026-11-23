# book-page-signals-and-variants

## Work contract

```markdown
#### book-page-signals-and-variants

One preprocessing recipe either under-treats the curved, shadowed pages or destroys the thin strokes
on the clean ones. Each page needs measured signals and only the corrections those signals call for,
with the original rendering always kept beside them. A geometric variant also moves every pixel:
the ledger records a block's region in the original render's frame, while today an engine stage
reads only that render and `variant` is a label, so a box read on a rotated or dewarped image has
no way back to the page it came from.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: the page inventory records each image's placement rotation, which does not give
  content orientation ([run ledger and resume](current/book-recovery/pipeline-and-resume.md#evidence));
  the shortlisted dewarp and illumination methods and the content each was seen to lose are in
  [tool selection](current/book-recovery/tool-selection.md#page-preprocessing-dewarp-deskew-illumination);
  the engine stages and their `variant` parameter are the [block ledger](current/book-recovery/block-ledger.md#engines)
  seam.
- User-visible outcome: per page, a recorded signal vector -- skew, line curvature, background
  unevenness, contrast, blur, gutter-shadow side and width, neighbour-page bleed band, orientation --
  and the set of image variants generated for it with their parameters, the original among them;
  an engine stage reads any named variant, and every block it proposes lands in the ledger in the
  original render's pixel frame.
- Scope boundary: in scope -- signal measurement; deskew, dewarp (the survey's shortlisted method),
  illumination and background normalization, local contrast, gutter-shadow correction, gentle
  denoising, and rotation, each applied only where its signal calls for it; the bleed band recorded
  as a region rather than blindly cropped; a stroke-preservation guard that refuses a variant
  erasing thin strokes measured on the original (minus signs, decimal points, rules), and a content
  guard that refuses a geometric variant losing a text line or the page edge (the survey saw
  dewarping cut both); no global binarization by default; for every geometric variant (rotation,
  deskew, dewarp) its recorded transform with a point map back to the original render, a variant
  whose map does not cover the page refused; the engine stages reading the image of the variant
  their `variant` parameter names, with each proposed box mapped back before the ledger merge, so
  regions, crops, and cross-variant reconciliation share one frame. Out of scope -- OCR, choosing the
  winning variant (reconciliation and the pilot decide that), and super-resolution unless the survey
  shortlists it as one more variant.
- Data and artifact paths: `src/llb/prep/book/preprocess/` and `src/llb/prep/book/stages/`;
  `pages/<pNNNN>/signals.json` and `pages/<pNNNN>/variants/` (each geometric variant with its
  transform) inside the run directory.
- Execution path: the OpenCV path runs inside `make ci` over synthetic pages with planted
  curvature, shadow, blur, bleed, and rotation; the GPU dewarp model is smoke-tested on the host.
- Acceptance gates: `make ci` green; each planted defect is detected and a clean synthetic page
  triggers no correction; a variant that erases a planted decimal point or minus sign is refused by
  the stroke guard, and a dewarp that drops a planted edge line by the content guard; a synthetic
  page placed upside down is detected and read upright; a block read on a rotated and deskewed
  variant lands within a stated pixel tolerance of the same block read on the original; the OpenCV
  variants are byte-identical on rerun (a GPU variant is reproduced through the stage cache, not
  recomputation); the original rendering is never altered.
- Documentation target: `docs/impl/current/book-recovery/page-preprocessing.md`.
```

## Implementation and decisions

The contracts are `llb.book-ocr-page-signals` and `llb.book-ocr-variant-manifest`, both `1.0.0`
with no older form. `signals` publishes `signals.json` up from its stage directory. `variants`
writes into `pages/<pNNNN>/variants/`, which is the stage directory: a manifest and one PNG per
kept correction. The original is a manifest entry with no image file. The render PNG is never
copied or rewritten.

Orientation is the printed content, not the PDF placement rotation. A page is sideways only when
turning it 90 degrees multiplies the row-ink variance by six or more. With eight or more letter
lines, band polarity does not vote, and a negative baseline vote turns the page 180 only when the
scan was placed at 90 or 270 degrees. A page with fewer lines than that, which is how the
synthetic fixtures are built, is decided by which facing has the darker lower half of each text
band. A component counts as a letter only when it is about one letter in size and the ring around
it is paper, so a photograph does not vote.

Skew is the median tilt of those letter lines when at least eight of them span 80 pixels or more.
Otherwise a projection-profile search runs, which is what a synthetic page of full-width lines
hits. Deskew is one affine back to the original: the quarter turn, then the residual, with the
canvas grown so the page still fits.

The OpenCV unbend moves a pixel by the bow of the text line there, the quadratic fit minus the
straight fit, and by nothing outside that line's span. A line bowed by more than 40 pixels, or
more than three times the page's typical line, is left out. An earlier remap also removed the
line's slope and pinned only the two edge pixels, which moved a margin column by about 100 pixels
on a page bowed by a few; the handbook variants were rebuilt without it. The default method is
`opencv`, so `make ci` stays deterministic. `page-dewarp` and `uvdoc` run in the survey
environments. A result with no point map is a cached refusal, not a stage failure, and an explicit
failure is not swapped for the other method.

The stroke guard requires 70 percent of the thin ink strokes and the paper-ring dots, measured on
the original, to survive. Denoising is a small bilateral filter. A median blur is not used: a
3-wide one erases a one-pixel minus. There is no global threshold. A gutter is lifted only when
the dark edge is at least 12 pixels and at most 12 percent of that side.

`layout`, `ocr`, `formula`, and `table` read the variant named by their `variant` parameter. The
default is `original`. A name that was not kept raises `VariantUnavailable` and the page stage
fails. Each proposed box is mapped through the stored transform before the reading is written.

Code and the handbook reading:
[page signals and variants](../current/book-recovery/page-preprocessing.md).

## Acceptance evidence

Status: accepted.

- **`make ci` green.** On 2026-11-22, with `UV_LINK_MODE` and `UV_CACHE_DIR` unset: 5,055 passed,
  2 skipped, 50 deselected. An earlier run in the same session, with `UV_LINK_MODE=copy` exported,
  failed two pre-existing venv tests because the setup script never called the fake `uv`, so its
log file was never created. Those two pass with the variable unset. The same suite on the tree
after the task left the plan was green with the same counts. `make lint-md` was green and
`make lint-spec-plan` reported 0 findings.
- **Planted defects, a clean page, the stroke guard, the content guard, upside-down detection,
  the 4-pixel map, and byte-identical OpenCV output.** `tests/llb/prep/book/test_page_signals.py`
  and `test_page_variants.py`. A planted 2 degree tilt reads -2.0. A planted bow is unbent without
  moving the margin, and the curvature falls. An erased minus and decimal are refused. A dewarp
  that drops an edge line is refused. An upside-down skewed block maps back within 4 pixels.
- **The handbook scan.** On 2026-11-22, GeForce RTX 5050 laptop (8 GB), OpenCV 5.1.0, CPU only,
  all 1,067 pages. Detail and every count:
  [page signals and variants](../current/book-recovery/page-preprocessing.md#evidence). The render
  was not rewritten when signals and variants were rebuilt.
- **UVDoc on this host.** paddleocr 3.7.0, Transformers engine, torch 2.15.0, PDF page 765. The
  model returned an image and no point map; the variant was refused and no image was kept.
  Resuming the run cached all three stages and did not run the model again. The GPU was idle
  afterwards (41 MiB, 0 percent).

## Handoff

`make plan-status` immediately before the task left the plan: 75 tasks (agent 50, human 25).
After: 74 (agent 49, human 25). `scanned-book-recovery` lost this task and gained none. The
priority agent task is `book-recovery-quality-metrics`. UVDoc's Transformers result carries no
point map, so that method stays a cached refusal; recovering a map was not added as a task.
