# book-input-and-pilot-audit

## Work contract

```markdown
#### book-input-and-pilot-audit

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: the [frozen pilot and class rules](current/book-recovery/tool-selection.md#page-classes).
- User-visible outcome: a reproducible decision on whether the input, geometry and scoring chain
  can distinguish an improved reading from a damaged one before selecting engines.
- Scope boundary: exercise inventory, variants, inverse coordinates, class routing and scoring on
  rotation, clipped gutter text, neighbour bleed, faint signs and decimal points, two columns and
  ordinary controls. Check source identity and pilot coverage against the frozen selection.
- Data and artifact paths: `tests/llb/prep/book/`, the pilot selection and the audit task record.
- Execution path: use synthetic scans with known marks and cached pilot metadata inside `make ci`.
- Acceptance gates: trace a rotated corner mark from source to variant to original crop; corrupt a
  thin stroke, swap columns and change a digit independently and identify each failure in its own
  measure. Record the property, fixture, command and consequence for each selection gate.
- Documentation target: `docs/impl/current/book-recovery/quality-metrics.md`.
```

## Implementation and decisions

- `make book-ocr-pilot-select` rehashes the source, checks each published class record's PDF page
  and page id, and accepts `BOOK_PILOT_FROZEN` to compare the complete selection before writing.
  The committed freeze and the class thresholds stay as they were.
- The variant-to-render map now applies pixel-center transforms to box edges and encloses interior
  grid intersections for dewarp. The recognition cache key records the box recipe for variant
  readers. The preprocessing guard checks faint marks against local paper and samples separate
  runs of edge ink. The variant recipe invalidates old decisions.
- Glyph counts use reference order if readable blocks pair completely by kind and region.
  Character and word error and the order score measure the sequence the pipeline emitted.

## Acceptance evidence

Status: accepted.

The focused book suite passed all 239 tests. The first full CI run passed its static checks.
Six unrelated socket tests hit the restricted sandbox, two in GPU guard forkserver coverage
and four in download guard coverage. `make ci` with local socket access passed with 5,208 tests,
7 skipped, and 50 deselected. `make lint-md` and `git diff --check` passed.

The [pilot input audit](../current/book-recovery/quality-metrics.md#pilot-input-audit) is the
property-to-fixture map, with the command, observed numbers, and the consequence of each gate.
The planted 143-pixel corner mark traces through a source PDF, the rotated image, a fake OCR
stage, the merged ledger, and a pixel-exact original crop. Independent corruptions exercise
the glyph counts and the order score, with the correct reading as a control.

On 2026-11-24, the Intel Core Ultra 7 265HX host ran the supplied 1,067-page *Space Mission
Engineering: The New SMAD* scan on CPU. The source SHA-256 was
`89101813d4c695e9f992d540e751c4e2f4b80c6550f24d5d23e1a795383cb7c8`. The full inventory,
source size, and tool versions are in the [handbook replay](../current/book-recovery/quality-metrics.md#handbook-replay).
The three stages in `make book-ocr BOOK_STAGES=render,signals,classes` each finished 1,067
pages. `make book-ocr-pilot-select BOOK_PILOT_FROZEN=<frozen-pilot.json>` returned 0 and
`cmp` confirmed an exact byte match to the committed selection (SHA-256
`976203d4ba933c5eeb65c1d5ba3d351cc6855ce79a877d99780662dbd2771980`). The 25 pages represented
all ten classes, with zero empty classes and zero unfilled classes.

The variants run on those 25 pages kept 35 corrections and refused one OpenCV dewarp. The
original and rotated pixels on pages 131, 132, and 135 matched exactly after a half-turn.
The initial real run exposed false refusals on those three rotations when the source and
candidate content tests used different thresholds. Applying the same content mask on both
sides removed the false refusals and kept the clipped-edge test protective. All three pages
were reviewed at preview scale alongside their originals. Page 131 showed a neighbouring
strip that the bleed heuristic missed. The reference transcription supplies the visual check
for this case during comparison.

The original-image RapidOCR draft finished all 25 pages, with 3,962 body blocks and 25 resolved
pages. Its configuration used `render,ocr,blocks,resolve`, with `rapidocr` and `original` both in
OCR and in the resolver's primary reading. On pages 131, 132, and 135 a run reused
`render,signals,variants` and ran `ocr,blocks,resolve`, using `rapidocr` and `rotated` for both
OCR and the primary reading. All 307 normalized boxes yielded exact crops from the original
when compared with the corresponding variant crops after rotating them back. The runs provide
pilot artifacts for inspection. Accuracy depends on the open transcription gate.

## Handoff

`make plan-status` changed from 69 to 68 tasks (44 to 43 agent, 25 human in both). The CLEAR count
went from 29 to 28. The other statuses and the count of 32 tasks waiting on open Requires were
unchanged. Only the `scanned-book-recovery` group changed, and its registry status stays
`planned`. The comparison and closing audit drop this task from their Requires. The comparison
still requires `book-pilot-reference-transcription`. The next agent priority is
`book-missing-region-accounting`.
