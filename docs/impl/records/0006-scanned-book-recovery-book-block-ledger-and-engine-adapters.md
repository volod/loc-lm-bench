# book-block-ledger-and-engine-adapters

## Work contract

```markdown
#### book-block-ledger-and-engine-adapters

Every later stage -- correction, structure, rendering, projection -- reads one thing: the block
ledger. It has to exist under a registered contract before any engine is wired, and engines have to
plug into it behind one adapter seam, so the pilot can swap them and a run records exactly which
engine and variant produced each reading.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: engines register as stages behind the `book-ocr` stage seam, whose committed outputs
  land in `pages/<pNNNN>/<stage>/` ([run ledger and resume](current/book-recovery/pipeline-and-resume.md));
  the shortlist per stage, its isolated environments, and the output each adapter must drop or
  normalise are in [tool selection](current/book-recovery/tool-selection.md).
- User-visible outcome: each page's blocks -- kind, region, reading-order index, raw text per
  (engine, image variant), confidence, engine and model identity -- persist per page and re-read
  without re-running an engine; the shortlisted engines are selectable by configuration per stage
  and per page class.
- Scope boundary: in scope -- the ledger contract with its block kinds (body, heading, formula,
  table, figure, caption, footnote, running header, page number, ornament, neighbour-page bleed,
  unknown); block ids derived from PDF page and in-page ordinal; an adapter protocol for layout, OCR,
  formula, and table engines with lazy imports and a named refusal when an engine is absent;
  adapters for the survey's shortlist; multi-variant reconciliation by character alignment, where a
  disagreement on a digit, sign, decimal point, or symbol is recorded as an uncertainty span and
  never settled by majority alone; an injected fake engine for CI. Out of scope -- choosing engines,
  model correction, structure, and rendering.
- Data and artifact paths: `src/llb/prep/book/ledger/`, `src/llb/prep/book/engines/`;
  `pages/<pNNNN>/blocks.json` inside the run directory.
- Execution path: fixture tests inside `make ci`; each real adapter smoke-tested on the CUDA host
  against the synthetic scan.
- Acceptance gates: `make ci` green; the ledger round-trips through its model and generated JSON
  Schema; block ids are stable across reruns and independent of engine timing; a block without a
  page and region fails validation; a fixture where two variants read `0.05` and `O.05`, or `-` and
  an em dash, yields an uncertainty span rather than a choice; an absent engine yields a named
  refusal, not a traceback.
- Documentation target: `docs/impl/current/book-recovery/block-ledger.md`.
```

## Implementation and decisions

The page ledger is `llb.book-ocr-block-ledger` 1.0.0. A block has a kind, a pixel region on the
rendered page, a reading-order index, and one raw reading per engine, model, and image variant.
There is no chosen-text field. Ids are `<page-id>-b<ordinal>` after a sort on geometry and engine,
so engine timing cannot change them.

Reconciliation aligns readings character by character. A difference that touches a digit, a sign,
a decimal point, or a symbol becomes an uncertainty span naming every slice. Agreement among some
readings does not outvote another. Letter-only differences stay on the raw readings.

Engines are `book-ocr` stages (`layout`, `ocr`, `formula`, `table`) selected by `engine` and, per
page class, `by_page_class`. `blocks` merges whichever of those the run configured. An absent
engine is `EngineUnavailable` with a reason and no traceback; a run that produced no usable output,
or a silent fallback, is `EngineRejected`. Heavy libraries stay in the survey environment or a
vLLM container. A server is opened once per batch. Figure descriptions are dropped. Boxes are
classified once per page as unit, thousandths, or pixels, so one line near the top of a tall page
is not rescaled by itself. RapidOCR, which has no reading order, is ordered left column then right
on a two-column page. Process paths are absolute because the stage runs in its own directory.

The registered adapters are the survey shortlist plus `fake`. Marker fast, PP-StructureV3,
granite-docling, Chandra, LightOnOCR, and Qwen are not registered.

Code: `src/llb/core/contracts/data_prep/book_blocks.py`, `src/llb/prep/book/ledger/`,
`src/llb/prep/book/engines/`, `src/llb/prep/book/stages/recognize.py`, `llb book-ocr engines`.
Detail and the measured pages: [block ledger](../current/book-recovery/block-ledger.md).

## Acceptance evidence

Status: accepted.

- **`make ci` green.** On 2026-11-22 the suite was 5,040 passed, 2 skipped, 50 deselected. A first
  run inside the sandbox failed one pre-existing survey hang test (`PermissionError` on `SIGKILL`)
  and stopped mypy on `src/llb/prep/book/survey/runners/opencv_preprocess.py`, where `imread` may
  return None. The runner now refuses an unreadable image before the blur. The re-run, able to
  signal a child process, was green. `make lint-md` was green on the docs; it is re-run after the
  task leaves the plan, with `make lint-spec-plan`.
- **The ledger round-trips, and a block without a page and region fails.**
  `tests/llb/prep/book/test_block_ledger.py` validates every kind through the model, the generated
  JSON Schema, and `DEFAULT_REGISTRY.read_current`. A block missing `pdf_page` or `region` fails
  both pydantic and jsonschema. `Block` has no chosen-text field.
- **Ids are stable and independent of engine timing.** The same proposals in two orders get
  `p0007-b0000` onward. Two fake-engine runs write byte-identical `blocks.json`. A patched
  `time.time` does not change the file.
- **`0.05` against `O.05`, and a hyphen against an em dash, are spans, not a choice.** Two of
  three readings agreeing on `0.05` still leaves both slices in the span. A letter-only difference
  (`cat` / `cot`) records no span.
- **An absent engine is a named refusal.** A dry run with `rapidocr` and an empty data directory
  raises `BookRunError` containing `EngineUnavailable` and `rapidocr`, and no traceback.
- **One real adapter smoked on this host, then the handbook.** On 2026-11-22, GeForce RTX 5050
  laptop (8 GB), RapidOCR 3.10.0 from the survey `light` environment, CPU only. The synthetic page
  through `render`, `ocr` (`rapidocr`), and `blocks` produced 30 blocks, including the planted
  `0.05 km/s`. All 1,067 pages of the handbook scan then ran the same stages: 3,201 page-stages,
  none failed, in 82 minutes. The ledger has 137,827 blocks, all `body`, no uncertainty spans.
  PDF page 300 is 109 blocks, one per recognition line. The first merge chained neighbouring
  lines; same-engine boxes now join only when the rectangles nearly coincide, and rebuilding the
  ledgers from the saved readings took 19 s. The other shortlisted engines were not smoked: their
  environments are not installed, Ollama is not installed, and no vLLM image is present. On the
  survey host those models need more than 8 GB. Detail:
  [block ledger](../current/book-recovery/block-ledger.md).

## Handoff

`make plan-status` immediately before the task left the plan: 60 tasks (agent 35, human 25).
After: 59 (agent 34, human 25). `scanned-book-recovery` lost this task and gained none. `make ci`
on the final tree: 5,042 passed, 2 skipped, 50 deselected. `make lint-md` green.
