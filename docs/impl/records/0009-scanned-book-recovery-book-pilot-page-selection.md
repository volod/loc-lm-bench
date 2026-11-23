# book-pilot-page-selection

## Work contract

```markdown
#### book-pilot-page-selection

Hand-picked pilot pages bias the pilot toward what someone happened to remember. Select them from
measured page signals and cheap content signals, stratified by the page classes the pilot must
cover, with ordinary pages kept as a control. The same classes are what a full run routes on, and
nothing yet assigns one: `page_class` is a single parameter for the whole run, so `by_page_class`
cannot fire on a real book, and the engine lease holds one engine at a time, so a batch whose pages
alternate between two routed engines would restart a server -- minutes for a vLLM model -- at every
switch.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- the selection over the target book is a CPU-only signal pass.
- Dependencies: the per-page signal vector is
  [page signals and variants](current/book-recovery/page-preprocessing.md); the target scan is
  available locally under `DATA_DIR`; per-class engine selection and the batch-scoped server are
  the [block ledger](current/book-recovery/block-ledger.md#engines) seam.
- User-visible outcome: `llb book-ocr pilot-select <run-dir> --size <n>` writes a reproducible pilot
  set of 20-30 PDF pages, each with the class it represents and the signal values that placed it;
  every page of a run carries its class set as a stage output, and an engine stage's
  `by_page_class` routes on it.
- Scope boundary: in scope -- stratified, seeded selection over the declared classes (severe
  curvature, gutter and corner text, grey or shadowed background, dense formulas, large or irregular
  tables, multi-column layout, labelled diagrams, small footnotes, landscape, normal); cheap content
  signals (formula density, ruling lines, figure area, column count, footnote band); a coverage
  report naming any class with no qualifying page; a class stage over every page with a declared
  precedence for a page in several classes; engine stages reading the class from that stage, so it
  enters their cache key through the input digest; a routed stage running each batch's pages grouped
  by selected engine, so each engine opens at most once per batch. Out of scope -- transcription,
  the comparison, and which engine serves a class (the pilot decides it).
- Data and artifact paths: the target book's pilot list committed at
  `samples/book_ocr/<book-slug>/pilot_pages.json` (source checksum, PDF page numbers, classes,
  signals -- no page image and no recovered text); the per-page class and the selection outputs
  inside the run directory.
- Execution path: CPU-only fixture tests inside `make ci`; one signal pass over the target scan;
  `make book-ocr-pilot-select` (with a `##` help description) wraps the command.
- Acceptance gates: `make ci` green; the selection is deterministic under its seed; every declared
  class is covered or reported empty by name; the committed list for the target book carries the
  source checksum it was selected from; a page whose class changes re-runs only the stages its class
  routes; a fixture batch of alternating classes opens each routed engine once.
- Documentation target: `docs/impl/current/book-recovery/tool-selection.md` and the routing section
  of [block ledger](current/book-recovery/block-ledger.md#engines).
```

## Implementation and decisions

The `classes` stage writes `llb.book-ocr-page-classes` 1.0.0 for every page. `pilot-select`
writes `llb.book-ocr-pilot-selection` 1.0.0 under the run's `pilot/` and, with `--out`, to the
committed handbook list. When `by_page_class` is set and `classes` is configured earlier, the
class file is an input of the engine stage, so a class change misses only the stages that route
on it. The digested parameters stay `engine` and `variant`. A run-wide `page_class` with no
`classes` stage still routes every page, and an id the class stage does not declare is not
refused. A batch is grouped by the engine each page will open, so each engine opens at most once.
The adapter opens from the stage, and only on a cache miss.

Two cuts were tightened after the first pass over the handbook. An index line's gap is full of
faint dots, so it is not an equation number (`formula_leader_gap` 0.05); without that, the index
outranked the display equations. A footnote rule has to sit in the lower part of the bottom
fifth (band 0.04 to 0.15), and selection prefers the shorter band, because a rule at the top of
that fifth is a separator. The table class stays a run of ten horizontal rules plus four
vertical rules. Counting vertical rules alone would also mark a page of graphs as a table.

`BlocksStage` lives in `stages/blocks.py` and the route in `stages/routing.py`, so
`recognize.py` stays under the soft line limit. The cuts, the precedence, the handbook counts,
and what would overturn them are in
[page classes](../current/book-recovery/tool-selection.md#page-classes).

## Acceptance evidence

Status: accepted.

- **`make ci` green.** On 2026-11-22: 5,099 passed, 2 skipped, 50 deselected. `make lint-md` was
  green and `make lint-spec-plan` reported 0 findings. No model was loaded. The class pass did
  not use the GPU.
- **Selection is deterministic under its seed, and every class is covered by name.** The
  handbook list at `samples/book_ocr/space-mission-engineering-the-new-smad/pilot_pages.json`
  carries SHA-256 `89101813d4c695e9f992d540e751c4e2f4b80c6550f24d5d23e1a795383cb7c8`, 25 pages,
  seed 0, and a non-zero coverage for every declared class. `empty_classes` and
  `unfilled_classes` are empty. The file has no page image and no recovered text.
  `tests/llb/prep/book/test_page_classes.py` pins the seed and the empty-class name.
  `tests/llb/prep/book/test_page_class_stage.py` reads the committed list.
- **A class change re-runs only the routed stage, and a mixed batch opens each engine once.**
  `test_a_changed_class_reruns_only_the_stage_that_routes_on_it` and
  `test_alternating_classes_open_each_routed_engine_once` in
  `tests/llb/prep/book/test_page_class_stage.py`. The older
  `test_page_class_selects_the_configured_engine` in
  `tests/llb/prep/book/test_engine_adapters.py` still routes on a run-wide `page_class` with no
  `classes` stage.
- **The handbook pass.** On 2026-11-22, CPU only, on the GeForce RTX 5050 laptop (8 GB). The
  counts and the pages checked against the scan are in
  [page classes](../current/book-recovery/tool-selection.md#page-classes).

## Handoff

`make plan-status` immediately before the task left the plan: 73 tasks (agent 48, human 25).
After: 72 (agent 47, human 25). `scanned-book-recovery` lost this task and gained none. The
priority agent task is `book-reading-resolution-and-correction-overlay`. Dropping this task's
`Requires` unblocked `book-pilot-reference-transcription`, which is now the priority human task.
The comparison still waits on that transcription. Which engine serves a class stays
`book-pilot-pipeline-comparison`.
