# book-pilot-pipeline-comparison

## Work contract

```markdown
#### book-pilot-pipeline-comparison

With engines behind one seam, metrics that see dangerous substitutions, a stratified
pilot set, and a verified reference, the pipeline can be chosen on evidence. The
existing Docling-plus-Tesseract lane is the baseline every candidate must beat on a
page class before it serves that class.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: RUN NEEDED -- every candidate runs over the pilot set on the CUDA host.
- Research: yes
- Requires: `book-pilot-reference-transcription`
- Dependencies: the [pilot input audit](current/book-recovery/quality-metrics.md#pilot-input-audit),
  the [quality metrics](current/book-recovery/quality-metrics.md), the
  [block ledger](current/book-recovery/block-ledger.md), the
  [page classes](current/book-recovery/tool-selection.md#page-classes), and the
  [page variants](current/book-recovery/page-preprocessing.md) a variant policy chooses among.
- User-visible outcome: a recorded comparison of each candidate pipeline -- engine
  combinations, variant policies, and per-class routing -- against the existing lane
  on every metric per page class, with worked examples of each dangerous-substitution
  class; the winner per class written as the default `book-ocr` configuration with
  its rationale, the peak VRAM it needs, and a projected full-book wall time.
- Scope boundary: in scope -- sequential candidate runs under GPU isolation; the
  existing lane run with its OCR language set to the book's recorded language, the
  setting recorded beside its numbers; per-class routing under the class stage's
  precedence; per-class margins with a page-bootstrap interval, because a class holds
  two or three pilot pages; VRAM and wall time recorded and reported, not optimized;
  the selected configuration, whose VRAM need the run's dry run checks against the
  host, and the projected wall time from the book's per-class page counts and the
  measured per-page times. Out of scope -- model correction
  (`book-evidence-constrained-correction`), and quoting selection-set numbers as an
  out-of-sample estimate (they are labelled as selection-set numbers).
- Data and artifact paths: `$DATA_DIR/book-ocr/<run_timestamp>/pilot/` per candidate;
  the selected configuration committed under `samples/book_ocr/<book-slug>/`.
- Execution path: heavy, one engine process at a time, per
  [heavy runs and evidence](../guides/development/heavy-runs-and-evidence.md);
  scoring recomputes from the saved ledgers with no GPU.
- Acceptance gates: `make ci` green; the reading states, per class, the page count,
  the winner, and its margin over the existing lane with its interval, with dangerous
  substitutions reported apart; retain the existing lane for a class when its
  comparison supplies no evidence of a gain; identify an inconclusive interval; each
  candidate's agreement with the unedited pre-draft is reported beside its score, so
  an anchoring advantage is visible; a candidate that wins no class is recorded as
  such; a dry run on a host with less VRAM than the selected configuration needs
  refuses by name; if no candidate beats the existing lane on any class, the negative
  goes to [future research](future-research.md) with what would reopen it and the
  existing lane becomes the configured engine.
- Documentation target: `docs/impl/current/book-recovery/tool-selection.md`.
```

## Implementation and decisions

Status: open. The comparison harness is in the tree (`llb book-ocr compare`,
`src/llb/prep/book/compare/`). It scores saved ledgers. It does not load a model and
does not invent an accept.

The margin is the existing lane's character error minus the candidate's. The interval
is the lowest and highest page margin, which is the page-bootstrap range. A gain
requires that whole interval above zero. A higher pooled dangerous-substitution rate
blocks a character gain. Pre-draft agreement is reported beside the score and does
not choose the winner. One dewarp method survives when two classes disagree; the
weaker class stays on the original image. An over-budget engine is not scored and
wins no class. The dry run refuses it by name. The wall-time figure is labelled
`selection-set` and counts every recognition stage that will call the engine.

The handbook reading is not done. The pilot reference is still the human gate
`book-pilot-reference-transcription`. A page left unaccepted keeps the command from
writing a winner. No config was committed under `samples/book_ocr/`. Future research
was not edited, because the accepted pages have not been scored.

## Acceptance evidence

Open. The handbook comparison has not started: the target scan is on the host and has
no accepts file.

Commands on this tree:

- The pilot tests plus `test_engine_adapters.py`, `test_book_run.py`,
  `test_quality_score.py`, and `test_page_class_stage.py`: 47 passed.
- `make lint-md` on the pages this task edited: 0 broken links, 0 run-path
  citations, 0 record-name findings.
- `make ci`: ruff, mypy, the complexity gate, the spec/plan gate, and the
  contract checks passed. Pytest reported 5229 passed, 2 skipped, 50
  deselected, and 1 failed:
  `test_a_silent_fallback_an_empty_output_and_a_hang_all_fail`. The failure is
  the sandbox refusing SIGKILL on the survey hang fixture. That test is not
  part of this change. Re-run outside the sandbox, it passed.

## Handoff

Next action: a reviewer accepts every pilot page. Then run the roster one engine at a
time and `make book-ocr-compare` against that reference. On an 8 GB card the four
vLLM page readers (Marker balanced, dots.ocr, DeepSeek-OCR-2, olmOCR-2) are over the
published survey peaks and are recorded as winning no class. They are not scored as
empty failures. The plan task stays until that accepted reading exists.
