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

## Scope change

The plan item was rewritten before this pass. The paste above is the earlier ask: it
required a per-class winner. The contract this pass closed does not select a handbook
profile. It freezes the scorer, rebuilds the ledgers so a repeated stage is not a
repeated paragraph, and reads MinerU from middle JSON so the boxes are the printed
regions. Selection and correction adoption wait for `book-pilot-reference-transcription`.
`book-technical-content-audit` was still open when this pass started; the user named
this task, and the audit was not implemented here.

## Implementation and decisions

Status: accepted. The comparison harness stays `llb book-ocr compare`
(`src/llb/prep/book/compare/`). It scores saved ledgers, refuses without an accepts
file, and does not invent one. The margin, the page-bootstrap interval, the
dangerous-substitution veto, pre-draft agreement, the single dewarp method, the
over-budget refusal, and the `selection-set` projection are unchanged. A whole-page
engine is now called once. Later recognition stages copy that reading. The projection
multiplies the mean stage duration by the stage count, which is the time spent reading
when the copies are near zero.

The ledger collapses same-engine boxes whose intersection over union is at least 0.85.
Identical text is kept once. A line inside a larger region stays a line. Formula and
table fields are filled from text when the reading file omitted them, including a
spaced `\tag {n}` and a Markdown header whose corner labels repeat. A trailing pipe
that is not a fence keeps an empty cell. MinerU's recovery command writes middle JSON.
Display math fenced by a `$$` pair is one formula.

The 2026-11-25 roster on the RTX 5050 (8,151 MB) still accounts for every frozen page:
a reading, or a named failure where a variant was not generated. The numbers, the
2026-11-26 MinerU reading, and the page checks are in
[tool selection](../current/book-recovery/tool-selection.md#handbook-pilot-reading).
No config was committed under `samples/book_ocr/`. Future research was not edited.
The other 1,042 pages of the scan were not read.

## Acceptance evidence

Accepted on the frozen scorer and the recovered pilot reading. Handbook selection
waits for the release reference. On 2026-11-26 the MinerU `standard` middle-JSON
reading of the 25-page pilot finished 25 of 25 pages, one engine call per page
(layout mean 11.115 s, copied stages about 0 s). Resolved primary text is 119,863
characters. Checked pages keep column order, formula numbers, and the page 768 table
(13 rows, two header rows, spans, five empty cells, a caption). The vault validated
with 0 findings. The review queue has 0 open entries on 25 read pages. The blind
worksheet has 0 of 25 pages accepted.

Commands on this tree, after that reading:

- `make lint-md`: 0 broken links, 0 run-path citations, 0 record-name findings.
- `make ci`: ruff, mypy, the complexity gate, the spec/plan gate, and the
  contract checks passed. Pytest reported 5277 passed, 2 skipped, 50 deselected.

## Handoff

Next action is the human gate `book-pilot-reference-transcription`: accept every pilot
page, then `make book-ocr-compare`. On this 8 GB card the four vLLM page readers stay
unstarted. PaddleOCR-VL and MinerU2.5-Pro were not re-timed under the single call.
