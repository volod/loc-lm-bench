# book-recovery-quality-metrics

## Work contract

```markdown
#### book-recovery-quality-metrics

The pilot comparison and the correction adoption gate both need metrics that understand how
technical OCR fails: an average character error rate hides the single `5` read as `S` in a table
cell that wrecks a value. Build the harness, and the reference format it scores against, before any
pilot number is taken.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: the [block ledger](current/book-recovery/block-ledger.md).
- User-visible outcome: `llb book-ocr score --reference <reference-dir> <run-dir>` reports, per page
  and per page class: character and word error, missing and duplicated text, reading-order
  agreement, the dangerous-substitution count per confusion class, formula fidelity, table-structure
  similarity, figure-boundary overlap, caption association, and printed-page mapping accuracy; the
  same report scores the existing PDF lane's conversion of the same pages.
- Scope boundary: in scope -- the reference-transcription format (per-page blocks with kind, region,
  text, LaTeX for formulas, structure for tables, figure boxes with caption ids, reading order, and
  the printed page number); alignment; the confusion classes (`0/O`, `1/l/I`, `5/S`, minus versus
  hyphen versus en and em dash, decimal point versus comma, Latin versus Greek lookalikes, sub- and
  superscript, unit symbols, brackets, equation numbers); LaTeX normalization before comparison plus
  a parse-validity check; formula fidelity compared term by term, so an equation that parses but
  lost a term scores as the loss (the survey's dropped `= v_E(t_0)`); tree-edit table similarity;
  box overlap; rank agreement on reading order; scoring the existing lane from its page-level
  citation spans as a text-only submission -- a formula it wrote as plain text is scored against the
  reference LaTeX like any other reading, and a structure it cannot express (figure boxes, table
  structure) is reported as absent, never dropped from the comparison. The dangerous-substitution
  rates are reported apart and never folded into a single score. Out of scope -- any reference
  content (that is `book-pilot-reference-transcription`) and a combined headline score.
- Data and artifact paths: `src/llb/prep/book/metrics/`; synthetic reference fixtures under
  `tests/llb/prep/book/`.
- Execution path: CPU-only inside `make ci`; `make book-ocr-score` (with a `##` help description)
  wraps the command.
- Acceptance gates: `make ci` green; every metric is pinned by a hand-computed fixture; a planted
  `5`-for-`S` substitution in a table cell is counted in its class while barely moving the character
  error rate, and the fixture asserts both; equivalent LaTeX spellings score equal; invalid LaTeX is
  flagged; a parsing formula with one term removed scores below the complete one; a fixture
  existing-lane conversion scores through the same harness with its absent structures reported as
  absent.
- Documentation target: `docs/impl/current/book-recovery/quality-metrics.md`.
```

## Implementation and decisions

The harness is `llb book-ocr score` and `make book-ocr-score`. A reference directory is
`pages/*.json`. A book-ocr lane is one engine of a block ledger (`book-ocr/<engine>`); the
existing PDF lane is one citation sidecar (`existing-pdf`) and is text-only. Figure boxes, table
grids, and caption links that lane cannot express are `absent`, with a null value. A formula it
wrote as plain text is scored against the reference LaTeX. Dangerous-substitution counts are a
separate map. `ScoreReport` has no combined score.

The reference and the report are pydantic models. They are not a new artifact family: this is
still the pre-release contract, and the task's paths do not ask for a schema. Character error
compares reading strings; fidelity compares canonical terms, so two equivalent spellings can move
the character error and still score fidelity 1. Script events are counted from their structure
and stripped before the character pass, so a subscript is not also a digit class. Equation
numbers are masked the same way. A hyphen against a minus inside an otherwise equal equation
number stays a minus-class event.

Detail, the hand-computed pins, and how to run it:
[quality metrics](../current/book-recovery/quality-metrics.md).

## Acceptance evidence

Status: accepted.

- **`make ci` green.** On 2026-11-22, with `UV_LINK_MODE` and `UV_CACHE_DIR` unset: 5,084 passed,
  2 skipped, 50 deselected. `make lint-md` was green and `make lint-spec-plan` reported 0 findings.
  No model was loaded and no GPU memory was taken.
- **Every metric is pinned by a hand-computed fixture.** `tests/llb/prep/book/test_quality_text.py`,
  `test_quality_structure.py`, and `test_quality_score.py`, on the synthetic page
  `tests/llb/prep/book/fixtures/metrics/reference/pages/p0001.json`. The fractions and what would
  overturn them are in [quality metrics](../current/book-recovery/quality-metrics.md#evidence).
- **A planted `5` for `S`.** A 200-character, 96-word table cell ending in `5`, read as `S`:
  character error 1/200, word error 1/96, class `5/S` equal to 1, every other class 0, table
  similarity 5/6.
- **Equivalent LaTeX, invalid LaTeX, and a dropped term.** `v_E` and `v_{E}` score fidelity 1.
  `\frac{a}{` does not parse. The four-term equation scores 1; dropping `= v_{E}(t_{0})` scores
  3/4, including when the remaining terms are plain text.
- **The existing lane.** The fixture reading submitted as a citation sidecar scores character
  error 0, reading order 1, printed page 1, and formula fidelity 1. Table similarity, figure
  overlap, and caption association are `absent`. The report has no `score` or `headline` field.

## Handoff

`make plan-status` immediately before the task left the plan: 74 tasks (agent 49, human 25).
After: 73 (agent 48, human 25). `scanned-book-recovery` lost this task and gained none. The
priority agent task is `book-pilot-page-selection`. The harness does not transcribe the target
handbook; that stays `book-pilot-reference-transcription`.
