# book-resolved-pipeline-scoring

## Work contract

```markdown
#### book-resolved-pipeline-scoring

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: [quality metrics](current/book-recovery/quality-metrics.md) and
  [resolved readings](current/book-recovery/uncertainty-and-review.md).
- User-visible outcome: score the actual routed, resolved pipeline beside the raw engine lanes
  and the existing PDF conversion on exactly the frozen reference pages.
- Scope boundary: read adopted text, table structure, formula output, caption links, printed-page
  mapping and unresolved spans from the artifacts each renderer consumes. Score the fields each
  pipeline can emit and mark unsupported fields absent. Keep raw-engine scores for diagnosis.
  Report missing submissions, unscorable fields and abstentions by class.
- Data and artifact paths: `src/llb/prep/book/metrics/` and the pilot score report.
- Execution path: exercise the score CLI with a synthetic multi-engine run inside `make ci`.
- Acceptance gates: a variant switch and one overlay edit each change only the expected lane and
  page; a missing page remains in the denominator; a wrong caption or folio in the output loses
  credit even when a nearest-caption or body-text guess would match. Verify class totals by hand.
- Documentation target: `docs/impl/current/book-recovery/quality-metrics.md`.
```

## Implementation and decisions

`llb book-ocr score` adds a `book-ocr/resolved` lane whenever the run has
`pages/<page-id>/resolved.json`. `--engine` still limits the raw lanes and still scores this
one. The lane reads the artifacts the vault and the corpus projection render from: adopted text
and unresolved spans from `resolved.json`, kind and region from the block ledger, the printed
folio and the equation inventory from the structure file. Caption links are the vault's neighbor
pairs (`linked_caption_ids` in `src/llb/prep/book/render/segment.py`). A table is the grid
`table_markdown` renders. A formula is the display math `formula_math` renders.

A structural field the resolved page did not emit is `absent`, not zero. Raw engine lanes still
score a missing block as zero. An unresolved span increments each of its uncertainty classes and
is not subtracted from character error. A reference page with no submission stays in the page
count and in the character-error mean, and `missing_submissions` counts it.

Identical readings return edit distance 0 without building the alignment matrix. Unequal readings
still use the edit script. The metric values are unchanged.

`segment.py` holds the neighbor rule because that rule is the vault's pairing rule. The file sits
a few lines over the soft size limit.

## Acceptance evidence

Status: accepted.

`make plan-status` before the plan edit: 70 tasks (45 agent, 25 human); statuses BLOCKED BY
HUMAN=4, CLEAR=30, HUMAN-GATED=21, RUN NEEDED=15; 33 tasks waiting on open Requires; agent
priority `book-resolved-pipeline-scoring`. After: 69 tasks (44 agent, 25 human); CLEAR=29; 32
waiting; agent priority `book-input-and-pilot-audit`. Human priority stayed
`book-pilot-reference-transcription`. Removing this task dropped its id from the Requires of
`book-input-and-pilot-audit` (that task no longer waits) and from
`scanned-book-recovery-implementation-audit` (that audit still waits on the remaining tasks).
`scanned-book-recovery` stays `planned`. No registry row changed.

The synthetic gates are `tests/llb/prep/book/test_quality_resolved.py`, run on 2026-11-24 with no
model loaded. The hand counts, the variant switch, the overlay edit, the missing page, the wrong
neighbor caption, the wrong folio, the absent fields, and the CLI summary are recorded in
[quality metrics](../current/book-recovery/quality-metrics.md#resolved-lane-evidence).

`make lint-md` passed. `make ci` passed: 5,194 tests, 7 skipped, 50 deselected.

**What ran.** On 2026-11-24, on the Intel Core Ultra 7 265HX host, CPU only. The resolved lane
read the 50-page slice (PDF pages 1-40 and 290-299) of the 1,067-page Space Mission Engineering
scan, SHA-256 `89101813d4c695e9f992d540e751c4e2f4b80c6550f24d5d23e1a795383cb7c8`, 163,709,872
bytes. The slice is one RapidOCR reading: 5,278 body blocks, no unresolved spans, and a
printed-page mapping on every page. Scored against a reference built from that adopted text and
those mapped folios, both the RapidOCR lane and `book-ocr/resolved` had character error 0 on all
50 pages and 0 missing submissions. The resolved lane had printed-page accuracy 1 on all 50 pages
and 0 abstentions. Pages 1-5 are unnumbered and both lanes score printed-page accuracy 1. On the
other 45 pages the raw lane scores 0, because the ledger has no page-number block. The mapped
folio occurs as a whole word in the body on 39 of those 45 pages, and that occurrence does not
give the raw lane credit. The temporary reference was deleted. A later score of this slice
against the same adopted text and the same mapped folios that moves the resolved character error
off 0, or the resolved printed-page accuracy off 1, overturns the round trip.

## Handoff

Nothing remains on this task. The next agent task is `book-input-and-pilot-audit`. Transcribing
the scan is still `book-pilot-reference-transcription`.
