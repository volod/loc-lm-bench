# `book-evidence-constrained-correction`

## Work contract

```markdown
#### book-evidence-constrained-correction

Even the best engine leaves probable misreadings that context plus the page image can resolve. A
local model can propose those fixes, but a model that is allowed to "improve" text will rewrite an
author's technical meaning. Build correction as a constrained proposer whose edits a mechanical
guard admits or refuses. Prepare its adoption check for the release pilot.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- fake model and deterministic edit and refusal cases.
- Dependencies:
  [reading order](current/book-recovery/block-ledger.md#reading-order-and-paragraph-boundaries)
  records where a line joins its region. The model is the strongest vision-capable model that fits
  the host, chosen per
  [heavy runs and evidence](../guides/development/heavy-runs-and-evidence.md).
- User-visible outcome: an optional `correct` stage that proposes per-block edits from the page crop
  and neighbouring blocks through a local Ollama endpoint; each admitted edit appears in the overlay
  with raw text, replacement, class, evidence, and model identity; a refused or unsupported edit
  becomes an uncertainty marker instead.
- Scope boundary: in scope -- a structured edit contract (span, replacement, confusion class,
  evidence); the guard (admitted substitution classes only, a per-block edit budget, no insertion or
  deletion beyond its class, and visual agreement from the crop required for any digit, sign,
  decimal, unit, index, or Greek/Latin edit); bounded context from neighbouring blocks and pages;
  per-block caching. Out of scope -- rewriting, summarizing, completing, or translating; frontier
  models by default (an opt-in under the egress rule, refused for non-redistributable material).
- Data and artifact paths: `src/llb/prep/book/correct/`; `pages/<pNNNN>/corrections.json` in the run
  directory.
- Execution path: fake-model and score-replay fixtures inside `make ci`.
- Acceptance gates: `make ci` green; a proposed rewrite is refused; a digit change without
  visual agreement becomes a marker; an admitted edit reaches both renderings through the
  correction overlay; the raw ledger stays unchanged. Freeze the guard's classes and edit budget
  and pin a report that separates errors fixed from correct tokens made wrong. The stage defaults
  off pending the release reading.
- Documentation target: `docs/impl/current/book-recovery/correction.md`.
```

## Implementation and decisions

The optional `correct` stage lives in `src/llb/prep/book/correct/` and is not part of the default
stage list. A local Ollama vision model, or a `fake:` model whose proposals are in the stage
parameters, proposes structured edits. The guard admits or refuses each one. Admitted edits are
`author: model` corrections. `source` is the model identity and `confusion_class` is the class.
Refusals are recorded on the same corrections file and `resolve` leaves those spans unadopted, so
both renderings show an uncertainty marker. The block ledger is not written.

The guard's classes and the per-block budget are constants. Changing them changes the stage's tool
identity, so a finished page is not reused under a different guard. Every admitted class requires
visual agreement from the crop: there is no class the guard will take on language alone. A backend
other than `ollama` or `fake` is refused at preflight, so a frontier model is not called. The
adoption report counts characters fixed and characters made wrong, and the confusion classes that
moved, for the release reading. The stage stays off until that reading. The prompt substitutes
the block with a template, so braces in the OCR text stay braces. An empty reply content falls
back to the thinking field; a reply that is still not an edit list proposes nothing.

This pass did not call a vision model. No Ollama daemon is installed on the host, so the
reference scan was not read by the stage.

## Acceptance evidence

Status: `accepted`.

`make ci` on 2026-11-27: 5299 passed, 2 skipped, 50 deselected. `make lint-md` reported
no broken links and no citation or record-name findings. The fixture gates are the
rewrite refusal, the digit with no visual agreement becoming a marker, the admitted `5` to
`S` in both renderings, the unchanged ledger, the frontier-backend refusal, and the pinned
adoption report. The numbers and what would overturn them are in
[evidence-constrained correction](../current/book-recovery/correction.md#evidence).

## Handoff

`make plan-status` before the plan edit: 71 tasks, agent lane 46, human lane 25, 31 waiting on
open Requires, priority agent task `book-evidence-constrained-correction`. After the edit: 70
tasks, agent lane 45, human lane 25, 30 waiting, priority agent task
`book-projection-reference-sidecar`. CLEAR went from 32 to 31.

Removing this task drops its id from `book-technical-content-audit`,
`book-synthetic-book-end-to-end`, and `scanned-book-recovery-implementation-audit`.
`book-technical-content-audit` no longer waits, which is why the waiting count fell by one. It
still sits behind `book-projection-reference-sidecar` in the lane. No capability registry row
moved. `scanned-book-recovery` still has open work.
