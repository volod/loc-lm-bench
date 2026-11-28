# Full run on the target scan

## Work contract

```markdown
#### book-full-run-on-the-target-scan

Qualify the whole pipeline on the 1,067-page target scan with a provisional recovery profile.
Interrupt it, resume it, and stage the vault and projection with a report of what needs review.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: RUN NEEDED -- a multi-day sequential GPU run on the CUDA host.
- Dependencies: the target scan under `DATA_DIR`, its operator-declared redistribution
  class, and a host with enough VRAM for the provisional configuration.
  See the [publication audit](current/book-recovery/full-run-evidence.md#publication-and-resume-audit).
- User-visible outcome: a mechanically validated vault and ingested corpus projection
  for all 1,067 pages, with limitations and a named provisional profile.
- Scope boundary: qualify the whole target run, including chapter-sized batches,
  deliberate interruption and resume, vault publication and ingestion. Use fixture and diagnostic
  checks to pick the provisional profile. Reference accuracy, profile adoption, and operator
  review belong to the release gates.
- Data and artifact paths: `$DATA_DIR/book-ocr/<run_timestamp>/`; the operator's vault path; the
  projected corpus directory.
- Execution path: `make book-ocr` over the full page range under the GPU isolation contract of
  [heavy runs and evidence](../guides/development/heavy-runs-and-evidence.md).
- Acceptance gates: `make ci` green; `book-ocr validate` green with all 1,067 pages
  accounted for; the resume repeats no completed stage; a resumed render matches a clean render
  from the same cache; ingested chunks have scan provenance; report wall time against the pilot
  estimate and queue sizes per class and chapter. Mark the report as provisional.
- Documentation target: `docs/impl/current/book-recovery/full-run-evidence.md` and the usage guide
  `docs/guides/data-prep/book-ocr.md`.
```

## Implementation and decisions

The [provisional MinerU Standard profile](../current/book-recovery/full-run-evidence.md#provisional-mineru-profile)
uses the middle-JSON pilot lane on the original render, with 25 pages per batch and one
recognition call per page. Stage preflight now checks a recorded VRAM budget on a dry resume
and on execution. The vault validation report counts open work by primary page class and
inferred chapter, with selected-page denominators and terminal page failures.

The projection carries the prior local-only classification for this scan. The vault,
projection, and ingested copy are staged in their run directory.

## Acceptance evidence

Status is accepted. The full-scan dry run found 1,067 image-only pages. The GPU start, deliberate
interruption, and resume are recorded with the
[profile](../current/book-recovery/full-run-evidence.md#provisional-mineru-profile).
All 78 stage records present at interruption kept their hashes. The resume log shows no
repeat execution for those records.

On 2026-11-28, the HP ZBook Fury G1i with RTX PRO 3000 Blackwell laptop GPU completed all 43
batches with zero failed stages. Recovery took 7,659 seconds across both invocations. The
25-page pilot estimate was 11,860 seconds on the RTX 5050 laptop. The provisional report states
the measured times and records zero open entries across 10 primary classes and 72 chapter
groups, including pages outside inferred chapters. MinerU supplied no confidence scores.

`make book-ocr-review-queue`, `make book-ocr-structure`, `make book-ocr-render`, and
`make book-ocr-validate` accounted for all 1,067 pages with zero findings. Two vault renders
from the same cache matched across 3,174 files and symlinks. The original PDF checksum and
all 1,067 render fallback images matched. `make book-ocr-project` and `make ingest-corpus`
retained all 1,216 local-only documents. All 32,062 fixed 180-character chunks carried scan
provenance and all 13,791 citation rows resolved to vault anchors. The
[full provisional run](../current/book-recovery/full-run-evidence.md#full-provisional-run)
records remaining folio, structure, and reference limitations.

`make format`, `make ci`, `make lint-md`, and `git diff --check` passed. CI reported 5,339
passed, 7 skipped, and 50 deselected.

## Handoff

The plan went from 64 tasks (39 agent and 25 human) to 63 (38 agent and 25 human). Only the
scanned-book recovery group moved. The library extraction and optional implementation audit
drop this prerequisite. Library extraction is the next agent task. The capability stays
planned pending library extraction and the human release gates.
