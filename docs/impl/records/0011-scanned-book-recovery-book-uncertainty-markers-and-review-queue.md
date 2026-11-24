# book-uncertainty-markers-and-review-queue

## Work contract

```markdown
#### book-uncertainty-markers-and-review-queue

The ledger can say a span is uncertain; nothing yet turns that into something a reader sees and a
reviewer can act on. An omitted passage and a passage nobody could read must never look the same.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: the resolved text and its overlay
  ([uncertainty and review](current/book-recovery/uncertainty-and-review.md)). The decision UI
  reuses the [review workbench](current/review-workbench.md) adapter seam.
- User-visible outcome: every unreadable or uncertain span becomes a short visible marker with its
  PDF page link and, where useful, a padded crop of the region, a row in a page-level quality
  manifest, and an entry in a review queue that opens in the review workbench; a reviewer decision
  becomes a correction-overlay entry that survives every re-render of an unchanged block, and an
  orphaned entry returns to the queue rather than disappearing.
- Scope boundary: in scope -- marker syntax that is readable in Obsidian and machine-parseable, with
  distinct markers for unreadable, low-confidence, and uncertain formula or table content;
  deterministic crop names under `review/`; `metadata/quality_manifest.json`; the review-queue file;
  a workbench adapter whose typed decisions (accept raw, accept candidate, enter text, mark
  illegible) write to the overlay. Out of scope -- model correction, and working the queue (that is
  `book-review-queue-resolution`).
- Data and artifact paths: `src/llb/prep/book/review/`, `src/llb/review/adapters/`; `review/` and
  `metadata/quality_manifest.json` inside the book directory.
- Execution path: CPU-only inside `make ci`; `make review-workbench REVIEW_PATH=<review-queue>` opens
  the queue.
- Acceptance gates: `make ci` green; a planted unreadable span yields a marker, an existing crop, a
  manifest row, and a queue entry -- never an omission; a workbench decision changes the rendered
  text and leaves the raw ledger untouched; reopening the queue resumes where it stopped; an
  orphaned overlay entry reappears in the queue with its old decision shown; the adapter passes the
  workbench's ledger-compatibility tests.
- Documentation target: `docs/impl/current/book-recovery/uncertainty-and-review.md` and the
  [review workbench](current/review-workbench.md) adapter table.
```

## Implementation and decisions

The queue is a command over a finished `resolve`, not a stage. `llb book-ocr review-queue` writes
`review/queue.json`, `metadata/quality_manifest.json`, and one padded grey crop per open span. The
workbench adapter `book-review-queue` turns accept-raw, accept-candidate, enter-text, and
mark-illegible into reviewer edits in `review/decisions.json`. The next `resolve` applies them.
The raw ledger is not rewritten. An orphan comes back with its old decision and an empty verdict.

An unresolved span, an empty content block, and a primary score below 0.9 each become a marker.
A figure or a piece of furniture is not queued only because its text is empty. A missing score
does not route. Mark illegible writes an `[!unreadable]` marker, so the passage is not a blank.
The 0.9 cut is the sample reading below: 0.5 queued nothing on this scan's RapidOCR scores.

## Acceptance evidence

Status: accepted for this task's gates.

- **Checks.** On 2026-11-23, `make ci`'s checks passed: format, ruff, mypy, the complexity
  gate, shell lint, spec-plan integrity, the toolchain check, and artifact contracts, including
  the two new families.
- **The planted span, the decision, resume, and the orphan.**
  `tests/llb/prep/book/test_review_queue.py` and `tests/llb/review/test_adapters.py`. An empty
  body block yields a marker, a crop, a manifest row, and a queue entry. Accept raw changes the
  resolved text and leaves the ledger bytes unchanged. Reopening the queue resumes at the pending
  entry. An orphaned edit comes back with its old replacement. The adapter's bytes match the
  decision writer. `tests/llb/prep/book`, `tests/llb/review`, and
  `tests/llb/cli/test_cli_book_ocr.py` passed.
- **Markdown.** `make lint-md` passed, including link and citation checks.
- **The handbook sample.** The counts are in
  [uncertainty and review](../current/book-recovery/uncertainty-and-review.md). On 2026-11-23,
  CPU RapidOCR, RTX PRO 3000 Blackwell laptop (12 GB). PDF pages 1-15 and 290-299. 179
  low-confidence entries, each with a crop. A cut of 0.5 queued none.
- **`make ci` pytest on this host.** 5,026 passed, 53 skipped, 50 deselected, 50 failed, 4
  errors. The failures are missing optional extras: `faiss` (`[rag]`), `langgraph` (`[eval]`),
  `splink` (linkage), `rdflib`, and the NVIDIA container toolkit. None is in the book or review
  suites.

## Handoff

`make plan-status` immediately before the task left the plan: 71 tasks (agent 46, human 25).
After: 70 (agent 45, human 25). `scanned-book-recovery` lost this task and gained none. The
priority agent task is `book-structure-inference`. The priority human task is still
`book-pilot-reference-transcription`. Dropping this id released `Requires` on
`book-vault-rendering-and-navigation`, `book-evidence-constrained-correction`, and
`scanned-book-recovery-implementation-audit`. Nothing from this task needed a new plan item.
Working the queue remains `book-review-queue-resolution`.
