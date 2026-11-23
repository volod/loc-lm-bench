# book-reading-resolution-and-correction-overlay

## Work contract

```markdown
#### book-reading-resolution-and-correction-overlay

The ledger keeps every engine's raw reading and chooses none, and the spec puts every correction --
by agreement, by a model, by a reviewer -- in an overlay over it. Neither half exists: no rule says
which reading a rendering shows for a block, and there is no overlay for a decision to land in. The
markers, both renderings, the correction stage, and the review queue all read this one seam, so it
is built once, before any of them. An overlay keyed by block id alone would also attach a decision
to the wrong block the first time a ledger is rebuilt under another engine, because ids follow
geometry.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: the [block ledger](current/book-recovery/block-ledger.md) and its uncertainty spans.
- User-visible outcome: every block has one resolved text with its provenance -- the configured
  primary reading, an overlay edit, or an unresolved span -- that both renderings read; each overlay
  edit records its author kind (agreement, model, reviewer), raw text, replacement, and evidence;
  an edit whose block changed under a ledger rebuild is reported as orphaned and re-queued, never
  applied to another block.
- Scope boundary: in scope -- the resolution rule (the primary reading per stage and page class from
  the run configuration; agreement resolves only what every reading agrees on; an uncertainty span
  stays unresolved unless an overlay edit covers it); the overlay as a registered artifact contract
  with one precedence (reviewer over model over agreement); entries anchored by block id plus the
  digest of the block's region and raw readings, so a changed block orphans its entries; model
  edits arriving as a stage output and reviewer decisions as a run-level file outside the stage
  cache, both entering the key of every stage that reads them, so a new decision re-runs exactly the
  pages it touches. Out of scope -- producing edits (the review queue and the correction stage do),
  and rendering.
- Data and artifact paths: `src/llb/prep/book/resolve/`; the overlay and the resolved text per page
  inside the run directory.
- Execution path: CPU-only inside `make ci` over synthetic ledgers.
- Acceptance gates: `make ci` green; a block with three readings and a digit span resolves to the
  primary reading with the span left unresolved; agreement across every reading resolves; a reviewer
  edit overrides a model edit on the same span; rebuilding a fixture ledger under a second fake
  engine orphans and re-queues the entries of every changed block while unchanged blocks keep
  theirs; a new reviewer decision changes the resolved text of its page alone on the next run; the
  raw ledger stays byte-identical throughout.
- Documentation target: `docs/impl/current/book-recovery/uncertainty-and-review.md`.
```

## Implementation and decisions

`resolve` writes `overlay.json` and `resolved.json` for one page and does not rewrite
`blocks.json`. Agreement edits are produced by the rule. Model edits are read from a `correct`
stage output this task does not produce. Reviewer decisions are `review/decisions.json`, outside
the stage cache, and only that page's edits enter the key. A missing file digests as an empty
list, so a file that does not edit a page does not re-run it.

Unresolved characters stay on the span and are omitted from the adopted text. A later marker can
see them, and a copy of `text` cannot hide them. A letter disagreement that is not an uncertainty
span stays the primary reading. Two of three readings do not outvote a third. A block that lacks
the configured reading records `fallback` instead of failing the page.

The anchor is the digest of the block's region and its raw readings. An edit whose block changed
is orphaned with `requeued` true and is not tried against another block, including one that kept
the old digest under a new id.

The handbook run showed a cache alias. Identical page images share a recognition key, and reusing
the ledger then copied another page's block ids onto the duplicate. The ledger and the signal
file name the page, so those keys now include the PDF page. Recognition stays a function of the
image. The rule and the handbook counts are in
[uncertainty and review](../current/book-recovery/uncertainty-and-review.md).

## Acceptance evidence

Status: accepted.

- **`make ci` green.** On 2026-11-22: 5,114 passed, 2 skipped, 50 deselected. No model was
  loaded. Resolution did not use the GPU. Two venv-setup tests had been failing on this host
  because they inherited `UV_LINK_MODE=copy` and then read a log the script never created.
  They now clear that variable. `make lint-md` was run on this tree.
- **A digit span stays unresolved, agreement resolves only a unanimous span, and a reviewer edit
  wins.** `tests/llb/prep/book/test_resolve.py`.
- **A changed block is re-queued, and the edit does not follow an id that moved.** The same file.
  The raw ledger in each test is unchanged.
- **A new reviewer decision re-runs its page alone.**
  `test_a_new_reviewer_decision_changes_its_page_alone` in
  `tests/llb/prep/book/test_resolve_stage.py`. A model edit whose stage output changes re-runs
  that page and leaves the next page's text.
- **Duplicate rasters keep their own block ids.**
  `test_reuse_keeps_each_duplicate_page_its_own_block_ids` in
  `tests/llb/prep/book/test_ledger_reuse.py`.
- **The handbook pass.** On 2026-11-22, CPU only, on the GeForce RTX 5050 laptop (8 GB). The
  counts are in [uncertainty and review](../current/book-recovery/uncertainty-and-review.md).
  Render and recognition were reused. The ledger stage ran and stayed byte-identical to the
  recognition run on all 1,067 pages. Resolve ran on all 1,067 pages, none failed, in 28 s.
  Resuming cached all 4,268 page-stages.

## Handoff

`make plan-status` immediately before the task left the plan: 72 tasks (agent 47, human 25).
After: 71 (agent 46, human 25). `scanned-book-recovery` lost this task and gained none. The
priority agent task is `book-uncertainty-markers-and-review-queue`. The priority human task is
still `book-pilot-reference-transcription`. Nothing from this task needed a new plan item: the
marker and the review queue are already the next task, and the page-identity cache key is part
of this delivery.
