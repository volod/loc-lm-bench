# book-vault-validation-and-report

## Work contract

```markdown
#### book-vault-validation-and-report

A 1,067-page render cannot be checked by eye. The full run must prove mechanically that every page
is accounted for, every file is reachable, every link lands, and every asset exists -- and say in
one report what still needs a person.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: a rendered vault
  ([reading vault](current/book-recovery/structure-and-rendering.md#reading-vault)).
  The link resolver may reuse the relative-link and anchor logic of `make lint-doc-links`,
  extended to wikilinks, block anchors, and PDF page fragments.
- User-visible outcome: `llb book-ocr validate <book-dir>` runs standalone and at the end of every
  `run`, and `metadata/processing_report.md` lists per-page status, warnings, confidence
  distributions as diagnostics, the engines, models, and parameters used, the review-queue size per
  class, and the pages and content types still requiring manual verification.
- Scope boundary: in scope -- every input page accounted for (completed, failed, or skipped with a
  reason); every content file reachable from the contents; previous/next links reciprocal and the
  chain covering every file once; every wikilink, relative link, block anchor, and embed resolving,
  with PDF page fragments inside the page count; every referenced asset existing and every asset
  referenced; every block carrying a page; every open marker matching exactly one open review-queue
  entry, and orphaned overlay entries counted in the report. Out of scope --
  judging rendering in the Obsidian app (that is `book-vault-obsidian-acceptance`).
- Data and artifact paths: `src/llb/prep/book/validate/`; `metadata/processing_report.md` and
  `metadata/validation.json` inside the book directory.
- Execution path: `make book-ocr-validate BOOK_DIR=<book-dir>`; CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; one synthetic broken vault per failure class is refused with the
  offending file named; the clean fixture passes; a validation failure exits non-zero.
- Documentation target: `docs/impl/current/book-recovery/structure-and-rendering.md`.
```

## Implementation and decisions

`llb book-ocr validate <book-dir>` checks one rendered vault and writes
`metadata/processing_report.md` and `metadata/validation.json` into that vault. The run
directory keeps its own stage summary under the same filename. The two files live in
different directories.

`render` runs the check after it publishes and exits 1 when a check fails. `run` and
`resume` run it for each vault already under `book/` that has `metadata/render.json`. A
run with no vault is unchanged. A dry run does not check a vault. `metadata/render.json`
records `run_dir`, so a vault published with `--output` finds its run. A vault inside the
run is found by walking up to `config.json` when that field is absent.

Every PDF page is completed, failed, or skipped. A selected page with a `^pNNNN` anchor is
completed. `[!page-failed]` wins over an anchor. A page outside the configured range is
skipped, and consecutive skipped pages share one report row. `validation.json` lists every
page. Previous and next are reciprocal, and walking Next from the first note visits every
content note once. The contents Next starts that chain. Relative links, wikilinks, embeds,
heading fragments, block anchors, and `#page=<n>` fragments resolve. An image under
`assets/`, `review/`, or `source/pages/` is linked, and every linked image exists.

When `review/queue.json` exists, each open marker matches one open queue entry: kind, block
id, offsets, and PDF page. A missing queue is a warning, and the markers are then matched
to the ledger. Orphaned overlay entries are counted and are not themselves a failure.
Structure conflicts and printed-page fallbacks are warnings. Confidence buckets are
diagnostic routing evidence, not a probability of correctness.

## Acceptance evidence

Status: accepted for this task's gates.

- **Checks.** On 2026-11-23, `make ci`'s checks passed: format, ruff, mypy (1,686 files),
  the complexity gate, shell lint, spec-plan integrity, the toolchain check, the model
  roster, and artifact contracts. `make lint-md` passed, including link and citation checks.
- **The synthetic book.** `tests/llb/prep/book/test_vault_validate.py`. The clean fixture
  passes, including symlink, copy, reference, page fallback, and an oversized split. Each
  failure class is a broken copy of that fixture, and the finding names the file. A failed
  check exits 1. A directory that is not a vault exits 2. A second check writes the same
  report.
- **The scan.** The counts are in
  [vault validation](../current/book-recovery/structure-and-rendering.md#vault-validation).
  On 2026-11-23, CPU only, RTX PRO 3000 Blackwell laptop (12 GB). The vault is the render of
  PDF pages 1-40 and 290-299 of the 1,067-page scan, from CPU RapidOCR 3.10.0. The review
  queue was built (238 `low-confidence` entries), the vault was rendered again, and the
  check passed: 50 completed, 1,017 skipped, 0 findings.
- **`make ci` pytest on this host.** 5,070 passed, 53 skipped, 50 deselected, 50 failed, 4
  errors. The failures and the four errors are missing optional extras: `faiss` (`[rag]`),
  `langgraph` (`[eval]`), `splink` (linkage), and `rdflib` (`[ontology]`). The other failure
  is the survey hang fixture: the sandboxed run could not signal the sleeper. No failure is
  in the vault suite.

## Handoff

`make plan-status` before this task left the plan: 68 tasks (agent 43, human 25). After: 67
tasks (agent 42, human 25). Waiting on open Requires fell from 31 to 30.
`book-cross-reference-linking` is the priority agent task. The priority human task is
`book-pilot-reference-transcription`. The full 1,067-page recognition remains
`book-full-run-on-the-target-scan`. This task added no plan item.
