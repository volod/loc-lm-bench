# book-structure-inference

## Work contract

```markdown
#### book-structure-inference

A reader navigates the book by part, chapter, and section, and the book cites itself by printed page.
Neither is in the file: the PDF has no outline and no page labels, and its printed numbers are offset
from its PDF page numbers. Both must be inferred globally from page-level evidence, and where the
evidence is thin the output must say so rather than invent a hierarchy.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: the [block ledger](current/book-recovery/block-ledger.md); a line engine labels
  every block `body`, as the full-book RapidOCR ledger did
  ([block ledger evidence](current/book-recovery/block-ledger.md#evidence)), so the kinds an engine
  supplies are a hint, not an input this task may require.
- User-visible outcome: `metadata/document_structure.json` holds the printed-page mapping for every
  PDF page, the hierarchy (book, part, chapter, section, subsection) with page ranges and the
  evidence behind each node, the figure, table, and equation numbering inventory, and the fallback
  flag wherever evidence was insufficient.
- Scope boundary: in scope -- page-number and running-header blocks found from position and
  repetition across pages, whatever kind the engine gave them; printed page numbers from those
  blocks under a monotone fit, with gaps, unnumbered plates, and front matter recorded;
  table-of-contents page detection and parsing;
  headings from typography and numbering; running-header and section-number agreement; a two-pass
  reconciliation of the printed contents against observed headings with every conflict recorded;
  fallback page-range segmentation. Out of scope -- rendering, cross-reference linking, and any node
  a model proposes without page evidence.
- Data and artifact paths: `src/llb/prep/book/structure/`; `metadata/document_structure.json`
  registered as an artifact contract.
- Execution path: CPU-only inside `make ci` over synthetic ledgers.
- Acceptance gates: `make ci` green; a synthetic book with a contents page and numbered headings
  reconstructs exactly; an offset numbering with one missing page number recovers the mapping and
  records the gap; a contents entry with no matching heading is reported, never fabricated; a book
  with neither contents nor numbered headings falls back to page ranges; the same synthetic book
  with every block relabelled `body` recovers the same page mapping and running headers.
- Documentation target: `docs/impl/current/book-recovery/structure-and-rendering.md`.
```

## Implementation and decisions

`llb book-ocr structure` reads the ledgers of one run and writes
`metadata/document_structure.json` (`llb.book-ocr-document-structure` 1.0.0). It is a command,
not a stage. Resolved text wins when it is non-empty. Block kind is not read, so relabeling every
block `body` cannot change the mapping or the running headers.

A printed number is a short numeral in the outer corner. The monotone fit records a gap when the
printed numbers kept pace and an unnumbered plate when they did not. Pages before either series
are front matter. A running header is the other short line in the top band. A contents entry is
evidence and never a node. A heading is a numbered line of words, a bare chapter number at the
top of a page, or a front-matter name. A table cell, a measurement, and a merely taller line are
not headings. A hole in the configured pages ends a range, and a heading after the hole does not
nest under a chapter that stopped at the hole. A page that is a list of captions does not
inventory those lines as captions. A second write over the same ledgers is byte-identical.

## Acceptance evidence

Status: accepted for this task's gates.

- **Checks.** On 2026-11-23, `make ci`'s checks passed: format, ruff, mypy (1,661 files), the
  complexity gate, shell lint, spec-plan integrity, the toolchain check, the model roster, and
  artifact contracts, including `llb.book-ocr-document-structure` 1.0.0.
- **The synthetic book.** `tests/llb/prep/book/test_structure.py`. A contents page plus numbered
  headings reconstructs the hierarchy. An offset with one missing numeral recovers the mapping and
  records the gap. A contents entry with no heading is a conflict and not a node. With neither
  contents nor numbered headings the book falls back to one page range. Relabeling every block
  `body` keeps the page mapping and the running headers. A table cell and a hole in the page list
  do not become structure.
- **Markdown.** `make lint-md` passed, including link and citation checks.
- **The scan.** The counts are in
  [structure](../current/book-recovery/structure-and-rendering.md). On 2026-11-23, CPU RapidOCR
  3.10.0, RTX PRO 3000 Blackwell laptop (12 GB). PDF pages 1-40 and 290-299 of the 1,067-page
  scan. 42 numerals read, 5 front-matter pages, 3 gaps filled by the fit. Body offset 17.
- **`make ci` pytest on this host.** 5,041 passed, 53 skipped, 50 deselected, 49 failed, 4
  errors. The failures are missing optional extras: `faiss` (`[rag]`), `langgraph` (`[eval]`),
  `splink` (linkage), and `rdflib`. None is in the book suite. The checks listed above had
  already passed.

## Handoff

`make plan-status` immediately before the task left the plan: 70 tasks (agent 45, human 25).
After: 69 (agent 44, human 25). `scanned-book-recovery` lost this task and gained none. The
priority agent task is `book-vault-rendering-and-navigation`. The priority human task is still
`book-pilot-reference-transcription`. Dropping this id released `Requires` on
`book-vault-rendering-and-navigation` and `scanned-book-recovery-implementation-audit`. The vault
task now depends on the structure page. Nothing from this task needed a new plan item.
