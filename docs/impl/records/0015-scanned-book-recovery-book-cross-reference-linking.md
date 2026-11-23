# book-cross-reference-linking

## Work contract

```markdown
#### book-cross-reference-linking

The book refers to itself constantly -- figure, table, and equation numbers, numbered sections,
chapters, printed pages, and a back-of-book index citing printed pages. Turning those references into
links is the largest readability gain the vault offers over the scan, and a wrong link is worse
than none, so resolution must be conservative and every miss counted.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: a vault that passes
  [vault validation](current/book-recovery/structure-and-rendering.md#vault-validation).
- User-visible outcome: references resolve to links to the target block anchor or file; generated
  lists of figures, tables, and equations; index entries linked to the content file holding the
  cited printed page; ambiguous or unresolved references left as plain text and counted in the
  report.
- Scope boundary: in scope -- a per-book reference grammar in configuration; resolution only to a
  target that the structure inventory identifies uniquely; printed-page references and index
  entries through the page mapping to the anchor at the first block of the cited page, so a page
  split across two files still lands where it starts. Out of scope -- concept or semantic links,
  model-suggested related sections, and a generated glossary.
- Data and artifact paths: `src/llb/prep/book/render/`; the generated lists inside the book
  directory.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; every reference type resolves on a fixture; a reference made
  ambiguous by two identically numbered figures stays unlinked and reported; a reference to a
  printed page inside a mapping gap stays unlinked; `book-ocr validate` passes on the linked output.
- Documentation target: `docs/impl/current/book-recovery/structure-and-rendering.md`.
```

## Implementation and decisions

`llb book-ocr render` links citations while it writes the vault. It stays a command, not a
stage. The handbook grammar is the default. `--reference-grammar` and `BOOK_REFERENCE_GRAMMAR`
replace that grammar entirely. A missing file, an unknown key, or a grammar with no page cues
or no index titles is exit code 2.

A number links only when the catalog has one target. A caption or a display number defines a
figure, table, or equation. A page-footer echo of the same number is the running index line,
not a second item. Two captions, or several footers and no caption, stay plain, and the key
is reported. A section or a chapter follows the same uniqueness rule. A printed page links
when one observed page carries that label, at the anchor of the first block of that PDF page,
so a page split across two notes opens where it starts. A mapping gap stays plain. It counts
as unresolved and as a gap, and validation warns. The warning does not fail the check.

The cue stays outside the link. Each number in a comma, `and`, or dash list resolves on its
own. `Chaps. 3-6` cites chapters 3 and 6. `Chapter 22.1.5` stays plain, because a dotted tail
is not a range. A figure or equation pattern that already contains a hyphen keeps `Fig. 1-1`
as one number. The scanned cues `Fig,` and `Eq,` stand in for the period the OCR dropped.

Lists of figures, tables, and equations sit at the vault root, outside the previous/next
chain, and the contents links them. A row is the number, its own anchor, the printed page
when the inventory has one, and the PDF page. `metadata/reference_links.json` holds the
counts and the ambiguous keys. It quotes no sentence and names no run.

## Acceptance evidence

Status: accepted for this task's gates.

- **Checks.** On 2026-11-23, `make ci`'s checks passed: format, ruff, mypy (1,693 files),
  the complexity gate, shell lint, spec-plan integrity, the toolchain check, the model
  roster, and artifact contracts. `make lint-md` passed, including link and citation checks.
- **The fixture.** `tests/llb/prep/book/test_reference_links.py`. Every reference type
  resolves. Two figures numbered `1-1` stay plain and the key is reported. A printed page
  inside a mapping gap stays plain. A page split across two notes links to the note where
  that page starts. A one-cue grammar replaces the handbook cues. `book-ocr validate` passes
  on the linked fixture.
- **`make ci` pytest on this host.** 5,077 passed, 53 skipped, 50 deselected, 50 failed,
  4 errors. The failures and the four errors are missing optional extras: `faiss` (`[rag]`),
  `langgraph` (`[eval]`), `splink` (linkage), and `rdflib` (`[ontology]`). The other failure
  is the survey hang fixture: the sandboxed run could not signal the sleeper. No failure is
  in the linking suite.
- **The scan.** The counts are in
  [cross-references](../current/book-recovery/structure-and-rendering.md#cross-references).
  On 2026-11-23, CPU only, RTX PRO 3000 Blackwell laptop (12 GB). Render read the structure
  and resolved text of PDF pages 1-40 and 290-299 of the 1,067-page scan, from CPU RapidOCR
  3.10.0. Recognition was not repeated. The slice has no index. Resolved 128, ambiguous 52,
  unresolved 251, mapping gaps 0. `book-ocr validate` reported 0 findings and the same two
  structure warnings. A second render wrote the same reference report.

## Handoff

`make plan-status` before this task left the plan: 67 tasks (agent 42, human 25), 30 waiting
on open Requires. After: 66 tasks (agent 41, human 25), 30 still waiting because both
downstream tasks have other open Requires. The priority agent task is
`book-corpus-projection`. The priority human task is `book-pilot-reference-transcription`.
`book-synthetic-book-end-to-end` and `scanned-book-recovery-implementation-audit` no longer
require this task. The synthetic book names the cross-references topic as a dependency. This
task added no plan item. The full 1,067-page recognition remains
`book-full-run-on-the-target-scan`. The index of this book is not in the 50-page slice.
