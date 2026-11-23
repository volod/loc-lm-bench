# book-vault-rendering-and-navigation

## Work contract

```markdown
#### book-vault-rendering-and-navigation

With a ledger, markers, and a structure, the reading vault can be rendered: the deliverable a person
opens in Obsidian. Every choice here is visible to the reader, and every name is a link someone else
will hold, so the output has to be both readable and stable.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: the inferred structure
  ([structure](current/book-recovery/structure-and-rendering.md)).
- User-visible outcome: a book directory with `00_Table_of_Contents.md`, `content/`, `assets/`,
  `review/`, `metadata/`, and `source/`; content split by chapter and section, each file opening and
  closing with links to the contents, the previous file, the next file, and its source PDF page
  range; formulas as display LaTeX carrying their equation numbers; tables as Markdown or HTML;
  figure embeds at their reading position; uncertainty markers in place; YAML front matter only for
  fields Obsidian or retrieval uses (book, hierarchy ids, PDF and printed page ranges, review
  counts).
- Scope boundary: in scope -- the resolved text of every block, never a raw reading picked here;
  segmentation that splits an oversized section into ordered page-range parts at block boundaries,
  never inside a formula, table, figure-caption pair, or paragraph unless the paragraph alone
  exceeds the cap; deterministic ASCII file names and block anchors, with the anchor of a numbered
  target (section, figure, table, equation) derived from its printed number so it survives an
  engine change that renumbers block ordinals, and an anchor at the first block of every PDF page; a
  page whose recovery failed rendered as a page-level marker with its PDF link, never left out; a
  figure block rendered as a crop of its region under the spec's asset naming (PDF page, in-page
  ordinal, kind), so `book-figure-and-caption-assets` changes what a crop holds, not what it is
  called; the PDF-page link form verified against the documentation and release notes of the
  recorded target Obsidian version (a `#page=<n>` fragment on a PDF link is the candidate), plus a
  configurable rendered-page fallback under `source/pages/`; links valid under each source
  deployment mode; atomic publish to `--output` that removes stale files. Out of scope --
  cross-reference linking, and the quality of formula, table, and figure recovery (this task renders
  whatever the ledger holds).
- Data and artifact paths: `src/llb/prep/book/render/`; `book/<book-slug>/` inside the run
  directory; the operator-chosen `--output` vault path.
- Execution path: `make book-ocr-render`; CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; a re-render is byte-identical; an oversized fixture section
  splits only at permitted boundaries, with a formula and a figure-caption pair at the would-be cut;
  a parameter that does not touch structure leaves every file name unchanged; a fixture ledger
  rebuilt with shifted block ordinals keeps every numbered-target anchor; a failed fixture page
  appears as its marker; front matter parses as YAML; every ledger block appears in exactly one
  file, in reading order, with no loss and no duplication.
- Documentation target: `docs/impl/current/book-recovery/structure-and-rendering.md` and the usage
  guide `docs/guides/data-prep/book-ocr.md`.
```

## Implementation and decisions

`llb book-ocr render` reads one run's document structure and resolved text and writes an Obsidian
vault. It is a command, not a stage. A page with no resolved text is one marker and a PDF link.
The resolved text is what the note shows. A raw reading is not chosen here.

Content notes follow the hierarchy. A node longer than 12,000 resolved characters splits into
ordered page-range parts at a block boundary. A formula, a table, and a figure with its caption
stay whole. A paragraph splits only when that paragraph alone passes the cap. A node that
resumes after a child is a later file, reached by previous and next, and omitted from the
contents. The contents lists the node's opening span, including the parts the cap created.

Numbered anchors use the printed number (`^ch-`, `^sec-`, `^sub-`, `^fig-`, `^tbl-`, `^eq-`).
The first block of a PDF page carries `^pNNNN`. A block of kind `figure` is a tight crop named
by PDF page, in-page ordinal, and kind. Review crops keep the queue's 16-pixel padding.

The page link is `#page=<n>`, counted from 1. Obsidian 0.10.8 (2021-01-11) introduced
`[[file.pdf#page=3]]`. The 1.15 help for PDF files and for embeds still documents it.
`metadata/render.json` records version 1.15 and the fragment name `page`.

Body lines in one column join. A footnote, a column break, an indent after a sentence, and the
column's Table/Fig/Eq index line start a paragraph. A word hyphenated at the column edge is
joined, and the rest of the right column stays in reading order. Sibling sections of equal depth
are owned by the later heading, so a subsection that shares its parent's page still receives its
own note.

Publish is atomic. The default directory is `book/<book-slug>/` inside the run. `symlink` is the
default source policy. Page images under `source/pages/` are off unless requested. Either switch
leaves content file names unchanged.

## Acceptance evidence

Status: accepted for this task's gates.

- **Checks.** On 2026-11-23, `make ci`'s checks passed: format, ruff, mypy (1,673 files), the
  complexity gate, shell lint, spec-plan integrity, the toolchain check, the model roster, and
  artifact contracts.
- **The synthetic book.** `tests/llb/prep/book/test_render.py`. Every resolved block appears once,
  in reading order. A failed page is its marker, and the raw line is absent. An unresolved span
  keeps the review marker. Front matter parses as YAML. Re-render is byte-identical and drops a
  stale file. Source policy and page fallback leave the content file names unchanged, and each
  policy's links resolve. An oversized section splits on a paragraph boundary while a formula and
  a figure-caption pair at the cut stay whole. Shifted block ordinals keep the numbered anchors
  and the file names. A column hyphen joins, and the footnote and index line stay separate. The
  command exits 2 when the structure file is missing.
- **The scan.** The counts are in
  [the reading vault](../current/book-recovery/structure-and-rendering.md#reading-vault). On
  2026-11-23, CPU only, RTX PRO 3000 Blackwell laptop (12 GB). PDF pages 1-40 and 290-299 of the
  1,067-page scan, the ledgers from CPU RapidOCR 3.10.0. 52 content notes, no failed page, 238
  review crops, no figure crops. A second render matched the first byte for byte.
- **Markdown.** `make lint-md` passed, including link and citation checks.
- **`make ci` pytest on this host.** 5,047 passed, 53 skipped, 50 deselected, 50 failed, 4
  errors. Forty-nine failures and the four errors are missing optional extras: `faiss`
  (`[rag]`), `langgraph` (`[eval]`), `splink` (linkage), and `rdflib`. The other failure is the
  survey hang fixture: the sandboxed run could not signal the sleeper. That test passed when run
  outside the sandbox. The vault suite passed inside the same pytest run. The checks above had
  already passed.

## Handoff

`make plan-status` immediately before this task left the plan: 69 tasks (agent 44, human 25).
After: 68 (agent 43, human 25). `scanned-book-recovery` lost this task and gained none. The
priority agent task is `book-vault-validation-and-report`. The priority human task is still
`book-pilot-reference-transcription`. Dropping this id released `Requires` on
`book-vault-validation-and-report`, `book-corpus-projection`, `book-formula-and-table-recovery`,
`book-figure-and-caption-assets`, and `scanned-book-recovery-implementation-audit`. Nothing from
this task needed a new plan item.
