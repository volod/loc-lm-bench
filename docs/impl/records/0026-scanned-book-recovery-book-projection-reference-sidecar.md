# book-projection-reference-sidecar

## Work contract

```markdown
#### book-projection-reference-sidecar

The vault links every citation the book prints that names one target, and the index note links
every page an index line cites. The corpus projection carries none of it. A retriever over the
projection sees `see Table 4-2` as words, and the table it names is a separate document with no
edge to the sentence that needs it. These references are the one relation set in the book that no
model has to infer. Write them into the projection as data, beside the citation sidecar, so a graph
lane can follow what the book prints
([method survey](../design/graph-construction-methods.md#the-printed-reference-graph)).

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- synthetic book fixtures; no model and no GPU.
- Dependencies: citation and index resolution is
  [cross-references](current/book-recovery/structure-and-rendering.md#cross-references); document
  offsets and the citation sidecar are [corpus projection](current/book-recovery/corpus-projection.md).
- User-visible outcome: `llb book-ocr project` also writes the printed references each projected
  document contains. Each resolved citation is recorded as its cue span in this document and the
  span of the target block in that block's document. Each index line is recorded as its term span
  and the PDF pages its printed pages map to. Each document records its place in the hierarchy. A
  later reader can follow a citation from the sentence to the table, equation, figure, section, or
  page it names without parsing the text again.
- Scope boundary: in scope -- one reference sidecar per projected document under a registered
  contract; the render's catalog and resolvers reused, so the projection and the vault resolve the
  same citations from one grammar; ambiguous keys and mapping gaps recorded as unresolved with no
  target; an index sub-entry carrying its printed parent where the indentation shows one;
  `ingest-corpus` copying the sidecar the way it copies the citation sidecar, and a stale document
  taking it along; the redistribution refusal applied to it on every export path. Out of scope --
  any link the book does not print (a concept, a co-occurrence, a similarity), settling an
  ambiguous key by proximity, and quoting a sentence in any report.
- Data and artifact paths: `src/llb/prep/book/project/` and the resolvers in
  `src/llb/prep/book/render/`; a `<document>.md.references.json` beside each projected document.
- Execution path: the synthetic book in `tests/llb/prep/book/test_project.py` and
  `tests/llb/prep/book/test_reference_links.py` inside `make ci`, CPU-only.
- Acceptance gates: `make ci` green; on the fixture the sidecar names exactly the citations the
  vault links, with the same targets, plus the ambiguous and gap citations the reference report
  counts; every recorded span lands in its document's text at its offsets; a second projection
  writes the same bytes; a `local-only` export refuses the sidecar with its document.
- Documentation target: `docs/impl/current/book-recovery/corpus-projection.md`.
```

## Implementation and decisions

`llb book-ocr project` writes `<document>.md.references.json` under `llb.book-references` 1.0.0.
The scan reuses the vault catalog and the citation resolvers, so a grammar file changes both.
A resolved citation stores the cited token's offsets and the target block's offsets. An
ambiguous key, a mapping gap, and a number with no target block stay unresolved and name no
target. An index line stores the term's offsets, the PDF pages its printed pages map to, and
the parent term's offsets when the line is indented under one. A printed range covers every
page it spans, up to 40, and a longer range keeps its endpoints. Each document stores its
hierarchy path from the book root.

A recognizer sometimes glues an index column into `[{'type': '...', 'content': '...'}]` cells
on one line. `index_segments` reads each cell as its own printed line, and the vault linker
uses the same split, so the sidecar and the vault still name the same citations. A line that
also contains other text stays one line. Parents come only from indentation.

`ingest-corpus` copies the sidecar and deletes it with a stale document. The fingerprint
includes it when the file is present and omits the field when it is absent. A corpus bundle
lists each sidecar. A `local-only` export refuses the document before writing the output.

## Acceptance evidence

Status: `accepted`.

`make ci` on 2026-11-27: 5304 passed, 2 skipped, 50 deselected. `make lint-md` reported no
broken links and no citation or record-name findings. The synthetic book projects a sidecar
whose resolved citations match the vault links, including targets, and whose ambiguous keys
and gap count match the reference report. Every recorded span is the document text at its
offsets. A second projection writes the same bytes. A glued column keeps an indented parent.
A grammar file that cites plates cites `Plate 3` and leaves `Fig. 3` as words. Ingest copies
the sidecar, a removed document takes it along, a second ingest keeps the fingerprint, and a
`local-only` export refuses before the output directory exists. The 25-page pilot reading,
the counts, and what would overturn them are in
[corpus projection](../current/book-recovery/corpus-projection.md#evidence).

## Handoff

`make plan-status` before the plan edit: 70 tasks, agent lane 45, human lane 25, statuses
BLOCKED BY HUMAN 4, CLEAR 31, HUMAN-GATED 21, RUN NEEDED 14, 30 waiting on open Requires,
priority agent task `book-projection-reference-sidecar`. After the edit: 69 tasks, agent lane
44, human lane 25, CLEAR 30, 29 waiting, priority agent task `book-technical-content-audit`.

Removing this task drops its id from `book-synthetic-book-end-to-end`,
`book-retrieval-reading`, `book-reference-graph-reading`, and
`scanned-book-recovery-implementation-audit`. `book-retrieval-reading` no longer waits, which
is why the waiting count fell by one. It still sits behind `book-technical-content-audit` in
the lane. No capability registry row moved. `scanned-book-recovery` still has open work.
