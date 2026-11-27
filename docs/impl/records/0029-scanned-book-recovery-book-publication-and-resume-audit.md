# book-publication-and-resume-audit

## Work contract

```markdown
#### book-publication-and-resume-audit

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: [vault validation](current/book-recovery/structure-and-rendering.md#vault-validation).
  The eleven-page chain is [full-run rehearsal](current/book-recovery/full-run-evidence.md).
- User-visible outcome: a go or repair decision before committing a full handbook to publication,
  based on source fidelity, resource use, restart behaviour and evidence links in both outputs.
- Scope boundary: exercise the normal commands on a long generated book with page fallbacks,
  a failed page, repeated folios, an interruption during publish, and a reviewer edit.
- Data and artifact paths: `tests/llb/prep/book/`, temporary vault and corpus trees.
- Execution path: CPU fixtures in `make ci`; compare clean and resumed outputs and inspect the
  mapping from a retrieved chunk back to the source pixels.
- Acceptance gates: bound decoded page memory independent of book length; verify exact source
  pixels in fallbacks, consistent publication after retry, byte-identical clean and resumed
  renderings, and a new revision only for the edited document. A broken queue, ambiguous printed
  link or unresolved gold span must surface at its own gate.
- Documentation target: `docs/impl/current/book-recovery/full-run-evidence.md`.
```

## Implementation and decisions

Vault publication stages the tree under a reserved sibling name and serializes publishers in
the destination's parent directory. Where supported, directory exchange exposes the new tree
atomically. The fallback retains the previous tree and restores it on retry. Run resume ignores
staging directories when it revalidates published vaults. The corpus writes sidecars before
it exposes a new Markdown file to ingestion.

Structure keeps repeated folio observations that the monotone fit omits. Vault links, index
entries, and the printed-reference sidecars treat those folios as ambiguous. Both outputs use
the same section sizing even when their PDF links differ in length. Split paragraphs project
only their own text. A page that continues into another note has a local anchor for corpus
citations in that note.

## Acceptance evidence

Accepted.

On 2026-11-27 on an ASUS V16 V3607VH with an Intel Core 5 210H, the generated-book gate drove
`book-ocr run`, resume, review queue, structure, render, validate, project, and `ingest-corpus`
with injected CPU engines and a 1,000-character cap. The
[publication audit](../current/book-recovery/full-run-evidence.md#publication-and-resume-audit)
records the results.

- The 64-page book decoded over four times as many pages as the 11-page control and both
  stayed within two live decoded page arrays. The all-page fallback comparison matched the
  cached render bytes for all 64 pages, and a direct comparison on three pages matched the
  PDF pixels, including the RGB page. The copied PDF matched its source bytes.
- The interrupted run kept completed stage records. Vault publication retry gave byte-identical
  output at the same destination. The no-exchange fallback restored the old vault before a
  retry that itself failed, and a subsequent retry completed. Interrupted projection exposed
  only complete documents. Across independent clean and resumed runs, the output trees matched
  after excluding the run directory and acquisition identity and timestamp.
- Vault validation reported a stale queue and a marker mismatch when the open entries were
  removed. Printed folio 18 remained unlinked with an ambiguous outcome in the reference
  report and corpus sidecar. Gold validation rejected the unresolved span. The ingested lexical
  hit on PDF page 64 resolved to its block and the continuation note, and the cited region
  contained the planted RGB pixels.
- A reviewer edit appended one document linked to its predecessor. Existing texts and
  citation offsets stayed intact. One incoming printed-reference sidecar updated its target.
  An immediate re-projection created zero further revisions and wrote identical bytes.

The diagnostic on the supplied *Space Mission Engineering: The New SMAD* scan ran
`make book-ocr` over all 1,067 pages with the render stage. Resume used all 1,067 cached renders.
The four-page RapidOCR 3.10.0 run on PDF pages 1, 30, 31, and 1,067 reused all 24 stage results
on resume. The `book-ocr-review-queue`, `book-ocr-structure`, `book-ocr-render`,
`book-ocr-validate`, `book-ocr-project`, and `ingest-corpus` Make targets produced a validated
local-only diagnostic with 3 review entries and 1 structure warning. Ingest retained provenance
on all 60 chunks and all 296 citation rows resolved to a vault anchor. Before the repair, 54
of those rows on PDF page 30 led to a missing anchor. The
[target scan diagnostic](../current/book-recovery/full-run-evidence.md#target-scan-diagnostic)
records the timing, layout comparison, and reference counts.

The focused publication, edge, and structure tests passed. `make generate-artifact-contracts`
regenerated the structure schema and passed external JSON Schema and ODCS validation.
`make format`, `make lint-md`, and `git diff --check` passed. The sandboxed `make ci` passed
5,322 tests and failed four download-guard tests at socket creation. With local socket access,
`make ci` passed all static gates and 5,326 tests, with 2 skipped and 50 deselected.

## Handoff

`make plan-status` went from 67 tasks (42 agent and 25 human) to 66 (41 agent and 25 human).
Only the scanned-book recovery group moved. `book-full-run-on-the-target-scan` and the optional
closing audit drop this dependency. The capability remains planned. The CPU diagnostic leaves
full recognition and provisional profile qualification with `book-full-run-on-the-target-scan`.
