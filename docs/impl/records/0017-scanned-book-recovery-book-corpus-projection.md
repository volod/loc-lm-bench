# book-corpus-projection

## Work contract

```markdown
#### book-corpus-projection

The vault is for a person. Fed to `ingest-corpus` it would index navigation, front matter, and
markers as content, and passthrough Markdown carries no page citations, so a chunk would lose the
page it came from. Render the book for retrieval from the ledger instead, in the corpus shape this
project already reads.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR
- Dependencies: the citation sidecar and chunk page metadata already carry PDF pages into retrieval
  ([mixed-corpus ingestion](current/data-prep/ingestion-corpora.md)); every export refuses a
  `licence` other than `redistributable` or absent
  ([redistribution gate](current/data-prep/acquired-provenance.md#redistribution-gate)); a
  non-local `source_system` puts the projection in the acquired lane, which refuses changed bytes
  under an existing document identity and asks for a new one with `revision_of`
  ([append-only revisions](current/data-prep/acquired-provenance.md#append-only-document-revisions)).
  The reading vault is a separate render of the same ledger
  ([reading vault](current/book-recovery/structure-and-rendering.md#reading-vault)).
- User-visible outcome: `llb book-ocr project <run-dir> --out <corpus-dir>` writes one corpus
  document per content file -- content only, formulas as LaTeX, tables as rendered, markers in a
  machine form -- with a `*.citations.json` sidecar (PDF page to character spans, block ids and
  regions, and the vault file and anchor) and manifest fields (`language`, `source_system`,
  `payload_digest` set to the source checksum, `licence` set to the operator-declared class), so
  `make ingest-corpus` admits the book and every chunk carries its source PDF pages.
- Scope boundary: in scope -- a projection renderer reading the resolved ledger, not the vault
  Markdown; a required redistribution-class argument with no default; stable document ids so a
  re-projection of an unchanged ledger is a no-op for `refresh-index`; a re-projection whose text
  changed (a resolved marker, an admitted correction) emitting a new document identity with
  `revision_of`, never an in-place rewrite, so a gold span accepted against the earlier text still
  resolves; a `validate-goldset` refusal of a source span overlapping an unresolved marker. Out of
  scope -- drafting and verifying gold items (`book-goldset-verification`) and any retrieval
  measurement.
- Data and artifact paths: `src/llb/prep/book/project/`; the corpus directory the operator names.
- Execution path: CPU-only inside `make ci`; `make book-ocr-project` (with a `##` help description)
  wraps the command, and `make ingest-corpus CORPUS_ROOT=<corpus-dir>` consumes the output
  unchanged.
- Acceptance gates: `make ci` green; a fixture projection ingests through `make ingest-corpus` with
  page metadata on every chunk; sidecar offsets match the document text exactly; a re-projection
  yields identical ids and fingerprints; a re-projection after one overlay edit ingests without
  refusal, retains the earlier version, and a gold span over the earlier text still validates; a
  projected book declared non-redistributable makes the fine-tune export refuse and name it; a gold
  span over a marker is refused; a missing redistribution class refuses.
- Documentation target: `docs/impl/current/book-recovery/corpus-projection.md`.
```

## Implementation and decisions

The projection reads the vault's layout and renders each content file again from the resolved
ledger. Document ids follow the text: the logical key is the source checksum, the structure
node, and that node's part index, and the filename hashes the key and the text. An unchanged
ledger rewrites the same bytes. A changed text is a new file with `revision_of`, and the
earlier file stays. `source_system` is `book-ocr`, so the acquired lane refuses an in-place
byte change. `payload_digest` is the source PDF checksum. The redistribution class is a
required argument.

Text ingest copies a sibling `*.citations.json` and removes it with a stale document, so
staged chunks receive `metadata.pages`. `validate-goldset` refuses a span that overlaps an
`unresolved` or `empty` marker token, or the same range recorded in the sidecar.

The sample scan was projected as `local-only`. Recovered text stays under the data directory
and is not part of this change. The measured counts are in
[corpus projection](../current/book-recovery/corpus-projection.md).

## Acceptance evidence

Status: accepted.

`make plan-status` before the plan edit: 71 tasks (46 agent, 25 human). After: 70 tasks
(45 agent, 25 human). The agent priority moved from this task to
`book-resolved-pipeline-scoring`. `scanned-book-recovery` stays `planned`.

On 2026-11-24, `.venv/bin/python -m pytest tests/llb/prep/book/test_project.py -q` passed
(3 tests). The fixture covers content-only text, exact page spans, a byte-identical
re-projection, one admitted correction that adds a file and keeps the earlier text, a gold
span that still validates, a gold span over the unresolved marker that is refused, a
`local-only` export that names the document, and a missing or unknown licence.

The same date, on the RTX PRO 3000 Blackwell laptop host (12 GB), CPU only,
`make book-ocr-project` over the 50-page RapidOCR 3.10.0 slice (PDF pages 1-40 and 290-299)
with `BOOK_LICENCE=local-only` wrote 50 documents and 0 revisions. A second run left the
bytes unchanged. `make ingest-corpus` admitted 44, skipped 6 under the 500-character floor,
copied 44 citation sidecars, and a second ingest kept the fingerprints. 1,194 fixed chunks of
180 characters each carried a page range, covering all 50 PDF pages.

`make lint-md` passed. The first `make ci` failed in pytest because this venv lacked the
locked `rag`, `eval`, `graph`, `linkage`, and `ontology` extras (`faiss`, `langgraph`,
`splink`, `rdflib`). `make install-extras EXTRAS=rag,eval,graph,linkage,ontology` installed
them on the lock. The second `make ci` passed: 5,187 tests, 7 skipped, 50 deselected.
`git diff --check` is clean.

## Handoff

Nothing remains on this task. The projected scan stays in the data directory as `local-only`
and is not part of the tree.
