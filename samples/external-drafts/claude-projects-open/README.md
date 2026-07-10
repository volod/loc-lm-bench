# External-draft fixture (Artifact B, open data)

Committed open-data fixture for the grounded-JSONL import lane
(`llb import-external-draft`; see [data prep](../../../docs/impl/current/data-prep.md)
grounded-JSONL import and the
[external-draft contract](../../../docs/design/external-draft-contract.md) Artifact B).

- `grounded_draft.jsonl` -- five contract Artifact B rows (`quote` + `source_doc_id`) drafted, by
  construction, against the committed open corpus `samples/goldsets/ip_regulation_uk/corpus/`.
  Three rows ground cleanly (one is whitespace-flattened to exercise re-grounding/repair); two are
  intentionally droppable (a paraphrased non-verbatim quote and an unknown `source_doc_id`).
- `external_provenance.json` -- the required data-classification sidecar
  (`data_classification: "open"`); a missing or non-open sidecar aborts the import before writing.

Import against the sibling IP-regulation corpus:

```bash
make import-external-draft \
  ARTIFACT=samples/external-drafts/claude-projects-open/grounded_draft.jsonl \
  CORPUS=samples/goldsets/ip_regulation_uk/corpus \
  SIDECAR=samples/external-drafts/claude-projects-open/external_provenance.json
```

Used by `tests/llb/prep/test_external_draft.py::test_committed_fixture_imports` (3 kept, 2 dropped).
