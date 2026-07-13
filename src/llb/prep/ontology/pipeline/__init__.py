"""Stage 7 -- orchestrate the ontology-assisted gold-set drafting pipeline.

Runs the grained stages in order:

    1 inventory -> 2 extract -> 3 induce ontology -> 4 sample coverage
    -> 5 draft QA -> 6 ground/dedup/reject -> 7 emit bundle

and writes a self-contained, traceable bundle under `$DATA_DIR/prepare-goldset/<timestamp>/`:
the `verified=false` canonical drafts, a copy of the corpus they index (so the validator runs
on the bundle), the induced ontology, the per-document extraction, and a provenance record
linking ontology / extraction / endpoint / prompt / model / cost / document hashes. Nothing is
verified -- a frontier cross-check and a human sample-verify (human verification gate) gate any scoring.

`complete` and `extraction_adapter` are injectable, so the whole flow is unit-tested with a
fake endpoint and never needs a server or a provider key.

Submodules (import from the specific one you need -- there is no re-export surface): `settings`
(the DraftSettings/PipelineResult data objects), `journaling` (resumable bundle setup), `stages`
(the seed/draft/graph/dedup stages), `bundle` (emit + provenance), and `run` (the `draft_goldset`
orchestrator).
"""
