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

The implementation is split into `settings` (the DraftSettings/PipelineResult data objects),
`journaling` (resumable bundle setup), `stages` (the seed/draft/graph/dedup stages), `bundle`
(emit + provenance), and `run` (the `draft_goldset` orchestrator); the public API is re-exported
here so callers keep importing `llb.prep.ontology.pipeline`.
"""

from llb.prep.ontology.pipeline.bundle import _log_calibration_gates
from llb.prep.ontology.pipeline.journaling import default_out_dir, load_journal_meta
from llb.prep.ontology.pipeline.run import draft_goldset
from llb.prep.ontology.pipeline.settings import DraftSettings, PipelineResult

__all__ = [
    "DraftSettings",
    "PipelineResult",
    "_log_calibration_gates",
    "default_out_dir",
    "draft_goldset",
    "load_journal_meta",
]
