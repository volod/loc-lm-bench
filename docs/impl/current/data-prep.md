# Data Prep

Data prep turns source documents into benchmarkable, verified records. Keep the distinction clear:
automated drafting and cross-checking can prepare evidence, but only reviewed `verified: true`
items score models.

This page is the AREA INDEX: each stage lives in its own page under [`data-prep/`](data-prep/),
in the order a corpus travels through them.

## From documents to a scored split

| Stage | Page | What it answers |
| --- | --- | --- |
| Contract | [Gold item contract and splits](data-prep/gold-contract.md) | What one gold item must carry, how splits validate, and the committed fixture |
| Import | [Ingestion](data-prep/ingestion-import.md) | Grounded-JSONL import into a draft bundle, and external-draft curation (merge / dedup / filter, including intra-document repeated blocks) |
| Import | [Mixed-corpus ingestion and review slices](data-prep/ingestion-corpora.md) | Mixed txt/md/pdf ingestion, widening a multi-hop review slice, and yield-max empirical acceptance |
| Import | [Acquired-corpus provenance fields](data-prep/acquired-provenance.md) | Which capture of which source produced a document: the seven projection-sidecar fields, where they travel, and what stays unchanged for a corpus carrying none |
| Draft | [Chain-of-questions artifacts](data-prep/chain-of-questions.md) | Multi-hop chain drafting and the complete chain-goldset workflow |
| Draft | [Drafting lanes and resumable extraction](data-prep/drafting-lanes.md) | Interrupt-safe drafting, the frontier ontology lane, and the sequential local Qwen/Gemma comparison |
| Verify | [Verification gate and judge calibration](data-prep/verification-gate.md) | The human gate, experiment-derived acceptance thresholds, reviewer throughput tooling, rejection feedback, multi-annotator adjudication, and judge calibration |
| Hand off | [Chunking and query glossary](data-prep/chunking-and-glossary.md) | What data prep hands the retrieval side |

## Corpus hygiene

| Page | What it answers |
| --- | --- |
| [Conflict detection](data-prep/conflict-detection.md) | Finding contradictory passages: effort tiers, relation vocabulary, calibrated rank cutoffs, encoder anisotropy, and why the rank cutoff is not a false-positive rate |
| [Measured claim-tier precision](data-prep/conflict-claim-precision.md) | What share of the returned candidate list survives adjudication: the two-way clustered bound, the free budget sweep, the two-tier frozen probe that gates which adjudicators may be quoted, and the optional cross-encoder claim prefilter |
| [Decision groups and their counts](data-prep/conflict-decision-groups.md) | How many decisions a row count is: the distinct-unit census, the two grouping rules and their range, `to decide` versus `to review`, policy projection, and the per-stage split of lost orderable pairs |
| [What a bundle can answer alone](data-prep/conflict-bundle-record.md) | Which questions a finished audit re-reads from its own record without the store -- the stage, the per-document exclusion reason and its recovery floor, a smaller candidate budget -- the two per-chunk readings the record refuses, what the record's own form costs per document, and the store identity plus portable location it records instead of a second copy of the store's manifest |
| [Independent-null research](data-prep/conflict-null-research.md) | Three generations of candidate nulls, why each failed, the control-bank size a usable tail would need, and the measured claim-tier precision that replaces the missing rate |
| [Closing the independent-null question](data-prep/conflict-null-closure.md) | The fourth generation: generated in-support controls, cross-encoder relation scoring, the distribution-free unit floor a certified tail needs, and the decision to stop |
| [Conflict resolution](data-prep/conflict-resolution.md) | The overlay and rollback contract, and the CUDA-host and large-corpus evidence behind it |
