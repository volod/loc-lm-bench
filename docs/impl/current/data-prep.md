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
| Draft | [Chain-of-questions artifacts](data-prep/chain-of-questions.md) | Multi-hop chain drafting and the complete chain-goldset workflow |
| Draft | [Drafting lanes and resumable extraction](data-prep/drafting-lanes.md) | Interrupt-safe drafting, the frontier ontology lane, and the sequential local Qwen/Gemma comparison |
| Verify | [Verification gate and judge calibration](data-prep/verification-gate.md) | The human gate, experiment-derived acceptance thresholds, reviewer throughput tooling, rejection feedback, multi-annotator adjudication, and judge calibration |
| Hand off | [Chunking and query glossary](data-prep/chunking-and-glossary.md) | What data prep hands the retrieval side |

## Corpus hygiene

| Page | What it answers |
| --- | --- |
| [Conflict detection](data-prep/conflict-detection.md) | Finding contradictory passages: effort tiers, relation vocabulary, calibrated rank cutoffs, encoder anisotropy, and why the rank cutoff is not a false-positive rate |
| [Independent-null research](data-prep/conflict-null-research.md) | Three generations of candidate nulls, why each failed, the control-bank size a usable tail would need, and the measured claim-tier precision that replaces the missing rate |
| [Conflict resolution](data-prep/conflict-resolution.md) | The overlay and rollback contract, and the CUDA-host and large-corpus evidence behind it |
