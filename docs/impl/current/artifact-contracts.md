# Artifact Contracts

This area owns stable schema identity, version compatibility, physical dataset bindings, and the
portable exports an application can use without importing `llb`. The foundation page records the
mechanism; each migration page records the domain surface moved onto it. The training and
integration surface is migrated in the remaining `artifact-contracts` plan tasks.

| Page | Owns |
| --- | --- |
| [Foundation and evolution](artifact-contracts/foundation-and-evolution.md) | Strict models, registry dispatch, migrations and refusals, dataset formats, generated JSON Schema and ODCS catalog |
| [Data-prep contracts](artifact-contracts/data-prep-contracts.md) | The corpus, PDF, gold, ontology, external-draft, conflict, linkage, and review families; reading pre-contract files; bundle validation and upgrade; the store-build and review gates |
| [Retrieval and graph contracts](artifact-contracts/retrieval-and-graph-contracts.md) | The chunk, store-meta, graph, prompt-system, glossary, and comparison-sidecar families; opaque bindings for indexes and databases; the published store manifest and its digest gate; `check-store` |
| [Run, board, and orchestration contracts](artifact-contracts/run-and-evaluation-contracts.md) | The run-manifest, case-score, benchmark-cell, retrieval, probe, abort, progress, study-sidecar, miss-analysis, agent-profile, and auto-RAG families; the typed additional members a bundle may hold; validation before the atomic rename; refusal before board admission; `check-run` |
