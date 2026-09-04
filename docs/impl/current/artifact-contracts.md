# Artifact Contracts

This area owns stable schema identity, version compatibility, physical dataset bindings, and the
portable exports an application can use without importing `llb`. The foundation page records the
mechanism; each migration page records the domain surface moved onto it. Run bundles, evaluation,
and board records are migrated in the remaining `artifact-contracts` plan tasks.

| Page | Owns |
| --- | --- |
| [Foundation and evolution](artifact-contracts/foundation-and-evolution.md) | Strict models, registry dispatch, migrations and refusals, dataset formats, generated JSON Schema and ODCS catalog |
| [Data-prep contracts](artifact-contracts/data-prep-contracts.md) | The corpus, PDF, gold, ontology, external-draft, conflict, linkage, and review families; reading pre-contract files; bundle validation and upgrade; the store-build and review gates |
| [Retrieval and graph contracts](artifact-contracts/retrieval-and-graph-contracts.md) | The chunk, store, graph, prompt-system, and comparison-sidecar families; opaque index bindings and the load-time digest gate; generation validation |
