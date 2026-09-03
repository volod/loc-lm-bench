# Artifact Contracts

This area owns stable schema identity, version compatibility, physical dataset bindings, and the
portable exports an application can use without importing `llb`. The foundation page records the
mechanism; each migration page records the domain surface moved onto it. Retrieval and graph
producers are migrated in the remaining `artifact-contracts` plan tasks.

| Page | Owns |
| --- | --- |
| [Foundation and evolution](artifact-contracts/foundation-and-evolution.md) | Strict models, registry dispatch, migrations and refusals, dataset formats, generated JSON Schema and ODCS catalog |
| [Data-prep contracts](artifact-contracts/data-prep-contracts.md) | The corpus, PDF, gold, ontology, external-draft, conflict, linkage, and review families; reading pre-contract files; bundle validation and upgrade; the store-build and review gates |
