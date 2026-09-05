# Sample Fixtures

Committed sample files are grouped by the workflow that consumes them.

| Directory | Contents | Typical consumers |
| --- | --- | --- |
| `artifact_contracts/` | Version dispatch/refusal cases, a bound dataset manifest, and a package-independent schema validator | `make check-artifact-contracts`, artifact contract tests |
| `artifact_contracts/data_prep/` | A staged corpus and a draft bundle at the current contracts, the pre-contract form of each migrating member, and a future-major refusal | `make check-bundle`, data-prep contract tests |
| `configs/` | Candidate model manifest and run-eval YAML examples | `make list-models`, `make prep-models`, `llb run-eval --config` |
| `benchmarks/` | Small Ukrainian category-suite seeds and catalogs, plus the adversarial answer-gate fixture | `bench-security`, `bench-tooling`, `bench-agentic`, `bench-summarization`, `bench-structured`, `make check-answer-gate`, composite smoke fixtures |
| `corpora/acquired_projection_v1/` | Synthetic acquired-corpus projection with twenty documents, complete sidecars, a revision pair, and a local-only document | acquired-projection round-trip conformance test |
| `data-prep/` | Import and synthetic RAG-item fixtures | `make ingest-squad`, `scripts/gen_rag_items.sh`, data-prep tests |
| `linkage/` | Record table, specification, and reviewer labels for the record-linkage seam | `make link-records`, `make replay-linkage`, linkage tests |
| `goldsets/` | Verified committed gold-set bundles with corpus files | default RAG and quickstart flows |
| `pdf_pages/` | PDF page/citation metadata fixtures | page-aware chunking and metadata tests |
| `query-prep/` | Query glossary and prompt dictionary fixtures | query-prep tests and examples |
| `text_analysis_bundle_uk/` | Text-analysis category bundle | text-analysis benchmark tests |
| `verification/` | Human-review sample manifests and worksheets | verification and composite sample smoke checks |
| `config-example/` | Serving config generator templates | `llb gen-serving-config` |
| `ontology/` | Candidate ontology axiom set (Turtle + typed JSON mirror) | `make validate-ontology-axioms`, `llb build-graph --axioms`, axiom-layer tests |

Runtime outputs, generated drafts, downloaded datasets, and private corpora belong under
`$DATA_DIR`, not under `samples/`.
