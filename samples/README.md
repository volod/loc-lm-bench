# Sample Fixtures

Committed sample files are grouped by the workflow that consumes them.

| Directory | Contents | Typical consumers |
| --- | --- | --- |
| `configs/` | Candidate model manifest and run-eval YAML examples | `make list-models`, `make prep-models`, `llb run-eval --config` |
| `benchmarks/` | Small Ukrainian category-suite seeds and catalogs | `bench-security`, `bench-tooling`, `bench-agentic`, `bench-summarization`, `bench-structured`, composite smoke fixtures |
| `data-prep/` | Import and synthetic RAG-item fixtures | `make ingest-squad`, `scripts/gen_rag_items.sh`, data-prep tests |
| `goldsets/` | Verified committed gold-set bundles with corpus files | default RAG and quickstart flows |
| `pdf_pages/` | PDF page/citation metadata fixtures | page-aware chunking and metadata tests |
| `query-prep/` | Query glossary and prompt dictionary fixtures | query-prep tests and examples |
| `text_analysis_bundle_uk/` | Text-analysis category bundle | text-analysis benchmark tests |
| `verification/` | Human-review sample manifests and worksheets | verification and composite sample smoke checks |
| `config-example/` | Serving config generator templates | `llb gen-serving-config` |

Runtime outputs, generated drafts, downloaded datasets, and private corpora belong under
`$DATA_DIR`, not under `samples/`.
