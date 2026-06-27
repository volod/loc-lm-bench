# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every line is open work. What already exists (behavior + results), and the operator
workflows an operator re-runs as needed (new gold set, cross-check, sample-verify, calibration, the
verified-data gate, graph-vs-FAISS comparison), live in [`current.md`](current.md); the spec (source
of truth) is [`docs/design/spec.md`](../design/spec.md).

The one open workstream is **Milestone 7** (extended + deferred + forward-verification). Its
sequence number is a stable identifier (AGENTS.md); it appears only while it has open work. The
extended-agentic harness comparison, the judge-diagnostic + smoke verification, the RAG prompt-system
generation lane, and the multi-vector-store adapters live in
[`current/milestone-7-extended-workflows.md`](current/milestone-7-extended-workflows.md) and
[`current/milestone-7-platform-matrix.md`](current/milestone-7-platform-matrix.md); only the
host-only validation residuals below remain open.

---

## Milestone 7 -- Remaining host-only validation residuals

Real-dependency / real-host validation for paths whose seam, adaptation, and scoring the lightweight
CI already covers through fakes, but whose true third-party API behavior the CI cannot exercise. Only
the live integration is open.

### M7.4r Validate the real Chroma/Qdrant/LanceDB adapters + one gold-set comparison

The vector-store seam and per-backend score conversion are tested; the live client APIs are not.
Install `[rag-chroma]` / `[rag-qdrant]` / `[rag-lancedb]` on a host and, for each backend:
`build-index --vector-store <backend>` over the committed gold-set corpus, confirm the persisted
store reloads and `validate-retrieval` matches the FAISS recall@k within noise, then run one
`compare-vector-stores --backends faiss,chroma,qdrant,lancedb --goldset
samples/goldsets/ua_squad_postedited_v1/goldset.jsonl` and record the source-span recall@k / MRR
table. The query/score APIs are version-sensitive (Qdrant `search` vs `query_points`, LanceDB
`.metric("cosine")`, Chroma `query` distances) -- pin working versions and fix any adapter
`_index` / `_search_row` that the live API rejects.

### M7.3r Extend the prompt-system board axis to the baseline RAG `run-eval` lane

The prompt-system package is harness-compatible (`wrap_complete`) and the agentic lane records
`config.prompt_system` with a `prompt_system_comparison` board axis. The baseline RAG `run-eval`
path does not yet record the prompt-system provenance. Add a `--prompt-system <id>` (and an optional
`--prompt-package <run_dir>/<id>`) to `run-eval` that wraps the generation `complete` with the
selected `PromptPackage`, records `prompt_system_provenance` (id + corpus/mapping/template digests +
tokenizer + context budget) in the run manifest, and surfaces a prompt-system comparison view over
`run-eval` bundles (mirroring the agentic axis) so an operator can answer whether the additional
system prompt helps a model on the grounded-answer RAG task, not only the agentic task.
