# Configuration And Command Path

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

## Configuration

`src/llb/core/config.py` defines `RunConfig`, the typed object that flows through retrieval,
generation, scoring, telemetry, and the manifest. YAML configs and CLI overrides share the same
validation path. Unknown keys and invalid ranges fail before work starts.

`src/llb/core/paths.py` loads `.env`, honors `DATA_DIR`, and resolves relative paths from the project
root instead of the caller's current directory.

## Command Path

```bash
llb prep-models
llb list-models
llb build-index --vector-store faiss
llb validate-retrieval --k 10
llb run-eval --model llama3.2:3b --backend ollama
llb run-eval --config samples/configs/run_config_uk.yaml
llb run-eval --split calibration --worksheet calibration.csv
llb run-eval --score-semantic
llb run-eval --resume .data/run-eval/<timestamp>-<run-id>   # continue an interrupted run
```

Make targets wrap the common path:

```bash
make prep-models
make build-index
make validate-retrieval
make run-eval MODEL=llama3.2:3b BACKEND=ollama LIMIT=20
make compare-context-strategies MODEL=<m> BACKEND=<b> GOLDSET=<gs> CORPUS=<corpus-dir>
```

The Makefile defaults `GOLDSET` and `CORPUS` to the committed fixture so smoke runs do not require
network access or data regeneration.
For the local PDF-corpus Gemma 4 quickstart, see
[`docs/guides/quickstart/quickstart-pdf-corpus.md`](../../../guides/quickstart/quickstart-pdf-corpus.md).
That flow builds
`.data/quickstart-pdf-corpus-rag/llb/rag/` from 19 converted PDFs: 13,211 recursive FAISS chunks
and 768-dimensional E5 embeddings (2026-07-02 build). A 4-document quick draft
(`QUICKSTART_PDF_DRAFT_DOCS=<4 ids>`, 70 unverified items, all citation-valid) scored
`recall@10=0.729`, `MRR=0.531` against that full-corpus index -- a true needle-in-haystack check;
the misses are questions phrased broadly enough that spans from OTHER doctrine documents outrank
the gold span. The PDF draft path now annotates `needle_items.jsonl` with `retrieval_rank` from the
full-corpus store; rows with a non-null rank are the retrieval-unique subset at the configured
top-k, and `pdf_ontology_report.json` records the unique-needle fraction. The matching GraphRAG
store lives under
`.data/quickstart-pdf-corpus-graph/llb/graph/` with 290 nodes, 159 edges, and 139 communities
from the same 4-document draft bundle.

## Standalone Closed-Service Runner

`src/llb/standalone/rag_squad_goldset.py` is a stdlib-only helper for operators who need to score a
closed RAG service outside the normal `llb run-eval` backend path. It reads SQuAD-shaped JSONL,
POSTs each `question` to `RAG_SERVICE_URL`, strips reasoning `<think>` blocks, and streams the
original row plus `predicted_answer`, `error`, `service`, and `latency_s` to an output JSONL file.

```bash
python src/llb/standalone/rag_squad_goldset.py INPUT.jsonl OUTPUT.jsonl --limit 10
```

The wire-format seam is intentionally narrow: edit `build_request()` and `parse_answer()` for the
remote service shape, or set `RAG_SERVICE_URL`, `RAG_SERVICE_NAME`, `RAG_API_KEY`,
`RAG_TIMEOUT_S`, and `RAG_RETRIES` in the environment. The helper is type-checked by the standard
`make ci` mypy pass.

## External Answer Log Scoring

`llb score-external-rag` / `make score-external-rag` reviews a JSONL file that already contains
answers from an external or closed RAG system. It does not launch the local RAG backend. It reads
the gold fields plus an answer field (`llm_answer`, `predicted_answer`, `model_answer`, or
`answer`), computes the same objective answer-correctness signals as `run-eval`, and opens an
interactive human scoring loop. Record state, objective scoring, CSV rendering, and orchestration
live in `src/llb/scoring/external_rag/{records,score,worksheet,run}.py`; aggregation, Markdown
reporting, and source mapping are explicit `external_rag_*` sibling modules. The terminal loop is
in `src/llb/scoring/external_rag_session/`; coverage lives in
`tests/llb/scoring/test_external_rag_score.py`.

The JSONL answer log is the session state. Each edit atomically writes `human_score_0_1`,
`human_decision`, `human_notes`, `human_corrected_answer`, and `human_status` back into the same
file, so partial sessions resume at the first unscored row. `EXTERNAL_RAG_CLEAR=1` / `--clear`
clears those human fields after confirmation. The card shown to the reviewer includes the
question, reference answer, gold source spans, raw answer, scored answer, first returned sources,
and error field.

Final artifacts are written only after all rows have a human score plus decision:

```text
<answered>.csv
<answered>.report.md
```

The CSV is sorted by `review_priority_rank` and includes the JSONL-backed human review fields. The
report records aggregate objective estimates, human decision counts, mean human score, split
estimates, common returned sources, actual scoring parameters, and improvement commands. A trailing
source footer in the answer text is stripped before objective scoring while the raw answer is
preserved in the CSV.

```bash
make score-external-rag EXTERNAL_RAG_ANSWERS=<answered-jsonl>
llb score-external-rag --answers <answered-jsonl> --answer-field predicted_answer
make score-external-rag EXTERNAL_RAG_ANSWERS=<answered-jsonl> EXTERNAL_RAG_CLEAR=1
```

This is an external-system diagnostic, not a certified local leaderboard.

### Source-span audit (external-rag-source-mapping)

When the answer log returns only provider-namespace source records (article ids, titles, URLs),
an operator-supplied mapping sidecar joins them onto benchmark corpus spans so retrieval evidence
can be audited, not only answer text:

```bash
make score-external-rag EXTERNAL_RAG_ANSWERS=<answered-jsonl> EXTERNAL_RAG_SOURCE_MAP=<map.jsonl>
llb score-external-rag --answers <answered-jsonl> --source-map <map.jsonl>
```

The sidecar (`.json` list, `.jsonl`, or `.csv`; lives beside the answer log or under
`$DATA_DIR/external-rag/<system>/`) maps provider keys to corpus locations: each record carries
`doc_id` (required), optional `char_start`/`char_end`, and at least one of `article_id`, `url`,
`article_title` (matched in that precedence order). `src/llb/scoring/external_rag/sources.py`
implements the audit; `tests/llb/scoring/test_external_rag_sources.py` covers it.

- A mapped source WITH a char range is scored by the same source-span metric as local retrieval
  (`llb.rag.retrieval.first_hit_rank` over the returned-source order): a span overlapping the
  item's gold spans is a hit.
- A mapping with only `doc_id` (typically title-keyed) can produce at most a doc-level match,
  flagged `source_hit_weak=true` -- weak evidence, never span proof.
- A returned source with no mapping counts into `source_unmapped_count` -- an audit gap reported
  separately from mapped retrieval misses.

The CSV gains additive columns (`source_hit`, `source_first_hit_rank`, `source_hit_weak`,
`source_mapped_count`, `source_unmapped_count`; absent without `--source-map`), and the report
gains a "Source-span audit" section with span-proof `recall@3` and MRR (weak hits and unmapped
sources reported beside them, never folded in). Without a sidecar the limitation stands: external
retrieval recall needs source records resolvable to corpus `doc_id`, `char_start`, `char_end`.
