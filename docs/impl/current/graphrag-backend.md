# GraphRAG backend Current State

## GraphRAG backend -- GraphRAG (knowledge-graph + narrative retrieval) (build COMPLETE)

A graph retrieval backend behind the RAG-store seam, selected with `--retrieval-backend graph`.
FAISS stays the default and is untouched. The BUILD is delivered + unit-tested without a GPU.
**text-analysis sign-off is signed off (2026-06-26):** the 13-type closed node vocabulary + relationship caps + the
GraphRAG scope are accepted, so the ontology/scope is trusted for HEADLINE use -- the signed schema
is [`docs/design/graph-ontology-schema.md`](../../design/graph-ontology-schema.md) (`APPROVED`; node
vocab + caps + scope + a worked example over `samples/corpus/ip_regulation_uk.md`). Real-model graph
HEADLINE numbers ride only the standing human verification gate data gate now, exactly like the category suite categories; the
objective graph boards rank regardless.

**Store choice -- DuckDB (the abandoned Kuzu pick was dropped).** `duckdb` is already a dependency
(now also its own `[graph]` extra) so no new/abandoned dep is added. The graph is persisted as
node/edge JSONL (inspectable, diffable -- the same shape as the FAISS store's chunks) and loaded
into an in-memory DuckDB engine that carries the two graph queries: local k-hop via a recursive CTE
over the (undirected) edge table, and community grouping via `WHERE community_id IN (...)`. The
community ids are detected ONCE offline (so query time needs no graph-analytics dep) -- the
condition under which "DuckDB covers narratives". `duckdb` is lazy-imported, so the base install
still imports.

### Modules -- `llb.graph.*`
- `model.py` -- the RAM-resident `KnowledgeGraph` (`GraphNode` / `GraphEdge` / `GraphMention`).
  Every mention + edge evidence keeps `doc_id` + char offsets + exact text, and carries the
  induced ontology `type`/`confidence`, the containing `section_title`, and the `community_id`.
- `build.py` -- `build_graph(extractions, docs, ontology=None)` REUSES the ontology-assisted drafting `DocExtraction`
  (no second extraction framework): entity mentions -> nodes, SRO facts -> directed edges; a fact
  endpoint that is not a known entity becomes a lightweight fact-only node (no grounded fact is
  dropped). Pure + deterministic; the ontology is induced from the extractions when not supplied,
  so each node carries its type confidence.
- `community.py` -- deterministic, seeded asynchronous label propagation (`detect_communities` /
  `assign_communities`): no graph-analytics dependency, the same corpus always partitions
  identically; an isolated node stays its own community.
- `retrieval.py` -- pure question linking + span-preserving serialization. Lexical entity linking
  keys on entity NAME + aliases (not mention text, which would link unrelated entities sharing a
  relation verb); `serialize_subgraph` renders member node mentions + intra-member edge evidence to
  ranked, span-deduplicated, offset-bearing `ChunkRecord`s. Linking is **morphology-aware**
  (`morph_key`): an exact token match scores full weight, and a shared leading-stem match (first
  `MIN_STEM_LEN` chars -- so the genitive "Франка" links "Франко") scores `STEM_MATCH_WEIGHT`
  below it, so inflected Ukrainian question forms still link while exact hits rank first. Pure +
  deterministic -- no lemmatizer, no embedder (constants in `graph/constants.py`).
- `store.py` -- `GraphStore`: `.build` / `.save` / `.load` + the `.retrieve(question, k)` seam (so
  the eval graph, scoring incl. the gated judge, isolation, and the board are UNCHANGED). The two
  strategies, recorded per run as `retrieval_strategy`:
  - **local_khop** -- entity-link the question to seed nodes, expand `graph_khop_depth` hops with
    the recursive CTE, serialize the subgraph (the multi-hop "connect these facts" path).
  - **global_community** -- map the question to its communities, serialize each community's member
    nodes/edges WITH their offsets (the narrative layer for corpus-level theme/trend questions).
- `summary.py` -- OPTIONAL `summarize_communities(graph, complete)`: an LLM one-paragraph summary
  per sizable community, recorded as a TAGGED DIAGNOSTIC (`community_summaries`, its own file) and
  NEVER returned by `.retrieve` -- the un-grounded abstraction never enters the span metric (the
  same recorded-but-not-ranked discipline as `--score-semantic`). Off by default; reachable from the
  operator CLI via `build-graph --summarize` (see below).
- `rag/compare.py` -- `compare_retrieval(stores, items, k)` scores several backends' `recall@k` /
  `MRR` on the SAME goldset by the one source-span metric, with `format_comparison` (ASCII table)
  and `load_compare_stores(cfg)` (loads `{faiss, graph/local_khop, graph/global_community}`, skipping
  any whose store is not built). Pure -- driven by the `.retrieve` seam, so it is unit-tested with
  fake stores (no FAISS / DuckDB / GPU). Backs the `compare-retrieval` command (below).
- `ingest.py` -- load the ontology-assisted drafting extraction back: `load_bundle` reads a `prepare-goldset` draft
  bundle's `extraction.jsonl` + `corpus/` + `ontology.json`; `load_extractions` reads an explicit
  `extraction.jsonl`.

### Config + CLI + manifest
`RunConfig` gained `retrieval_backend` (`faiss` | `graph`), `retrieval_strategy` (`local_khop` |
`global_community`), and `graph_khop_depth`, plus `graph_dir()` (`$DATA_DIR/llb/graph/`). They ride
in the config fingerprint, so the manifest records backend + strategy and graph-vs-FAISS /
local-vs-global runs are comparable. `run-eval` and `validate-retrieval` take `--retrieval-backend`
/ `--retrieval-strategy`; the runner's `_load_store` picks `GraphStore` vs `RagStore` by backend.

    llb build-graph --bundle <prepare-goldset dir>     # reads extraction.jsonl + corpus/
    llb build-graph --extraction <e.jsonl> --corpus-root <dir>   # explicit ontology-assisted drafting extraction
    llb build-graph --corpus-root <dir> --extract-model llama3.2:3b   # extract fresh (ontology-assisted drafting)
    llb build-graph --bundle <dir> --summarize --summarize-model llama3.2:3b  # + diagnostic summaries
    llb validate-retrieval --retrieval-backend graph --retrieval-strategy local_khop
    llb compare-retrieval --k 10 [--out report.json]   # faiss vs both graph strategies on one goldset
    llb run-eval --retrieval-backend graph --retrieval-strategy global_community ...

`build-graph --summarize` attaches the tagged-diagnostic community summaries via the ontology-assisted drafting local
endpoint adapter (`--summarize-model`, defaulting to `--extract-model`); they persist to
`community_summaries.json` and are NEVER span-scored. `compare-retrieval` reuses the runner's
`_load_store` (varying backend/strategy via `with_overrides`) to rank `{faiss, graph/local_khop,
graph/global_community}` on `recall@k` / `MRR` over the same goldset, skipping any backend whose
store is not built (answer-quality comparison rides `run-eval --retrieval-backend ...`, which needs a
model). Or via make: `make build-graph BUNDLE=<dir>`. The `graph` extra (`duckdb`) is in the default
`make venv` EXTRAS.

### Tests + acceptance
`tests/test_graph.py` (25 tests): construction (offsets/section/confidence carry-through,
fact-endpoint linking, fact-only nodes), deterministic community detection, linking +
serialization, **morphology-aware linking** (`morph_key` collapses inflected forms, an inflected
question links the node, exact match outranks a stem match), both strategies returning
offset-bearing context that scores recall@k=1.0 on the existing span metric, k-hop depth bounding,
save/load round-trip, the tagged-diagnostic summaries (asserting they never appear as retrieved
chunks) + the `build-graph --summarize` no-model guard, ingest round-trip, and the full vertical
through `run_eval` (manifest records backend + strategy). `tests/test_compare_retrieval.py` (5
tests) covers the comparison core from fake stores (recall ranking, MRR tie-break, empty backends,
ASCII output). The DuckDB-engine tests `importorskip("duckdb")` so the lightweight CI install skips
them while the pure-graph logic runs everywhere; `make ci` is green (ruff + mypy strict + `-m "not
slow"`, 671 passed). Smoke-validated end to end: `build-graph` from a bundle ->
`validate-retrieval --retrieval-backend graph` recall@5=1.000, and `compare-retrieval` over a saved
GraphStore renders the ranked table (FAISS skipped when not built).

### GraphRAG backend residuals (engineering)
The three optional/forward GraphRAG backend residuals are delivered + CI-proven from fakes:
1. **Morphology-aware entity linking** -- `morph_key` stem matching in `graph/retrieval.py` (above);
   lifts graph recall on inflected Ukrainian questions without an embedder, behind the pure linking
   seam.
2. **`build-graph --summarize`** -- exposes `summarize_communities` to operators via the ontology-assisted drafting local
   endpoint adapter (above); summaries stay tagged-diagnostic.
3. **`compare-retrieval`** -- the graph-vs-FAISS comparison tool (`rag/compare.py` + the CLI
   command, above), RUN on the real CUDA host over the committed corpus (result below).

### Reasoning-model extraction (build-graph fresh extraction)
Fresh `build-graph` extraction can drive a calibrated reasoning model. `EndpointConfig` gained
`think` (None | True | False) and `chat_once` an `extra_body` passthrough; when `think` is set the
local adapter routes to Ollama's **native `/api/chat`** (`_ollama_native_complete` /
`_native_chat_url`) because the OpenAI `/v1` layer ignores `think` and the model then burns the whole
token budget on hidden reasoning and returns empty JSON. `build-graph` exposes `--extract-base-url`
(point at a vLLM `/v1`), `--extract-max-tokens`, and `--extract-no-think`. Tests:
`tests/test_ontology_draft.py` (`_native_chat_url` mapping, think routes to native + records tokens +
provenance). See [`graph-vs-faiss-comparison.md`](../../guides/graph-vs-faiss-comparison.md) for the
operator flow incl. the Ollama-vs-vLLM throughput finding on the 16 GB host.

### GraphRAG backend verification -- real-host graph-vs-FAISS (2026-06-26, RTX 4060 Ti 16 GB)
Graph built from a fresh `gemma4:26b` extraction (Ollama, reasoning disabled, 3 concurrent workers,
~2.6 h) over the committed `ua_squad_postedited_v1` corpus: 250 docs -> **2396 nodes, 558 edges,
1839 communities** (21 docs yielded no extraction). `compare-retrieval --k 10` over the 250-item
gold set:

| backend | recall@10 | MRR |
| --- | --- | --- |
| **faiss** | **0.980** | **0.847** |
| graph/local_khop | 0.292 | 0.091 |
| graph/global_community | 0.284 | 0.229 |

**Flat vector retrieval strongly beats GraphRAG on this factoid corpus.** SQuAD paragraphs are
short and weakly connected, so the LLM extraction is sparse (0.23 edges/node) and the answer span is
usually not an extracted entity mention / edge evidence -- the graph has nothing to return for most
questions; `local_khop`'s low MRR (0.091 vs recall 0.292) shows the answer ranks behind many node
mentions when it is covered. GraphRAG is expected to pay off on multi-hop / narrative corpora, which
the committed factoid set is not -- this run quantifies that. Throughput finding: on the 16 GB card
Ollama's llama.cpp offload (~32 s/doc) out-throughputs vLLM `--cpu-offload-gb` (MoE 26B ~49 s/doc;
dense 31B w4a16 <5 tok/s) -- a model that fits fully in VRAM is the only way to get native speed.
