# GraphRAG

GraphRAG is an alternate retrieval backend selected with `--retrieval-backend graph`. It reuses the
RAG store seam, so generation, scoring, manifests, judge gating, and boards do not need separate
graph-specific code.

FAISS remains the default for factoid QA. GraphRAG is most useful when the corpus has connected
entities, multi-hop facts, or narrative community structure.

## Store Decision

The graph store persists inspectable node and edge JSONL and loads them into DuckDB for query-time
retrieval. DuckDB is used because it is already a Python dependency, supports recursive CTEs, and
keeps the graph runtime local without introducing a separate graph database service.

Community ids are computed offline and stored. Query time only needs DuckDB table queries.

## Modules

`src/llb/graph/model.py`
: Defines `KnowledgeGraph`, `GraphNode`, `GraphEdge`, and `GraphMention`. Mentions and edge
  evidence keep doc ids, offsets, exact text, section titles, ontology type confidence, and
  community ids.

`src/llb/graph/build.py`
: Converts ontology-assisted `DocExtraction` records into nodes and directed edges. Fact endpoints
  that are not known entities become fact-only nodes so grounded evidence is not dropped.

`src/llb/graph/community.py`
: Deterministic label-propagation community assignment. No graph-analytics dependency is needed at
  query time.

`src/llb/graph/retrieval.py`
: Links question terms to graph nodes and serializes subgraphs back into offset-bearing chunk
  records. Linking uses exact alias/name hits plus a conservative Ukrainian stem key for inflected
  forms.

`src/llb/graph/store.py`
: Implements `GraphStore.build`, `save`, `load`, and `retrieve(question, k)`.

`src/llb/graph/summary.py`
: Optional diagnostic community summaries. Summaries are stored separately and are not returned as
  retrieval context because they are abstractive and not span-scored.

`src/llb/rag/compare.py`
: Compares FAISS and graph strategies through the shared `.retrieve` seam.

## Retrieval Strategies

`local_khop`
: Entity-link the question to seed nodes, expand `graph_khop_depth` hops, and serialize node
  mentions plus edge evidence. This is the graph path for connected fact questions.

`global_community`
: Link the question to communities and serialize member nodes and edges from those communities.
  This is the narrative/theme path.

Both strategies return chunk-like records with exact source spans so the normal retrieval metric
applies.

## CLI

```bash
llb build-graph --bundle <prepare-goldset-dir>
llb build-graph --extraction <extraction.jsonl> --corpus-root <dir>
llb build-graph --corpus-root <dir> --extract-model llama3.2:3b
llb build-graph --bundle <dir> --summarize --summarize-model llama3.2:3b
llb validate-retrieval --retrieval-backend graph --retrieval-strategy local_khop
llb compare-retrieval --k 10 --out report.json
llb run-eval --retrieval-backend graph --retrieval-strategy global_community ...
```

`RunConfig` carries `retrieval_backend`, `retrieval_strategy`, and `graph_khop_depth`. These values
are part of the config fingerprint and manifest, so vector and graph runs remain comparable.

## Extraction Inputs

Graph build inputs come from ontology-assisted drafting:

- a full draft bundle with `extraction.jsonl`, `corpus/`, and `ontology.json`;
- an explicit extraction file plus corpus root;
- fresh local extraction over a corpus.

Fresh extraction can disable hidden reasoning with `--extract-no-think`. For Ollama reasoning
models this uses the native `/api/chat` path because the OpenAI-compatible `/v1` path does not
honor the `think` control.

The graph build path has been smoke-tested with
`.data/prepare-goldset/20260628T131439Z-smoke`: it loaded two drafted extractions and wrote a graph
with 19 nodes, 7 edges, and 12 communities under `$DATA_DIR/llb/graph/`.

## Ontology Scope

Graph nodes use the closed 13-type vocabulary in
`docs/design/graph-ontology-schema.md`. The closed vocabulary matters because graph retrieval needs
stable typed nodes and relation caps; allowing a model to invent schema labels would make graph
quality and comparison unstable across runs.
