# Graph-vs-FAISS retrieval comparison -- how to run it

This is the operator manual for the GraphRAG backend verification that answers one question:
**on a given corpus, does the GraphRAG backend beat flat vector (FAISS) retrieval?**
It scores `recall@k` / `MRR` for `{faiss, graph/local_khop, graph/global_community}` over
the SAME gold set, by the source-span metric, so the three are directly comparable
(the manifest already records backend + strategy). The *why* of GraphRAG lives in
[`current.md`](../../impl/current.md) (GraphRAG backend); this page is the *how*,
plus the first real-host result.

Answer-quality comparison (the model's answers, not just retrieval) rides the normal
`run-eval --retrieval-backend ...` path because it needs a model; this tool isolates the
model-independent **retrieval** signal, which is CI-provable from fakes.

## At a glance

    1. build the FAISS baseline   make build-index CORPUS=<corpus-dir>
    2. build the graph store      llb build-graph --extract-model <capable-model> ...
                                  [needs a local endpoint; disable thinking on reasoning models]
    3. compare                    make compare-retrieval GOLDSET=<goldset> RAG_K=10
                                  [recall@k + MRR per backend, same source-span metric]
    4. sweep the fusion weight    make compare-graph-fusion CONFIG=<cfg> GOLDSET=<goldset>
                                  [optional; multi-hop slice + all-spans@k + intervals]

The human decision sits in step 2 (pick a capable extraction model and the right serving knobs)
and in reading step 3: GraphRAG pays off on multi-hop narrative corpora, not single-span factoid
lookup -- see the [reference result](#reference-factoid-corpus-result) before concluding.

## The commands

| Command | What it does | Needs |
| --- | --- | --- |
| `llb build-graph` | Builds the GraphRAG store (node/edge JSONL + communities) from an ontology-assisted drafting extraction over the corpus. | a local endpoint when extracting fresh |
| `llb compare-retrieval` | Scores recall@k / MRR for every BUILT backend on one gold set; skips a backend whose store is absent. | a built FAISS index and/or graph store |
| `llb compare-graph-fusion` | Sweeps the fused graph weight and decides it on the multi-hop slice, with multi-span metrics and paired intervals. | both stores built, plus multi-hop-labeled gold items |

`make compare-retrieval GOLDSET=... RAG_K=10` and `make compare-graph-fusion CONFIG=... GOLDSET=...`
wrap the last two.

## Step 1 -- build the FAISS index (the baseline)

    llb build-index --corpus-root samples/goldsets/ua_squad_postedited_v1/corpus
    # or: make build-index CORPUS=samples/goldsets/ua_squad_postedited_v1/corpus

## Step 2 -- build the graph from a real extraction

The graph REUSES the ontology-assisted drafting extraction (entities + SRO facts -> nodes + edges).
Extracting fresh needs a local endpoint. **Use a capable instruction model** -- a tiny model
(llama3.2:3b) is fine only for a smoke test; it does not reliably emit the structured JSON
the extractor parses.

### Reasoning models need their thinking disabled

The calibrated gemma4 family are **reasoning** models: left alone they spend the whole output-token
budget on hidden thinking and return empty structured output. Disable it and raise the budget.

**Ollama (recommended on a 16 GB card -- fastest here):**

    ollama pull gemma4:26b
    llb build-graph \
      --corpus-root samples/goldsets/ua_squad_postedited_v1/corpus \
      --extract-model gemma4:26b --extract-no-think --extract-max-tokens 4096

`--extract-no-think` is honored only by Ollama's **native** `/api/chat` (`think:false`); the
OpenAI-compatible `/v1` layer ignores it, so the endpoint adapter routes the think-disabled case to
the native API automatically.

**vLLM (point the extractor at a served HF checkpoint):**

    VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve <hf-quant> --port 8000 \
      --served-model-name m --gpu-memory-utilization 0.92 --max-model-len 4096 \
      --cpu-offload-gb <n> --enforce-eager
    llb build-graph --corpus-root <corpus> \
      --extract-model m --extract-base-url http://localhost:8000/v1 --extract-max-tokens 2048

`VLLM_USE_FLASHINFER_SAMPLER=0` avoids the flashinfer-sampler import mismatch on this host. vLLM
serves a clean `/v1` (gemma4-it emits JSON directly there -- no `--extract-no-think` needed).

### Throughput note (16 GB host, 250-doc corpus)

A model that does **not** fit fully in 16 GB must offload weights. **Ollama's llama.cpp offload
out-throughputs vLLM's `--cpu-offload-gb`** here: measured on the committed corpus,

- Ollama `gemma4:26b` (dense, q4): ~32 s/doc at 3 concurrent workers -> ~2.5 h.
- vLLM `gemma-4-26B-A4B` w4a16 (MoE, ~5 GB offload): ~49 s/doc at 4 workers -> ~3.4 h.
- vLLM `gemma-4-31B` w4a16 (23 GB, ~12 GB offload): <5 tok/s -> impractical (~14-40 h).

vLLM's per-step weight streaming over PCIe dominates when a large share is offloaded; concurrency
helps (vLLM continuous-batches), but a model that fits fully in VRAM is the only way to get its
native speed. Pick the largest model that **fits**, or accept the offload penalty.

## Step 3 -- compare

    llb compare-retrieval \
      --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl --k 10 \
      --out compare.json

It prints an ASCII table and (with `--out`) writes the JSON report. A backend whose store is not
built is skipped with a log line, so you can compare whatever is present.

## Step 4 (optional) -- sweep the fusion weight on multi-hop questions

`compare-retrieval` ranks backends at one graph weight. When the gold set has multi-hop items
(drafted with `--multi-hop`, labeled in `needle_items.jsonl`), sweep the weight instead:

    llb compare-graph-fusion --config <run-config.yaml> --k 10 \
      --graph-weights 0,0.1,0.2,0.3,0.5,0.7,1.0 --out-dir <artifact-dir>
    # or: make compare-graph-fusion CONFIG=<cfg> GOLDSET=<goldset> GRAPH_WEIGHTS=0,0.3,1.0

It writes `report.md` / `comparison.json` with, per graph weight and strategy, `recall@k` beside
`all-spans@k` (did the context carry EVERY hop, not just one), paired bootstrap intervals, the
item-level win/loss ledger, and an adopt / inconclusive / reject verdict for the multi-hop slice.
Read `all-spans@k` first: a healthy `recall@k` on multi-hop questions usually means one hop was
retrieved and the other was not. The measured host result is in
[GraphRAG](../../impl/current/graphrag-backend.md#graph-vector-fusion-evidence).

## Reference factoid-corpus result

Graph built from a fresh `gemma4:26b` extraction (Ollama, think-disabled) over the committed
`ua_squad_postedited_v1` corpus: 250 docs -> **2396 nodes, 558 edges, 1839 communities** (21 docs
yielded no extraction). Comparison over the 250-item gold set:

| backend | recall@10 | MRR |
| --- | --- | --- |
| **faiss** | **0.980** | **0.847** |
| graph/local_khop | 0.292 | 0.091 |
| graph/global_community | 0.284 | 0.229 |

**Verdict: flat vector retrieval strongly beats GraphRAG on this corpus.** SQuAD is factoid QA over
short, weakly-connected paragraphs, so the LLM extraction yields a sparse graph (0.23 edges/node)
and the answer span is often not captured as an entity mention or edge evidence -- so the graph
simply has nothing to return for most questions. `local_khop`'s MRR (0.091) is far below its recall
(0.292): when it does cover the answer, the span ranks low behind many node mentions. GraphRAG is
expected to pay off on **multi-hop / narrative** corpora (connect-these-facts, corpus-level theme
questions), not single-span factoid lookup; this run quantifies that the committed factoid set is
**not** such a corpus. The committed gold set is already verified, so this scoring needs no further
data gate.
