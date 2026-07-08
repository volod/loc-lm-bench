# Run RAG Core

This guide runs retrieve -> generate -> score for one local model and writes a reproducible run
bundle. The default path uses the committed verified UA-SQuAD fixture and Ollama.

## At a glance

```text
1. prerequisites   make venv; ollama serve; make prep-models     [one-time setup]
2. build index     make build-index                              [chunks + FAISS store]
3. gate retrieval  make validate-retrieval RAG_K=10              [gate: recall@10 >= 0.8]
4. score a model   make run-eval MODEL=<tag> LIMIT=20            [writes the run bundle]
```

No step needs human review here -- the committed fixture is already verified. The one gate to
respect is retrieval: if `validate-retrieval` fails the recall threshold, retrieval is the
bottleneck and model scores are capped by it, not by the model.

## Prerequisites

```bash
make venv
ollama serve
make prep-models PREP_BACKEND=ollama
```

The repository already contains the default gold set and matching corpus under
`samples/goldsets/ua_squad_postedited_v1/`.

## Steps

```bash
make build-index
make validate-retrieval RAG_K=10
make run-eval MODEL=llama3.2:3b LIMIT=20
```

`run-eval` prints the retrieval context line and one ranked row, then writes:

```text
$DATA_DIR/run-eval/<timestamp>-<run-id>/
  manifest.json
  scores.jsonl
```

Each invocation gets a new directory.

## Choosing a chunking strategy

`build-index` chunks with `recursive` by default. Eight strategies are available
(`fixed | sentence | recursive | markdown | semantic | page | heading | late`; see the
[RAG core](../../impl/current/rag-core.md) chunking section for what each does):

```bash
make build-index CHUNK_STRATEGY=heading
make compare-retrieval CHUNK_STRATEGIES=page,heading,late,markdown,semantic RAG_K=10
```

`compare-retrieval CHUNK_STRATEGIES=...` builds one store per strategy over the same corpus and
pinned embedder (persisted under `$DATA_DIR/llb/rag/<strategy>/`) and ranks them by recall@k /
MRR, so pick the demonstrated winner rather than assuming one. `page` needs the PDF-lane
`*.citations.json` sidecars to differ from `recursive`; `late` is flat-mode only and costs a
whole-document embedding pass. `llb tune --extended-chunkers` adds the three new strategies to
the Optuna search space.

## Notes

- The embedding model is pinned. If `validate-retrieval` is below `recall@10 >= 0.8`, retrieval is
  the bottleneck and RAG scores should be interpreted as capped.
- The judge is gated. Without `JUDGE_RHO`, objective reference correctness ranks alone.
- CLI flags override `RunConfig` fields and the final config is recorded in `manifest.json`.
- Draft items with `verified: false` are excluded from scoring.
- Use the from-scratch gold-set guide before treating a private corpus as benchmark data.
