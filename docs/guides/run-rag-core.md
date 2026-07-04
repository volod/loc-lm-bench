# Run RAG Core

This guide runs retrieve -> generate -> score for one local model and writes a reproducible run
bundle. The default path uses the committed verified UA-SQuAD fixture and Ollama.

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

## Notes

- The embedding model is pinned. If `validate-retrieval` is below `recall@10 >= 0.8`, retrieval is
  the bottleneck and RAG scores should be interpreted as capped.
- The judge is gated. Without `JUDGE_RHO`, objective reference correctness ranks alone.
- CLI flags override `RunConfig` fields and the final config is recorded in `manifest.json`.
- Draft items with `verified: false` are excluded from scoring.
- Use the from-scratch gold-set guide before treating a private corpus as benchmark data.
