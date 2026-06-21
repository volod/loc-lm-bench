# Run the eval skeleton (Milestone 1)

The walking skeleton: retrieve -> generate -> score -> one ranked row + a reproducible
manifest, on one Ollama model. It is **compile-free** -- prebuilt Ollama (which still runs
on your GPU), no vLLM/flash-attn source build -- so the loop is proven before Milestone 2
takes on that build. Module detail is in
[implementation/current.md](../implementation/current.md).

## Prerequisites

`make venv` already installed the eval deps (FAISS, sentence-transformers, langgraph). Then:

    ollama serve                           # in another shell
    make prep-models PREP_BACKEND=ollama   # detect GPU + pull the candidate Ollama tags

`make prep-models` reads `samples/models_uk.yaml`, detects the host GPU, pulls the Ollama
tags, and caches any vLLM (Hugging Face) weights once (`PREP_BACKEND=all` for both;
oversized models are skipped or flagged). Or pull one tag by hand: `ollama pull llama3.2:3b`.

The repository already contains the default 250-item public development gold set and its
matching corpus under `samples/goldsets/ua_squad_postedited_v1/`; no ingestion is required.

## Steps

    make build-index                    # chunk + embed the committed fixture -> FAISS store
    make validate-retrieval RAG_K=10    # recall@10 of the pinned embedding (Premise 4 gate)
    make run-eval MODEL=llama3.2:3b LIMIT=20

`run-eval` logs the retrieval context line + one ranked row and writes the canonical
record under `$DATA_DIR/run-eval/<UTC timestamp>-<run id>/`
(`manifest.json` + `scores.{parquet,jsonl}`). Each invocation gets a new directory, so a
later run cannot overwrite an earlier result.

## Notes

- **Embedding is pinned** (`intfloat/multilingual-e5-base` by default) and validated
  separately. If `validate-retrieval` reports recall@10 below 0.8, retrieval -- not the
  model -- is the bottleneck; record it and treat RAG scores as capped.
- **The judge is gated.** By default it is demoted and the objective reference-correctness
  score ranks alone. Pass `--judge-rho <value>` once you have a calibration result; below
  the 0.6 threshold it stays demoted.
- **Config in one place.** Every knob lives in `RunConfig`; copy `samples/run_config_uk.yaml`
  and pass `--config`. CLI flags override individual fields, and the full config is recorded
  in the manifest for reproducibility. Unknown keys and invalid ranges fail before work starts.
- **Verified items only.** `run-eval` excludes draft gold items where `verified: false`.
- **Public fixture is development-only.** It proves the implementation consistently but does
  not replace evaluation on your private corpus. Use the from-scratch guide for that workflow.
- **Determinism.** `temperature: 0.0` and a fixed `n_shot` keep scoring comparable across
  models.
