"""Typer application root for loc-lm-bench.

Commands by milestone:
  build-index / validate-retrieval / run-eval        M1 skeleton (retrieve -> generate -> score)
  prep-models / list-models / build-vllm             M1/M2 model prep + feasibility + vLLM build
  detect-gpu-vram / gen-serving-config             per-GPU-tier serve + run-eval artifacts
  resolve-models                                     M3.2 pick the backend that can serve a model
  sweep                                              M3.3 isolated cell-per-model sweep (resume)
  tune                                               M3.4 two-stage Optuna (tuning -> final)
  prepare-goldset / prepare-synthetic-corpus         M3.5 frontier data-prep (litellm)
  prepare-goldset-draft                              M4.4 ontology-assisted draft (local/frontier)
  judge-experiment                                   M3.8 local DeepEval UA smoke artifact
  screen-public                                      M3.1 Tier-1 lm-eval-harness-uk screen
  board / mlflow-ui                                  M3.7 Streamlit leaderboard / MLflow UI

Heavy collaborators (FAISS, sentence-transformers, langgraph, optuna, litellm, streamlit, a
running backend) are lazy-imported at call time, so the module imports in the base install.
Config comes from a YAML file (`--config`) with CLI flags overriding individual fields.
"""

import typer

app = typer.Typer(
    add_completion=False,
    rich_markup_mode=None,
    help="loc-lm-bench: local Ukrainian LLM benchmark.",
)
