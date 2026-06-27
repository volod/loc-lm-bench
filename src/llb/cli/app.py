"""Typer application root for loc-lm-bench.

Commands by area:
  build-index / validate-retrieval / run-eval        RAG core (retrieve -> generate -> score)
  prep-models / list-models / build-vllm             RAG/vLLM model prep + feasibility + vLLM build
  detect-gpu-vram / gen-serving-config             per-GPU-tier serve + run-eval artifacts
  resolve-models                                     backend resolver pick the backend that can serve a model
  sweep                                              hard-isolation cell-per-model sweep (resume)
  tune                                               two-stage Optuna (tuning -> final)
  prepare-goldset / prepare-synthetic-corpus         frontier data-prep (litellm)
  prepare-goldset-draft                              ontology-assisted draft (local/frontier)
  judge-experiment                                   local judge calibration DeepEval UA smoke artifact
  screen-public                                      Tier-1 public lm-eval-harness-uk screen
  board / mlflow-ui                                  Streamlit leaderboard / MLflow UI

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
