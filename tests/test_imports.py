"""Every module must import with only the base + dev deps installed.

Heavy extras (faiss, langgraph, mlflow, sentence-transformers, pynvml, ragas) are absent
in CI, so any accidental top-level import of them would break the lightweight job. This
test imports the whole package surface; it fails loudly if a heavy dep stops being lazy.
"""

import importlib

MODULES = [
    "llb.config",
    "llb.main",
    "llb.rag.chunking",
    "llb.rag.retrieval",
    "llb.rag.embedding",
    "llb.rag.index",
    "llb.rag.store",
    "llb.backends.base",
    "llb.backends.openai_client",
    "llb.backends.ollama",
    "llb.backends.hardware",
    "llb.backends.prepare",
    "llb.backends.planner",
    "llb.backends.vllm",
    "llb.backends.telemetry",
    "llb.eval.graph",
    "llb.scoring.correctness",
    "llb.scoring.judge",
    "llb.scoring.aggregate",
    "llb.tracking.manifest",
    "llb.executor.vram",
    "llb.executor.runner",
]


def test_all_modules_import_under_base_install():
    for name in MODULES:
        importlib.import_module(name)
