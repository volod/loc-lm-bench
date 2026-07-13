"""Every module must import with only the base + dev deps installed.

Heavy extras (faiss, langgraph, mlflow, sentence-transformers, pynvml, deepeval) are absent
in CI, so any accidental top-level import of them would break the lightweight job. This
test imports the whole package surface; it fails loudly if a heavy dep stops being lazy.
"""

import importlib

MODULES = [
    "llb.core.config",
    "llb.core.env",
    "llb.core.runtime",
    "llb.main",
    "llb.rag.chunking",
    "llb.rag.retrieval",
    "llb.rag.embedding",
    "llb.rag.index",
    "llb.rag.store",
    "llb.rag.vector_index",
    "llb.rag.stores.base",
    "llb.rag.stores.chroma",
    "llb.rag.stores.qdrant",
    "llb.rag.stores.lancedb",
    "llb.backends.base",
    "llb.backends.openai_client",
    "llb.backends.ollama",
    "llb.backends.hardware",
    "llb.backends.prepare",
    "llb.backends.planner.plan",
    "llb.backends.vllm",
    "llb.backends.telemetry",
    "llb.eval.graph",
    "llb.prompt_system",
    "llb.prompt_system.corpus",
    "llb.prompt_system.budget",
    "llb.prompt_system.template",
    "llb.prompt_system.review",
    "llb.prompt_system.tuning",
    "llb.prompt_system.manifest",
    "llb.prompt_system.pipeline",
    "llb.bench.agentic",
    "llb.bench.harness",
    "llb.bench.harness.base",
    "llb.bench.harness.langgraph",
    "llb.bench.harness.crewai",
    "llb.scoring.correctness",
    "llb.scoring.judge.model",
    "llb.scoring.aggregate",
    "llb.tracking.manifest",
    "llb.tracking.mlflow",
    "llb.tracking.server",
    "llb.executor.vram",
    "llb.executor.runner",
    "llb.graph",
    "llb.graph.build",
    "llb.graph.community",
    "llb.graph.retrieval",
    "llb.graph.store",
    "llb.graph.ingest",
    "llb.graph.summary",
]


def test_all_modules_import_under_base_install():
    for name in MODULES:
        importlib.import_module(name)
