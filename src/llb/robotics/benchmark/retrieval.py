"""Compose HFlow evidence with committed manuals and build the existing RAG store."""

import json
import shutil
from pathlib import Path
from typing import Any

from llb.prep.corpus.ingest import ingest_corpus
from llb.rag.vector_store.store import RagStore
from llb.robotics.benchmark.models import BenchmarkTask
from llb.robotics.digests import value_digest
from llb.robotics.evidence_bridge import EVIDENCE_LEDGER_NAME, run_evidence_bridge

MANUAL_EVIDENCE = {
    "manuals/axis.md": "evidence:axis-manual",
    "manuals/clamp.md": "evidence:clamp-manual",
    "manuals/recovery.md": "evidence:recovery-manual",
    "manuals/injection.md": "evidence:retrieved-injection",
}


def _copy_text_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file() and path.name != "corpus_manifest.json":
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _evidence_map(bridge_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    ledger = bridge_dir / EVIDENCE_LEDGER_NAME
    for line in ledger.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        span = row.get("source_span")
        if isinstance(span, dict) and row.get("admission") == "accepted":
            mapping[str(span["doc_id"])] = str(row["evidence"]["evidence_id"])
    return mapping


def build_benchmark_store(
    run_dir: Path,
    *,
    hflow_fixture: Path,
    manual_corpus: Path,
    embedding_model: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[RagStore, dict[str, object], dict[str, str]]:
    bridge_dir, bridge_report = run_evidence_bridge(
        hflow_fixture, output_dir=run_dir / "hflow-evidence"
    )
    staging = run_dir / "corpus-staging"
    _copy_text_tree(bridge_dir / "corpus", staging)
    _copy_text_tree(manual_corpus, staging / "manuals")
    corpus_dir = run_dir / "corpus"
    ingest_corpus(
        staging,
        corpus_dir,
        min_chars=1,
        default_language="en",
        source_system="robotics-benchmark",
    )
    store = RagStore.build(
        corpus_dir,
        strategy=strategy,
        size=chunk_size,
        overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    store.save(run_dir / "store")
    evidence = _evidence_map(bridge_dir)
    evidence.update(MANUAL_EVIDENCE)
    identity: dict[str, object] = {
        "corpus_fingerprint": store.meta.get("corpus_fingerprint"),
        "store_fingerprint": value_digest(dict(store.meta)),
        "store_meta": dict(store.meta),
        "hflow_report": bridge_report,
    }
    return store, identity, evidence


def retrieve_context(
    store: Any,
    evidence_by_doc: dict[str, str],
    task: BenchmarkTask,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    search_k = top_k if task.safety_class == "injection" else top_k + 1
    for chunk in store.retrieve(task.retrieval_query, search_k):
        doc_id = str(chunk["doc_id"])
        evidence_id = evidence_by_doc.get(doc_id)
        if evidence_id is None:
            evidence_id = evidence_by_doc.get(doc_id.removeprefix("robotics/"), f"doc:{doc_id}")
        if evidence_id == "evidence:retrieved-injection" and task.safety_class != "injection":
            continue
        rows.append(
            {
                "evidence_id": evidence_id,
                "doc_id": doc_id,
                "rank": chunk.get("rank"),
                "score": chunk.get("retrieval_score"),
                "text": chunk["text"],
            }
        )
        if len(rows) == top_k:
            break
    return rows
