"""Curation of externally drafted benchmark artifacts (merge + dedup + filter).

One entry point per external-draft contract artifact kind:

- ``squad``     -- Artifact A goldset drafts -> one SQuAD JSON for `make ingest-squad`;
- ``security``  -- Artifact C security cases -> one array for `make bench-security`;
- ``chains``    -- Artifact D chain drafts   -> one JSONL (review-only until chains land);
- ``inventory`` -- prompt-01 coverage plans  -> one merged inventory.json.

`curate()` dispatches by kind and returns `(payload, CurationReport)`; the CLI command is
`llb curate-drafts` (make: `curate-drafts`). See
`docs/guides/external-ai-service-artifacts.md` for the workflow position: curation runs after
export, before import/validation.
"""

import json
from pathlib import Path
from typing import Any

from llb.goldset.schema import load_goldset
from llb.prep.curation.chains import curate_chains
from llb.prep.curation.common import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_MIN_CONTEXT_CHARS,
    CurationReport,
    QuestionEmbedder,
    load_corpus_texts,
    resolve_embedder,
)
from llb.prep.curation.inventory import curate_inventory
from llb.prep.curation.security import curate_security
from llb.prep.curation.squad import curate_squad

KINDS = ("squad", "security", "chains", "inventory")

__all__ = [
    "KINDS",
    "CurationReport",
    "curate",
    "curate_chains",
    "curate_inventory",
    "curate_security",
    "curate_squad",
    "load_prior_bundle_questions",
    "resolve_embedder",
    "write_curated",
]


def load_prior_bundle_questions(bundles: list[Path]) -> list[str]:
    """Questions of prior draft bundles' goldsets (for cross-bundle near-dup suppression)."""
    questions: list[str] = []
    for bundle in bundles:
        path = Path(bundle) / "goldset.jsonl"
        if path.is_file():
            questions.extend(item.question for item in load_goldset(path))
    return questions


def curate(
    kind: str,
    inputs: list[Path],
    *,
    corpus_root: Path | None = None,
    embedder: QuestionEmbedder | None = None,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    min_context_chars: int = DEFAULT_MIN_CONTEXT_CHARS,
    dedup_spans: bool = False,
    prior_questions: list[str] | None = None,
) -> tuple[Any, CurationReport]:
    """Dispatch to the artifact-kind curator; returns (payload, report)."""
    corpus_texts = load_corpus_texts(corpus_root) if corpus_root is not None else None
    if kind == "squad":
        return curate_squad(
            inputs,
            corpus_texts=corpus_texts,
            embedder=embedder,
            dedup_threshold=dedup_threshold,
            min_context_chars=min_context_chars,
            dedup_spans=dedup_spans,
            prior_questions=prior_questions,
        )
    if kind == "security":
        return curate_security(
            inputs,
            corpus_texts=corpus_texts,
            embedder=embedder,
            dedup_threshold=dedup_threshold,
            prior_questions=prior_questions,
        )
    if kind == "chains":
        return curate_chains(
            inputs,
            corpus_texts=corpus_texts,
            embedder=embedder,
            dedup_threshold=dedup_threshold,
            prior_questions=prior_questions,
        )
    if kind == "inventory":
        return curate_inventory(inputs, corpus_texts=corpus_texts)
    raise SystemExit(f"[curate] unknown artifact kind: {kind!r} (expected one of {KINDS})")


def write_curated(kind: str, payload: Any, out: Path, report: CurationReport) -> Path:
    """Write the curated artifact (JSONL for chains, JSON otherwise) + the report sidecar."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if kind == "chains":
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload)
        out.write_text(content, encoding="utf-8")
    else:
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = out.with_name(f"{out.stem}.curation_report.json")
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path
