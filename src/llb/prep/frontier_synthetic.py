"""Focused frontier synthetic implementation."""

import json
from pathlib import Path
from typing import Any, cast
from llb.goldset.schema import GoldItem, Split, dump_goldset
from llb.goldset.splits import assign_splits
from llb.prep.frontier_telemetry import LLMComplete, ProvenanceLog
from llb.prompts.registry import render_text
from llb.prep.frontier import (
    _LOG,
    _object_list,
    build_drafted_items,
    litellm_complete,
    parse_json_block,
)


def synthetic_doc_prompt(topic: str, n_labels: int) -> str:
    """Ask for a short UA factual doc on `topic` plus `n_labels` planted, span-grounded QA."""
    return render_text(
        "prep.frontier.synthetic_doc",
        {"topic": topic, "n_labels": n_labels},
    )


def prepare_synthetic_corpus(
    topics: list[str],
    *,
    planter_model: str,
    judge_model: str,
    n_labels: int = 3,
    complete: LLMComplete | None = None,
    out_dir: Path | str | None = None,
    seed: int = 13,
    log: ProvenanceLog | None = None,
) -> tuple[dict[str, str], list[GoldItem]]:
    """Generate synthetic docs + planted-label gold items. Planter MUST differ from the judge.

    The corpus is written under `out_dir/corpus/` so `build-index --corpus-root <that>` can
    index it directly, and the planted labels under `out_dir/planted_labels.jsonl` -- a
    self-contained, explicitly-synthetic bundle for a separately reported scored run.
    """
    if planter_model == judge_model:
        raise ValueError(
            "planter_model must differ from judge_model: a model must not grade answers it "
            "authored (planter != judge)."
        )
    log = log if log is not None else ProvenanceLog()
    complete = complete or litellm_complete(planter_model, log=log)
    out_dir = Path(out_dir) if out_dir is not None else None
    corpus_dir = out_dir / "corpus" if out_dir is not None else None

    docs: dict[str, str] = {}
    items: list[GoldItem] = []
    for i, topic in enumerate(topics):
        doc_id = f"synth-{i:03d}"
        payload = _synthetic_doc_payload(complete, topic, n_labels)
        if payload is None:
            continue
        document = str(payload.get("document", "")).strip()
        if not document:
            continue
        docs[doc_id] = document
        labels = _object_list(payload.get("labels", []), source=doc_id)
        items += build_drafted_items(doc_id, document, labels, "final")
        if corpus_dir is not None:
            corpus_dir.mkdir(parents=True, exist_ok=True)
            (corpus_dir / f"{doc_id}.md").write_text(document, encoding="utf-8")

    splits = assign_splits([it.id for it in items], seed=seed)
    for it in items:
        it.split = cast(Split, splits[it.id])
    if out_dir is not None:
        _write_synthetic_bundle(out_dir, docs, items, planter_model, judge_model, log)
    return docs, items


def _synthetic_doc_payload(
    complete: LLMComplete, topic: str, n_labels: int
) -> dict[str, Any] | None:
    """The planter's parsed JSON object for one topic, or None (logged) when unusable."""
    raw = complete(synthetic_doc_prompt(topic, n_labels))
    try:
        payload = parse_json_block(raw)
    except json.JSONDecodeError:
        _LOG.warning("[prepare-corpus] unparseable completion for topic %r; skipping", topic)
        return None
    if not isinstance(payload, dict):
        _LOG.warning("[prepare-corpus] expected a JSON object for topic %r; skipping", topic)
        return None
    return payload


def _write_synthetic_bundle(
    out_dir: Path,
    docs: dict[str, str],
    items: list[GoldItem],
    planter_model: str,
    judge_model: str,
    log: ProvenanceLog,
) -> None:
    """Persist the planted-labels goldset + `synthetic: true` provenance beside the corpus."""
    dump_goldset(items, out_dir / "planted_labels.jsonl")
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "kind": "synthetic-planted",
                "synthetic": True,  # planted docs, NOT real corpus -- tag every scored run
                "planter_model": planter_model,
                "judge_model": judge_model,
                "n_docs": len(docs),
                "n_items": len(items),
                "corpus_root": "corpus",
                "cost": log.summary(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _LOG.info("[prepare-corpus] %d docs, %d planted items -> %s", len(docs), len(items), out_dir)
