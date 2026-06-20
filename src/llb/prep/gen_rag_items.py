"""Generate canonical Ukrainian RAG gold items from a committed sample spec.

DATA lives in the spec file (samples/*.json): source docs + item definitions.
CODE (this module) only transforms it: write the corpus docs, compute source-span char
offsets from the exact text, build schema-validated GoldItems, write JSONL, and validate
against the on-disk corpus.

Run via `scripts/gen_rag_items.sh` or `make gen-rag-items`, or directly:
    python -m llb.prep.gen_rag_items --spec samples/rag_items_uk.json --out-dir .data/llb
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast

from llb.contracts import RagDataSpec
from llb.goldset.schema import GoldItem, dump_goldset
from llb.goldset.validate import validate_items

_LOG = logging.getLogger(__name__)


def load_spec(spec_path: Path) -> RagDataSpec:
    with Path(spec_path).open(encoding="utf-8") as fh:
        spec = json.load(fh)
    for key in ("lang", "docs", "items"):
        if key not in spec:
            raise ValueError(f"spec missing required key: {key}")
    return cast(RagDataSpec, spec)


def write_corpus(docs: dict[str, str], corpus_root: Path) -> None:
    for doc_id, text in docs.items():
        path = corpus_root / doc_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def build_items(spec: RagDataSpec) -> list[GoldItem]:
    """Turn raw spec items into schema-validated gold items with computed source spans."""
    docs = spec["docs"]
    lang = spec["lang"]
    items: list[GoldItem] = []
    seen: set[str] = set()
    for raw in spec["items"]:
        item_id = raw["id"]
        if item_id in seen:
            raise ValueError(f"duplicate item id: {item_id}")
        seen.add(item_id)
        doc_id = raw["doc"]
        if doc_id not in docs:
            raise ValueError(f"{item_id}: unknown doc {doc_id}")
        span = raw["answer_span"]
        start = docs[doc_id].find(span)
        if start < 0:
            raise ValueError(f"{item_id}: answer_span not found in {doc_id}")
        items.append(
            GoldItem.model_validate(
                {
                    "id": item_id,
                    "lang": lang,
                    "question": raw["question"],
                    "reference_answer": raw["reference_answer"],
                    "source_doc_id": doc_id,
                    "source_spans": [
                        {
                            "doc_id": doc_id,
                            "char_start": start,
                            "char_end": start + len(span),
                            "text": span,
                        }
                    ],
                    "provenance": raw.get("provenance", "sample-generated"),
                    "verified": bool(raw.get("verified", False)),
                    "split": raw["split"],
                }
            )
        )
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate canonical UA RAG gold items from a sample spec."
    )
    parser.add_argument("--spec", required=True, type=Path, help="path to sample spec JSON")
    parser.add_argument("--out-dir", required=True, type=Path, help="output root (e.g. .data/llb)")
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    corpus_root = args.out_dir / "corpus"
    out_path = args.out_dir / "goldset" / "sample_rag_items.jsonl"

    write_corpus(spec["docs"], corpus_root)
    items = build_items(spec)
    dump_goldset(items, out_path)

    report = validate_items(items, corpus_root)
    if report["errors"]:
        for err in report["errors"]:
            _LOG.error("[gen_rag_items] ERROR: %s", err)
        return 1

    splits = report["splits"]
    _LOG.info("[gen_rag_items] wrote %d items -> %s", len(items), out_path)
    _LOG.info("[gen_rag_items] corpus docs -> %s", corpus_root)
    _LOG.info(
        "[gen_rag_items] splits: %s",
        ", ".join(f"{key}={splits[key]}" for key in sorted(splits)),
    )
    return 0


if __name__ == "__main__":
    from llb.runtime import run

    sys.exit(run(main))
