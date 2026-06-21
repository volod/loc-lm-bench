"""Build the committed UA-SQuAD development fixture from a pinned upstream JSON export.

This is a repository-maintenance command, not the runtime Hugging Face importer. It accepts
only the pinned FIdo-AI validation export, selects the first grounded QA for each distinct
context, and emits canonical verified items plus their corpus. The upstream dataset card says
the Ukrainian translation/adaptation was post-edited and its answer spans were aligned.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from llb.goldset.schema import dump_goldset
from llb.goldset.validate import validate_items
from llb.prep.ingest_squad import load_squad_json, squad_to_gold, write_corpus
from llb.prep.ua_squad_source import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    DEFAULT_ITEMS,
    SOURCE_FILE,
    SOURCE_SHA256,
    select_context_diverse,
)

_LOG = logging.getLogger(__name__)

HASH_BLOCK_BYTES = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fixture(source: Path, out_dir: Path, max_items: int = DEFAULT_ITEMS) -> None:
    """Build and validate the pinned canonical fixture under `out_dir`."""
    actual_sha256 = _sha256(source)
    if actual_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"unexpected {SOURCE_FILE} SHA-256: {actual_sha256}; expected {SOURCE_SHA256}"
        )

    records = select_context_diverse(load_squad_json(source), max_items)
    docs, items, skipped = squad_to_gold(records, verified=True, max_items=max_items)
    if skipped or len(items) != max_items:
        raise ValueError(f"fixture conversion produced {len(items)} items and skipped {skipped}")

    corpus_root = out_dir / "corpus"
    write_corpus(docs, corpus_root)
    dump_goldset(items, out_dir / "goldset.jsonl")
    metadata = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "selection": "first grounded QA per distinct context in upstream validation order",
        "items": len(items),
        "documents": len(docs),
        "verification_basis": (
            "project human review of the pinned selection plus exact source-span validation"
        ),
        "upstream_url": "https://huggingface.co/datasets/FIdo-AI/ua-squad",
        "data_license": "CC BY-SA 4.0 per the upstream dataset card's derivative-data note",
    }
    (out_dir / "source.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    report = validate_items(items, corpus_root)
    if report["errors"]:
        raise ValueError(f"generated fixture failed validation: {report['errors'][:3]}")
    _LOG.info(
        "[published-goldset] wrote %d items and %d docs -> %s", len(items), len(docs), out_dir
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the pinned published UA-SQuAD fixture.")
    parser.add_argument("--source", type=Path, required=True, help=f"downloaded {SOURCE_FILE}")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=DEFAULT_ITEMS)
    args = parser.parse_args(argv)
    build_fixture(args.source, args.out_dir, args.max_items)
    return 0


if __name__ == "__main__":
    from llb.runtime import run

    sys.exit(run(main))
