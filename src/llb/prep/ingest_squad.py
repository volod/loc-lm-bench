"""Ingest SQuAD-format Ukrainian QA into canonical gold items (Ukrainian SQuAD ingest).

Input is either a local SQuAD-format JSON (`--squad-json`: flattened records, or nested
SQuAD `data/paragraphs/qas`), or a Hugging Face dataset id (`--hf-dataset`, needs the
optional `datasets` dep). Each record becomes a gold item with a source span computed
from the answer's char offset (validated; falls back to substring search if the offset
is missing or wrong).

Runtime imports start as `provenance: public-reused` and `verified: false`. By default, the
ingester then adopts canonical items with matching ids from the committed human-verification
ledger. Nonmatching imports remain unverified and cannot score models.
"""

import argparse
import os
import sys
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

from llb.core.contracts.rag import SquadRecord
from llb.core import env
from llb.goldset.schema import GoldItem, dump_goldset
from llb.goldset.validate import validate_items
from llb.core.paths import resolve_data_dir, resolve_project_path
from llb.prep.verified_ledger import (
    DEFAULT_VERIFIED_GOLDSET,
    apply_verified_ledger,
    copy_verified_documents,
    load_verified_ledger,
)
from llb.prep.ua_squad_source import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    DEFAULT_ITEMS,
    iter_context_diverse,
)
from llb.prep.squad_records import _LOG, hf_rows_to_records, load_squad_json, squad_to_gold


def load_hf(
    dataset_id: str,
    split: str,
    token: str | None = None,
    limit: int | None = None,
    revision: str | None = None,
    context_diverse: bool = False,
) -> list[SquadRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            'ERROR: "datasets" not installed. Run: uv pip install -e ".[goldset]"'
        ) from exc
    token = token or os.environ.get(env.HF_TOKEN)
    # When a limit is set, stream so we don't download the whole split. Over-fetch a
    # little because some rows get skipped (no answer / answer not in context).
    stream = limit is not None
    cap = None if limit is None else max(limit * 3, limit + 50)
    load_kwargs: dict[str, Any] = {"split": split, "token": token, "streaming": stream}
    if revision is not None:
        load_kwargs["revision"] = revision
    ds = load_dataset(dataset_id, **load_kwargs)
    records: Iterable[SquadRecord] = hf_rows_to_records(ds, dataset_id)
    if context_diverse:
        records = iter_context_diverse(records)
        cap = limit
    return list(records if cap is None else islice(records, cap))


def write_corpus(docs: dict[str, str], corpus_root: Path) -> None:
    for doc_id, text in docs.items():
        path = corpus_root / doc_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest SQuAD-format UA QA into gold items.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--squad-json", type=Path, help="local SQuAD-format JSON")
    source.add_argument("--hf-dataset", type=str, help="HF dataset id (needs the [goldset] extra)")
    source.add_argument(
        "--pinned-development-source",
        action="store_true",
        help="use the exact dataset revision, split, and selection behind the reviewed fixture",
    )
    parser.add_argument("--hf-split", default="validation")
    parser.add_argument("--hf-revision", default=None, help="pinned Hugging Face revision")
    parser.add_argument(
        "--context-diverse",
        action="store_true",
        help="select the first grounded QA per distinct context before applying --max-items",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output root (default: $DATA_DIR/llb)",
    )
    parser.add_argument("--out-name", default="squad_uk.jsonl")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--lang", default="uk")
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument(
        "--verified-goldset",
        action="append",
        type=Path,
        help=(
            "reviewed gold set used as an id-keyed verification ledger; repeat to combine "
            "ledgers (default: committed ua_squad_postedited_v1 fixture)"
        ),
    )
    verification.add_argument(
        "--no-verification-ledger",
        action="store_true",
        help="leave every imported item unverified even when its id has been reviewed",
    )
    args = parser.parse_args(argv)

    records = _load_source_records(args)
    docs, items, skipped = squad_to_gold(records, lang=args.lang, max_items=args.max_items)
    items, adopted_documents, adopted_count = _apply_verification(args, items)

    out_dir = resolve_project_path(args.out_dir) if args.out_dir else resolve_data_dir(None) / "llb"
    corpus_root = out_dir / "corpus"
    out_path = out_dir / "goldset" / args.out_name
    write_corpus(docs, corpus_root)
    copy_verified_documents(adopted_documents, corpus_root)
    dump_goldset(items, out_path)

    report = validate_items(items, corpus_root)
    _LOG.info(
        "[ingest_squad] wrote %d items (%d adopted as human-verified, %d skipped) -> %s",
        len(items),
        adopted_count,
        skipped,
        out_path,
    )
    _LOG.info("[ingest_squad] splits=%s", report["splits"])
    if report["errors"]:
        for err in report["errors"][:20]:
            _LOG.error("[ingest_squad] ERROR: %s", err)
        return 1
    _LOG.info("[ingest_squad] validation PASS")
    return 0


def _load_source_records(args: argparse.Namespace) -> list[SquadRecord]:
    """The raw QA records: local SQuAD JSON, the pinned development source, or a HF dataset."""
    if args.squad_json:
        return load_squad_json(resolve_project_path(args.squad_json))
    if args.pinned_development_source:
        pinned_limit = args.max_items if args.max_items is not None else DEFAULT_ITEMS
        return load_hf(
            DATASET_ID,
            DATASET_SPLIT,
            limit=pinned_limit,
            revision=DATASET_REVISION,
            context_diverse=True,
        )
    return load_hf(
        args.hf_dataset,
        args.hf_split,
        limit=args.max_items,
        revision=args.hf_revision,
        context_diverse=args.context_diverse,
    )


def _apply_verification(
    args: argparse.Namespace, items: list[GoldItem]
) -> tuple[list[GoldItem], dict[str, Path], int]:
    """Adopt human-verified rows from the ledger unless verification was disabled."""
    if args.no_verification_ledger:
        return items, {}, 0
    ledger_paths = args.verified_goldset or [DEFAULT_VERIFIED_GOLDSET]
    ledger = load_verified_ledger([resolve_project_path(path) for path in ledger_paths])
    items, adopted_documents, adopted_count = apply_verified_ledger(items, ledger)
    if items and ledger.items and adopted_count == 0:
        _LOG.warning(
            "[ingest_squad] no imported ids matched the verification ledger; "
            "all generated items remain unverified"
        )
    return items, adopted_documents, adopted_count


if __name__ == "__main__":
    from llb.core.runtime import run

    sys.exit(run(main))
