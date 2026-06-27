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
import ast
import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterable, Iterator, Mapping
from itertools import islice
from pathlib import Path
from typing import Any, cast

from llb.contracts import JsonObject, SquadAnswers, SquadRecord
from llb import env
from llb.goldset.schema import GoldItem, Provenance, dump_goldset
from llb.goldset.splits import assign_splits
from llb.goldset.validate import validate_items
from llb.paths import resolve_data_dir, resolve_project_path
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

_LOG = logging.getLogger(__name__)


def _doc_id(context: str) -> str:
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
    return f"squad/{digest}.txt"


def _flatten_squad_paragraph(context: str, qa: Mapping[str, Any]) -> SquadRecord:
    answers = qa.get("answers", [])
    return {
        "id": qa.get("id"),
        "context": context,
        "question": qa["question"],
        "answers": {
            "text": [a["text"] for a in answers],
            "answer_start": [a.get("answer_start") for a in answers],
        },
    }


def _flatten_squad_articles(data: object) -> list[SquadRecord]:
    if isinstance(data, dict):
        articles = [data]
    elif isinstance(data, list):
        articles = data
    else:
        raise ValueError("unrecognized SQuAD data field (expected an article or list)")
    records: list[SquadRecord] = []
    for article in articles:
        for para in article.get("paragraphs", []):
            context = para["context"]
            for qa in para.get("qas", []):
                records.append(_flatten_squad_paragraph(context, qa))
    return records


def normalize(raw: object) -> list[SquadRecord]:
    """Accept flattened records or nested SQuAD; return a list of flattened records."""
    if isinstance(raw, dict) and "data" in raw:
        return _flatten_squad_articles(raw["data"])
    if isinstance(raw, list):
        return cast(list[SquadRecord], raw)
    raise ValueError("unrecognized SQuAD JSON shape (expected a list or a {'data': ...} object)")


def _answer_char_start(context: str, answer: str, starts: list[int | None]) -> int | None:
    start = starts[0] if starts else None
    if isinstance(start, int) and context[start : start + len(answer)] == answer:
        return start
    found = context.find(answer)
    return found if found >= 0 else None


def _squad_record_to_gold(
    rec: SquadRecord,
    index: int,
    *,
    lang: str,
    provenance: Provenance,
    verified: bool,
) -> tuple[str, str, JsonObject] | None:
    """Map one SQuAD record to (doc_id, context, raw gold item dict), or None if skipped."""
    context = rec["context"]
    answers = rec.get("answers") or {}
    texts = answers.get("text") or []
    if not texts:
        return None
    answer = texts[0]
    start = _answer_char_start(context, answer, answers.get("answer_start") or [])
    if start is None:
        return None
    doc_id = _doc_id(context)
    item_id = str(rec.get("id") or f"squad-uk-{index:05d}")
    raw_item: JsonObject = {
        "id": item_id,
        "lang": lang,
        "question": rec["question"],
        "reference_answer": answer,
        "source_doc_id": doc_id,
        "source_spans": [
            {
                "doc_id": doc_id,
                "char_start": start,
                "char_end": start + len(answer),
                "text": answer,
            }
        ],
        "provenance": provenance,
        "verified": verified,
    }
    return doc_id, context, raw_item


def squad_to_gold(
    records: list[SquadRecord],
    lang: str = "uk",
    provenance: Provenance = "public-reused",
    verified: bool = False,
    max_items: int | None = None,
    seed: int = 13,
) -> tuple[dict[str, str], list[GoldItem], int]:
    docs: dict[str, str] = {}
    raw_items: list[JsonObject] = []
    skipped = 0
    for i, rec in enumerate(records):
        if max_items is not None and len(raw_items) >= max_items:
            break
        mapped = _squad_record_to_gold(rec, i, lang=lang, provenance=provenance, verified=verified)
        if mapped is None:
            skipped += 1
            continue
        doc_id, context, raw_item = mapped
        docs[doc_id] = context
        raw_items.append(raw_item)
    split_map = assign_splits([r["id"] for r in raw_items], seed=seed)
    items = [GoldItem.model_validate({**r, "split": split_map[r["id"]]}) for r in raw_items]
    return docs, items, skipped


def load_squad_json(path: Path) -> list[SquadRecord]:
    return normalize(json.loads(Path(path).read_text(encoding="utf-8")))


def coerce_answers(row: Mapping[str, Any]) -> SquadAnswers:
    """Normalize a row's answers into {text: [...], answer_start: [...]}.

    Handles three shapes seen in the wild: a real dict (HF squad), a dict serialized as
    a Python-repr string (HPLT/ua-squad), and flat answer_text/answer_start columns.
    """
    ans = row.get("answers")
    if isinstance(ans, str):
        try:
            ans = ast.literal_eval(ans)
        except (ValueError, SyntaxError):
            ans = None
    if isinstance(ans, dict) and ans.get("text"):
        return {
            "text": list(ans.get("text") or []),
            "answer_start": list(ans.get("answer_start") or []),
        }
    text = row.get("answer_text")
    if isinstance(text, str) and text:
        start = row.get("answer_start")
        starts: list[int | None]
        try:
            starts = (
                [int(float(start))] if isinstance(start, str | int | float) and start != "" else []
            )
        except (TypeError, ValueError):
            starts = []
        return {"text": [text], "answer_start": starts}
    return {"text": [], "answer_start": []}


def hf_rows_to_records(rows: Iterable[Mapping[str, Any]], dataset_id: str) -> Iterator[SquadRecord]:
    """Flatten either regular QA rows or article rows from a nested SQuAD dataset."""
    for i, row in enumerate(rows):
        if "data" in row:
            yield from normalize({"data": row["data"]})
            continue
        if "context" not in row or "question" not in row:
            continue
        yield {
            "id": row.get("id") or f"{dataset_id.replace('/', '_')}-{i:05d}",
            "context": row["context"],
            "question": row["question"],
            "answers": coerce_answers(row),
        }


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

    if args.squad_json:
        records = load_squad_json(resolve_project_path(args.squad_json))
    elif args.pinned_development_source:
        pinned_limit = args.max_items if args.max_items is not None else DEFAULT_ITEMS
        records = load_hf(
            DATASET_ID,
            DATASET_SPLIT,
            limit=pinned_limit,
            revision=DATASET_REVISION,
            context_diverse=True,
        )
    else:
        records = load_hf(
            args.hf_dataset,
            args.hf_split,
            limit=args.max_items,
            revision=args.hf_revision,
            context_diverse=args.context_diverse,
        )

    docs, items, skipped = squad_to_gold(records, lang=args.lang, max_items=args.max_items)
    adopted_documents: dict[str, Path] = {}
    adopted_count = 0
    if not args.no_verification_ledger:
        ledger_paths = args.verified_goldset or [DEFAULT_VERIFIED_GOLDSET]
        ledger = load_verified_ledger([resolve_project_path(path) for path in ledger_paths])
        items, adopted_documents, adopted_count = apply_verified_ledger(items, ledger)
        if items and ledger.items and adopted_count == 0:
            _LOG.warning(
                "[ingest_squad] no imported ids matched the verification ledger; "
                "all generated items remain unverified"
            )

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


if __name__ == "__main__":
    from llb.runtime import run

    sys.exit(run(main))
