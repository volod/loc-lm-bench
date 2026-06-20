"""Ingest SQuAD-format Ukrainian QA into canonical gold items (M0.3).

Input is either a local SQuAD-format JSON (`--squad-json`: flattened records, or nested
SQuAD `data/paragraphs/qas`), or a Hugging Face dataset id (`--hf-dataset`, needs the
optional `datasets` dep). Each record becomes a gold item with a source span computed
from the answer's char offset (validated; falls back to substring search if the offset
is missing or wrong).

Items get `provenance: public-reused` and `verified: false`: a human must review before
they score models.
"""

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

from llb.goldset.schema import GoldItem, dump_goldset
from llb.goldset.splits import assign_splits
from llb.goldset.validate import validate_items


def _doc_id(context: str) -> str:
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
    return f"squad/{digest}.txt"


def normalize(raw: object) -> list[dict]:
    """Accept flattened records or nested SQuAD; return a list of flattened records."""
    if isinstance(raw, dict) and "data" in raw:
        records: list[dict] = []
        for article in raw["data"]:
            for para in article.get("paragraphs", []):
                context = para["context"]
                for qa in para.get("qas", []):
                    answers = qa.get("answers", [])
                    records.append(
                        {
                            "id": qa.get("id"),
                            "context": context,
                            "question": qa["question"],
                            "answers": {
                                "text": [a["text"] for a in answers],
                                "answer_start": [a.get("answer_start") for a in answers],
                            },
                        }
                    )
        return records
    if isinstance(raw, list):
        return raw
    raise ValueError("unrecognized SQuAD JSON shape (expected a list or a {'data': ...} object)")


def squad_to_gold(
    records: list[dict],
    lang: str = "uk",
    provenance: str = "public-reused",
    max_items: int | None = None,
    seed: int = 13,
) -> tuple[dict[str, str], list[GoldItem], int]:
    docs: dict[str, str] = {}
    raw_items: list[dict] = []
    skipped = 0
    for i, rec in enumerate(records):
        if max_items is not None and len(raw_items) >= max_items:
            break
        context = rec["context"]
        answers = rec.get("answers") or {}
        texts = answers.get("text") or []
        if not texts:
            skipped += 1
            continue
        answer = texts[0]
        starts = answers.get("answer_start") or []
        start = starts[0] if starts else None
        if not isinstance(start, int) or context[start : start + len(answer)] != answer:
            start = context.find(answer)
        if start < 0:
            skipped += 1
            continue
        doc_id = _doc_id(context)
        docs[doc_id] = context
        item_id = str(rec.get("id") or f"squad-uk-{i:05d}")
        raw_items.append(
            {
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
                "verified": False,
            }
        )
    split_map = assign_splits([r["id"] for r in raw_items], seed=seed)
    items = [GoldItem.model_validate({**r, "split": split_map[r["id"]]}) for r in raw_items]
    return docs, items, skipped


def load_squad_json(path: Path) -> list[dict]:
    return normalize(json.loads(Path(path).read_text(encoding="utf-8")))


def coerce_answers(row: dict) -> dict:
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
    if text:
        start = row.get("answer_start")
        try:
            starts = [int(float(start))] if start not in (None, "") else []
        except (TypeError, ValueError):
            starts = []
        return {"text": [text], "answer_start": starts}
    return {"text": [], "answer_start": []}


def load_hf(
    dataset_id: str, split: str, token: str | None = None, limit: int | None = None
) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            'ERROR: "datasets" not installed. Run: uv pip install -e ".[goldset]"'
        ) from exc
    token = token or os.environ.get("HF_TOKEN")
    # When a limit is set, stream so we don't download the whole split. Over-fetch a
    # little because some rows get skipped (no answer / answer not in context).
    stream = limit is not None
    cap = None if limit is None else max(limit * 3, limit + 50)
    ds = load_dataset(dataset_id, split=split, token=token, streaming=stream)
    records: list[dict] = []
    for i, row in enumerate(ds):
        if cap is not None and i >= cap:
            break
        if "context" not in row or "question" not in row:
            continue
        records.append(
            {
                "id": row.get("id") or f"{dataset_id.replace('/', '_')}-{i:05d}",
                "context": row["context"],
                "question": row["question"],
                "answers": coerce_answers(row),
            }
        )
    return records


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
    parser.add_argument("--hf-split", default="validation")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-name", default="squad_uk.jsonl")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--lang", default="uk")
    args = parser.parse_args(argv)

    if args.squad_json:
        records = load_squad_json(args.squad_json)
    else:
        records = load_hf(args.hf_dataset, args.hf_split, limit=args.max_items)

    docs, items, skipped = squad_to_gold(records, lang=args.lang, max_items=args.max_items)
    corpus_root = args.out_dir / "corpus"
    out_path = args.out_dir / "goldset" / args.out_name
    write_corpus(docs, corpus_root)
    dump_goldset(items, out_path)

    report = validate_items(items, corpus_root)
    print(f"[ingest_squad] wrote {len(items)} items ({skipped} skipped) -> {out_path}")
    print(f"[ingest_squad] splits={report['splits']}")
    if report["errors"]:
        for err in report["errors"][:20]:
            print(f"[ingest_squad] ERROR: {err}", file=sys.stderr)
        return 1
    print("[ingest_squad] validation PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
