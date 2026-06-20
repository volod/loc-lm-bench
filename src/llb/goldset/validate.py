"""Validate a gold set against its corpus and structural invariants (M0 acceptance).

Schema validity is enforced when items are loaded (pydantic). This adds the checks that
need the corpus on disk and the whole set: span offsets resolve to the labeled text,
ids are unique, and each id lands in exactly one split.
"""

import argparse
import sys
from pathlib import Path

from llb.goldset.schema import GoldItem, load_goldset


def validate_items(items: list[GoldItem], corpus_root: Path) -> dict:
    """Return a report dict: {n, splits, errors}. Empty errors == PASS."""
    corpus_root = Path(corpus_root)
    errors: list[str] = []
    cache: dict[str, str] = {}
    seen_split: dict[str, str] = {}
    splits: dict[str, int] = {}

    for item in items:
        if item.id in seen_split:
            errors.append(f"duplicate id: {item.id}")
        for span in item.source_spans:
            if span.doc_id not in cache:
                path = corpus_root / span.doc_id
                if not path.exists():
                    errors.append(f"{item.id}: missing corpus doc {span.doc_id}")
                    cache[span.doc_id] = None  # type: ignore[assignment]
                    continue
                cache[span.doc_id] = path.read_text(encoding="utf-8")
            text = cache[span.doc_id]
            if text is None:
                continue
            if span.char_end > len(text):
                errors.append(f"{item.id}: span out of range in {span.doc_id}")
                continue
            got = text[span.char_start : span.char_end]
            if got != span.text:
                errors.append(f"{item.id}: span mismatch ({got!r} != {span.text!r})")
        seen_split[item.id] = item.split
        splits[item.split] = splits.get(item.split, 0) + 1

    return {"n": len(items), "splits": splits, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a gold set against its corpus.")
    parser.add_argument("--goldset", required=True, type=Path, help="gold set JSONL")
    parser.add_argument("--corpus-root", required=True, type=Path, help="corpus root dir")
    args = parser.parse_args(argv)

    items = load_goldset(args.goldset)
    report = validate_items(items, args.corpus_root)
    print(f"[validate] items={report['n']} splits={report['splits']}")
    if report["errors"]:
        for err in report["errors"][:50]:
            print(f"[validate] ERROR: {err}", file=sys.stderr)
        print(f"[validate] FAIL ({len(report['errors'])} errors)", file=sys.stderr)
        return 1
    print("[validate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
