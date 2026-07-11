"""Validate a gold set against its corpus and structural invariants (data bootstrap acceptance).

Schema validity is enforced when items are loaded (pydantic). This adds the checks that
need the corpus on disk and the whole set: span offsets resolve to the labeled text,
ids are unique, and each id lands in exactly one split.
"""

import argparse
import logging
import sys
from pathlib import Path

from llb.core.contracts import ValidationReport
from llb.goldset.chains import load_chains, validate_chains
from llb.goldset.schema import GoldItem, SourceSpan, load_goldset

_LOG = logging.getLogger(__name__)
CorpusCache = dict[str, str | None]


def _get_corpus_text(
    corpus_root: Path,
    doc_id: str,
    cache: CorpusCache,
    errors: list[str],
    item_id: str,
) -> str | None:
    if doc_id in cache:
        return cache[doc_id]
    path = corpus_root / doc_id
    if not path.exists():
        errors.append(f"{item_id}: missing corpus doc {doc_id}")
        cache[doc_id] = None
        return None
    cache[doc_id] = path.read_text(encoding="utf-8")
    return cache[doc_id]


def _validate_span(item_id: str, span: SourceSpan, text: str, errors: list[str]) -> None:
    if span.char_end > len(text):
        errors.append(f"{item_id}: span out of range in {span.doc_id}")
        return
    got = text[span.char_start : span.char_end]
    if got != span.text:
        errors.append(f"{item_id}: span mismatch ({got!r} != {span.text!r})")


def _validate_item_spans(
    item: GoldItem,
    corpus_root: Path,
    cache: CorpusCache,
    errors: list[str],
) -> None:
    for span in item.source_spans:
        text = _get_corpus_text(corpus_root, span.doc_id, cache, errors, item.id)
        if text is None:
            continue
        _validate_span(item.id, span, text, errors)


def validate_items(items: list[GoldItem], corpus_root: Path) -> ValidationReport:
    """Return a report dict: {n, splits, errors}. Empty errors == PASS."""
    corpus_root = Path(corpus_root)
    errors: list[str] = []
    cache: CorpusCache = {}
    seen_split: dict[str, str] = {}
    splits: dict[str, int] = {}

    for item in items:
        if item.id in seen_split:
            errors.append(f"duplicate id: {item.id}")
        _validate_item_spans(item, corpus_root, cache, errors)
        seen_split[item.id] = item.split
        splits[item.split] = splits.get(item.split, 0) + 1

    return {"n": len(items), "splits": splits, "errors": errors}


def _log_report(label: str, report: ValidationReport) -> None:
    _LOG.info("[validate] %s items=%s splits=%s", label, report["n"], report["splits"])
    if report["errors"]:
        for err in report["errors"][:50]:
            _LOG.error("[validate] ERROR: %s", err)
        _LOG.error("[validate] %s FAIL (%d errors)", label, len(report["errors"]))
        return
    _LOG.info("[validate] %s PASS", label)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate gold data against its corpus.")
    parser.add_argument("--goldset", type=Path, help="gold set JSONL")
    parser.add_argument("--chains", type=Path, help="chain-of-questions JSONL")
    parser.add_argument("--corpus-root", required=True, type=Path, help="corpus root dir")
    args = parser.parse_args(argv)

    if args.goldset is None and args.chains is None:
        parser.error("provide --goldset, --chains, or both")

    reports: list[ValidationReport] = []
    if args.goldset is not None:
        report = validate_items(load_goldset(args.goldset), args.corpus_root)
        _log_report("goldset", report)
        reports.append(report)
    if args.chains is not None:
        report = validate_chains(load_chains(args.chains), args.corpus_root)
        _log_report("chains", report)
        reports.append(report)
    if any(report["errors"] for report in reports):
        return 1
    return 0


if __name__ == "__main__":
    from llb.core.runtime import run

    sys.exit(run(main))
