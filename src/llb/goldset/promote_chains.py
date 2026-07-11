"""Promote a human-accepted chain bundle into a compact committed fixture."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from llb.goldset.chains import ChainItem, dump_chains, load_chains, validate_chains
from llb.goldset.schema import SourceSpan

_LOG = logging.getLogger(__name__)
_ACCEPTED_DIR = "accepted"
_CHAINS_FILE = "chains.jsonl"
_CORPUS_DIR = "corpus"
_MANIFEST_FILE = "fixture_manifest.json"
_SPAN_SEPARATOR = "\n\n---\n\n"


def _safe_doc_path(root: Path, doc_id: str) -> Path:
    relative = Path(doc_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source_doc_id: {doc_id}")
    return root / relative


def _raise_validation_errors(label: str, errors: list[str]) -> None:
    if errors:
        details = "; ".join(errors[:5])
        raise ValueError(f"{label} validation failed ({len(errors)} errors): {details}")


def _span_key(span: SourceSpan) -> tuple[str, int, int, str]:
    return (span.doc_id, span.char_start, span.char_end, span.text)


def _compact_corpus(
    chains: list[ChainItem], source_root: Path, target_root: Path
) -> tuple[list[ChainItem], list[dict[str, Any]]]:
    spans_by_doc: dict[str, dict[tuple[str, int, int, str], SourceSpan]] = {}
    for chain in chains:
        for step in chain.steps:
            for span in step.source_spans:
                spans_by_doc.setdefault(span.doc_id, {})[_span_key(span)] = span

    remapped: dict[tuple[str, int, int, str], SourceSpan] = {}
    documents: list[dict[str, Any]] = []
    for doc_id, keyed_spans in sorted(spans_by_doc.items()):
        source_path = _safe_doc_path(source_root, doc_id)
        source_text = source_path.read_text(encoding="utf-8")
        parts: list[str] = []
        cursor = 0
        ordered = sorted(keyed_spans.values(), key=lambda span: (span.char_start, span.char_end))
        for index, span in enumerate(ordered):
            if index:
                parts.append(_SPAN_SEPARATOR)
                cursor += len(_SPAN_SEPARATOR)
            start = cursor
            parts.append(span.text)
            cursor += len(span.text)
            remapped[_span_key(span)] = SourceSpan(
                doc_id=doc_id,
                char_start=start,
                char_end=cursor,
                text=span.text,
            )
        compact_text = "".join(parts)
        target_path = _safe_doc_path(target_root, doc_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(compact_text, encoding="utf-8")
        documents.append(
            {
                "doc_id": doc_id,
                "original_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
                "original_chars": len(source_text),
                "fixture_chars": len(compact_text),
                "reviewed_spans": len(ordered),
            }
        )

    compact_chains: list[ChainItem] = []
    for chain in chains:
        data = chain.model_dump()
        for step in data["steps"]:
            step["source_spans"] = [
                remapped[
                    (
                        span["doc_id"],
                        span["char_start"],
                        span["char_end"],
                        span["text"],
                    )
                ].model_dump()
                for span in step["source_spans"]
            ]
        compact_chains.append(ChainItem.model_validate(data))
    return compact_chains, documents


def promote_chain_bundle(
    bundle: Path,
    output_dir: Path,
    *,
    min_chains: int = 10,
) -> dict[str, Any]:
    """Validate and promote accepted chains with an exact-span-only corpus."""
    if min_chains < 1:
        raise ValueError("min_chains must be at least 1")
    accepted = bundle / _ACCEPTED_DIR
    chains_path = accepted / _CHAINS_FILE
    corpus_root = accepted / _CORPUS_DIR
    if not chains_path.is_file():
        raise ValueError(f"accepted chain ledger not found: {chains_path}")
    if not corpus_root.is_dir():
        raise ValueError(f"accepted corpus not found: {corpus_root}")
    if output_dir.exists():
        raise ValueError(f"fixture destination already exists: {output_dir}")

    chains = load_chains(chains_path)
    if len(chains) < min_chains:
        raise ValueError(f"accepted chains {len(chains)} is below required minimum {min_chains}")
    unverified = [chain.chain_id for chain in chains if not chain.verified]
    if unverified:
        raise ValueError(f"accepted ledger contains unverified chains: {', '.join(unverified[:5])}")
    source_report = validate_chains(chains, corpus_root)
    _raise_validation_errors("accepted chain", source_report["errors"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        compact_root = temp_dir / _CORPUS_DIR
        compact_chains, documents = _compact_corpus(chains, corpus_root, compact_root)
        compact_report = validate_chains(compact_chains, compact_root)
        _raise_validation_errors("promoted chain", compact_report["errors"])
        dump_chains(compact_chains, temp_dir / _CHAINS_FILE)
        manifest = {
            "kind": "verified-chain-fixture",
            "verified": True,
            "chains": len(compact_chains),
            "minimum_required": min_chains,
            "compact_corpus": True,
            "documents": documents,
        }
        (temp_dir / _MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    _LOG.info(
        "[chain-goldset] promoted %d verified chains -> %s",
        len(chains),
        output_dir,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote accepted chain rows into a compact committed fixture."
    )
    parser.add_argument("--bundle", required=True, type=Path, help="reviewed draft bundle")
    parser.add_argument("--out", required=True, type=Path, help="new fixture directory")
    parser.add_argument("--min-chains", type=int, default=10, help="minimum accepted chain count")
    args = parser.parse_args(argv)
    promote_chain_bundle(args.bundle, args.out, min_chains=args.min_chains)
    return 0


if __name__ == "__main__":
    from llb.core.runtime import run

    sys.exit(run(main))
