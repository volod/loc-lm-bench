"""Acceptance arithmetic and accepted-ledger emission for the human verification gate.

Coded rejection reasons, accept-with-edit re-grounding, the global/per-stratum/weighted policies,
and ledger emission live here. Shared constants and worksheet I/O live in `verify_base.py`.
"""

import shutil
from collections.abc import Sequence
from pathlib import Path

from llb.goldset.chains import CHAINS_FILENAME, ChainItem, dump_chains, load_chains
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset, load_goldset
from llb.goldset.verify_base import (
    ACCEPT,
    CORPUS_DIRNAME,
    DEFAULT_TOLERANCE,
    GOLDSET_FILENAME,
    KIND_CHAINS,
    POLICY_GLOBAL,
    _worksheet_sample_kind,
    find_chains,
    find_goldset,
)
from llb.goldset.verify_acceptance_report import (
    acceptance_report,
    ground_answer,
    worksheet_edits,
)
from llb.goldset.verify_acceptance_io import (
    _LOG,
    _accept_rows,
    _log_report,
    write_rejection_reasons,
)


# --- coded rejection reasons ----------------------------------------------------------------


# --- accept-with-edit re-grounding ----------------------------------------------------------


# --- acceptance arithmetic ----------------------------------------------------------------


# --- accepted-ledger emission (the flip is an ADOPTION, not a boolean edit) -----------------


def accepted_ids(rows: Sequence[dict[str, str]]) -> list[str]:
    """Item ids the human explicitly accepted."""
    return [
        (row.get("item_id") or "").strip()
        for row in rows
        if (row.get("decision") or "").strip() == ACCEPT and (row.get("item_id") or "").strip()
    ]


def _apply_edit(item: GoldItem, edited_answer: str, corpus_root: Path) -> GoldItem:
    """Re-ground `edited_answer` in the item's span doc and return the edited item.

    The edited answer must exist VERBATIM in the corpus doc -- an edit that no longer matches any
    span is refused here (raises), so an un-groundable edit can never certify through the ledger
    even if the worksheet cell was hand-edited after the session's own re-grounding.
    """
    span = item.source_spans[0]
    doc_path = corpus_root / span.doc_id
    if not doc_path.is_file():
        raise FileNotFoundError(f"{item.id}: corpus doc for edited answer not found: {doc_path}")
    text = doc_path.read_text(encoding="utf-8")
    offsets = ground_answer(text, edited_answer, hint_start=span.char_start)
    if offsets is None:
        raise ValueError(
            f"{item.id}: edited answer is not a verbatim span of {span.doc_id}; "
            "re-ground it in the review session before accepting"
        )
    start, end = offsets
    new_span = SourceSpan(doc_id=span.doc_id, char_start=start, char_end=end, text=text[start:end])
    return item.model_copy(
        update={
            "reference_answer": edited_answer,
            "source_spans": [new_span, *item.source_spans[1:]],
        }
    )


def emit_accepted_ledger(
    bundle: Path,
    accepted: Sequence[str],
    out_dir: Path,
    *,
    edits: dict[str, str] | None = None,
) -> int:
    """Write an accepted-ledger bundle (`goldset.jsonl` verified=true + sibling `corpus/`).

    The accepted items are taken VERBATIM from the draft bundle (canonical content + grounded
    spans) with `verified` flipped to true, and every corpus doc they depend on is copied so the
    ledger is self-contained. Feeding this to `ingest_squad --verified-goldset <out_dir>/goldset.jsonl`
    re-adopts those ids by REPLACEMENT, which is what stops a reused id from certifying changed
    content. We never hand-edit the boolean in the draft bundle.

    `edits` (item id -> accept-with-edit reference answer from the worksheet) are applied through
    `_apply_edit`: the edited answer is re-grounded against the bundle corpus and the primary span
    replaced; an edit that no longer grounds raises instead of certifying.
    """
    bundle = Path(bundle)
    out_dir = Path(out_dir)
    keep = set(accepted)
    edits = edits or {}
    src_corpus = bundle / CORPUS_DIRNAME
    dst_corpus = out_dir / CORPUS_DIRNAME
    verified: list[GoldItem] = []
    doc_ids: set[str] = set()
    for item in load_goldset(find_goldset(bundle)):
        if item.id not in keep:
            continue
        edited_answer = edits.get(item.id, "")
        if edited_answer:
            item = _apply_edit(item, edited_answer, src_corpus)
        verified.append(item.model_copy(update={"verified": True}))
        doc_ids.add(item.source_doc_id)
        doc_ids.update(span.doc_id for span in item.source_spans)
    dump_goldset(verified, out_dir / GOLDSET_FILENAME)
    for doc_id in sorted(doc_ids):
        source = src_corpus / doc_id
        if not source.is_file():
            raise FileNotFoundError(f"corpus doc for accepted item not found: {source}")
        dest = dst_corpus / doc_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return len(verified)


def emit_accepted_chain_ledger(
    bundle: Path,
    accepted: Sequence[str],
    out_dir: Path,
) -> int:
    """Write an accepted chain ledger (`chains.jsonl` verified=true + sibling `corpus/`)."""
    bundle = Path(bundle)
    out_dir = Path(out_dir)
    keep = set(accepted)
    src_corpus = bundle / CORPUS_DIRNAME
    dst_corpus = out_dir / CORPUS_DIRNAME
    verified: list[ChainItem] = []
    doc_ids: set[str] = set()
    for chain in load_chains(find_chains(bundle)):
        if chain.chain_id not in keep:
            continue
        verified.append(chain.model_copy(update={"verified": True}))
        for step in chain.steps:
            doc_ids.add(step.source_doc_id)
            doc_ids.update(span.doc_id for span in step.source_spans)
    dump_chains(verified, out_dir / CHAINS_FILENAME)
    for doc_id in sorted(doc_ids):
        source = src_corpus / doc_id
        if not source.is_file():
            raise FileNotFoundError(f"corpus doc for accepted chain not found: {source}")
        dest = dst_corpus / doc_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return len(verified)


def run_accept(
    worksheet: Path,
    bundle: Path,
    out_dir: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    *,
    policy: str = POLICY_GLOBAL,
    stratum_tolerances: dict[str, float] | None = None,
) -> int:
    """The `accept` subcommand: acceptance report, rejection-reason export, ledger emission."""
    rows = _accept_rows(Path(worksheet))
    report = acceptance_report(
        rows, tolerance, policy=policy, stratum_tolerances=stratum_tolerances
    )
    _log_report(report)
    out_dir = out_dir or (Path(bundle) / "accepted")
    reasons_path = write_rejection_reasons(rows, out_dir)
    if reasons_path is not None:
        _LOG.info("[verify] rejection reasons for draft feedback -> %s", reasons_path)
    accepted = accepted_ids(rows)
    if not accepted:
        _LOG.info("[verify] no accepted items -- nothing to flip; resolve the sample first")
        return 0 if report["passed"] else 1
    kind = _worksheet_sample_kind(Path(worksheet))
    if kind == KIND_CHAINS:
        n = emit_accepted_chain_ledger(bundle, accepted, out_dir)
        _LOG.info("[verify] wrote %d accepted chain(s) -> %s", n, out_dir / CHAINS_FILENAME)
    else:
        edits = {k: v for k, v in worksheet_edits(rows).items() if k in set(accepted)}
        n = emit_accepted_ledger(bundle, accepted, out_dir, edits=edits)
        if edits:
            _LOG.info(
                "[verify] applied %d accept-with-edit answer(s) through re-grounding", len(edits)
            )
        _LOG.info("[verify] wrote %d accepted item(s) -> %s", n, out_dir / GOLDSET_FILENAME)
        _LOG.info(
            "[verify] flip via the ledger: python -m llb.prep.ingest_squad "
            "--squad-json <source> --verified-goldset %s",
            out_dir / GOLDSET_FILENAME,
        )
    return 0 if report["passed"] else 1
