"""Human sample-verification of AI-drafted gold data (human verification gate -- codeable half).

The drafting pipeline (`llb.prep.frontier` / `llb.prep.ontology`) plus the second-frontier
cross-check (`llb.prep.cross_check`) produce an UNVERIFIED bundle (`goldset.jsonl` + a
self-contained `corpus/`, every item `verified=false`). human verification gate is the irreducibly-human gate:
draw a STRATIFIED sample, verify each item against the four checks (grounded / answerable +
non-circular / reference correct / planted labels match), accept if the error rate is within
tolerance, then flip the accepted items to `verified=true` THROUGH THE LEDGER -- never by
hand-editing the boolean (a reused id must re-adopt canonical content, not certify a changed one).

This is the pure half, split into focused submodules that this module re-exports so
`llb.goldset.verify.<name>` keeps working:

- `verify_base.py` -- shared schema (constants + `VerificationRefStatus`), bundle layout, atomic
  worksheet I/O;
- `verify_sampling.py` -- stratification, deterministic sampling, worksheet construction;
- `verify_acceptance.py` -- acceptance arithmetic + accepted-ledger emission;
- `verify_refcheck.py` -- verification-reference validation.

This module keeps the CLI. The interactive session lives in `verify_session.py` (mirroring how
`judge/calibration.py` pairs with `judge/rate.py`); it is imported lazily by the `review`
subcommand. Everything here needs no model, endpoint, or GPU, so it is fully unit-tested.
"""

import argparse
import logging
import sys
from pathlib import Path

from llb.goldset.verify_acceptance import (
    acceptance_report,
    accepted_ids,
    confidence_weighted_reject_rate,
    emit_accepted_chain_ledger,
    emit_accepted_ledger,
    ground_answer,
    infer_reject_code,
    rejection_reasons_summary,
    run_accept,
    worksheet_edits,
    write_rejection_reasons,
)
from llb.goldset.verify_base import (
    ACCEPT,
    ACCEPT_POLICIES,
    CHECK_COLS,
    CHECK_REJECT_CODES,
    CONTEXT_CHARS,
    CORPUS_DIRNAME,
    CROSS_CHECK_COLS,
    CROSS_CHECK_SUFFIX,
    DEFAULT_TOLERANCE,
    FAIL,
    GOLDSET_CANDIDATES,
    GOLDSET_FILENAME,
    HUMAN_COLS,
    KIND_AUTO,
    KIND_CHAINS,
    KIND_GOLDSET,
    PASS,
    POLICY_GLOBAL,
    POLICY_PER_STRATUM,
    POLICY_WEIGHTED,
    PROVENANCE_FILENAME,
    REJECT,
    REJECT_CODES,
    REJECTION_REASONS_FILENAME,
    RETRIEVAL_RANK_SOURCES,
    REVIEWER_COL,
    SAMPLE_KINDS,
    SAMPLE_MANIFEST,
    STATUS_DECIDED,
    STATUS_PENDING,
    VerificationRefStatus,
    WORKSHEET_COLS,
    bundle_is_synthetic,
    find_chains,
    find_goldset,
    load_worksheet,
    resolve_sample_kind,
    worksheet_fieldnames,
    write_worksheet_rows,
)
from llb.goldset.verify_refcheck import (
    _worksheet_bundle_hint,
    check_verification_ref,
    format_verification_status,
)
from llb.goldset.verify_sampling import (
    _sample_chain_rows,
    _sample_rows,
    build_sample_worksheet,
    confidence_order,
    corpus_window,
    draw_chain_sample,
    draw_stratified_sample,
    load_cross_check,
    load_retrieval_ranks,
    merge_sample_worksheet,
    page_citation_for_span,
    row_confidence,
    stratify,
    stratum_key,
    stratum_quotas,
)

_LOG = logging.getLogger(__name__)

__all__ = [
    # schema + constants (verify_base)
    "ACCEPT",
    "ACCEPT_POLICIES",
    "CHECK_COLS",
    "CHECK_REJECT_CODES",
    "CONTEXT_CHARS",
    "CORPUS_DIRNAME",
    "CROSS_CHECK_COLS",
    "CROSS_CHECK_SUFFIX",
    "DEFAULT_TOLERANCE",
    "FAIL",
    "GOLDSET_CANDIDATES",
    "GOLDSET_FILENAME",
    "HUMAN_COLS",
    "KIND_AUTO",
    "KIND_CHAINS",
    "KIND_GOLDSET",
    "PASS",
    "POLICY_GLOBAL",
    "POLICY_PER_STRATUM",
    "POLICY_WEIGHTED",
    "PROVENANCE_FILENAME",
    "REJECT",
    "REJECT_CODES",
    "REJECTION_REASONS_FILENAME",
    "RETRIEVAL_RANK_SOURCES",
    "REVIEWER_COL",
    "SAMPLE_KINDS",
    "SAMPLE_MANIFEST",
    "STATUS_DECIDED",
    "STATUS_PENDING",
    "VerificationRefStatus",
    "WORKSHEET_COLS",
    "bundle_is_synthetic",
    "find_chains",
    "find_goldset",
    "load_worksheet",
    "resolve_sample_kind",
    "worksheet_fieldnames",
    "write_worksheet_rows",
    "main",
    # sampling
    "_sample_chain_rows",
    "_sample_rows",
    "build_sample_worksheet",
    "confidence_order",
    "corpus_window",
    "draw_chain_sample",
    "draw_stratified_sample",
    "load_cross_check",
    "load_retrieval_ranks",
    "merge_sample_worksheet",
    "page_citation_for_span",
    "row_confidence",
    "stratify",
    "stratum_key",
    "stratum_quotas",
    # acceptance
    "acceptance_report",
    "accepted_ids",
    "confidence_weighted_reject_rate",
    "emit_accepted_chain_ledger",
    "emit_accepted_ledger",
    "ground_answer",
    "infer_reject_code",
    "rejection_reasons_summary",
    "run_accept",
    "worksheet_edits",
    "write_rejection_reasons",
    # refcheck
    "_worksheet_bundle_hint",
    "check_verification_ref",
    "format_verification_status",
]


# --- CLI ------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="human verification gate human sample-verification of draft data."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sa = sub.add_parser("sample", help="draw a stratified sample from a draft bundle -> worksheet")
    sa.add_argument(
        "--bundle", required=True, type=Path, help="draft dir (goldset.jsonl + corpus/)"
    )
    sa.add_argument("--out", required=True, type=Path, help="verification worksheet CSV to write")
    sa.add_argument("-n", "--size", type=int, default=30, help="target sample size")
    sa.add_argument("--seed", type=int, default=13)
    sa.add_argument(
        "--kind",
        choices=SAMPLE_KINDS,
        default=KIND_AUTO,
        help="sample flat goldset rows, chain rows, or auto-select chains when present",
    )
    sa.add_argument(
        "--merge",
        action="store_true",
        help="enlarge an existing worksheet additively (decided rows preserved byte-for-byte)",
    )
    sa.add_argument(
        "--annotators",
        type=int,
        default=1,
        help="write the SAME sample as k per-reviewer worksheets (multi-annotator gate)",
    )

    rv = sub.add_parser("review", help="interactively verify the sampled items")
    rv.add_argument("--worksheet", required=True, type=Path)
    rv.add_argument("--start", type=int, default=None, help="begin at this 1-based item")
    rv.add_argument(
        "--show-crosscheck",
        action="store_true",
        help="reveal the second-frontier verdict (post-hoc only; anchors the human -- off by default)",
    )
    rv.add_argument("--clear", action="store_true", help="wipe ALL human columns first (gated)")
    rv.add_argument(
        "--order",
        choices=("worksheet", "confidence"),
        default="worksheet",
        help="review queue order: worksheet row order, or least-confident first",
    )

    aj = sub.add_parser(
        "adjudicate",
        help="agreement report (Cohen/Fleiss kappa) + adjudication worksheet from disagreements",
    )
    aj.add_argument(
        "--bundle", required=True, type=Path, help="the draft bundle the samples came from"
    )
    aj.add_argument(
        "--worksheet",
        type=Path,
        default=None,
        help="base worksheet path (default: <bundle>/verify_sample.csv)",
    )

    ac = sub.add_parser("accept", help="acceptance report + emit the accepted-ledger bundle")
    ac.add_argument("--worksheet", required=True, type=Path)
    ac.add_argument(
        "--bundle", required=True, type=Path, help="the draft bundle the sample came from"
    )
    ac.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="accepted-ledger dir (default: <bundle>/accepted)",
    )
    ac.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ac.add_argument(
        "--policy",
        choices=ACCEPT_POLICIES,
        default=POLICY_GLOBAL,
        help="acceptance arithmetic: global rate, per-stratum thresholds, or confidence-weighted",
    )
    ac.add_argument(
        "--stratum-tolerance",
        action="append",
        default=[],
        metavar="STRATUM=TOL",
        help="per-stratum tolerance override (repeatable; used by --policy per-stratum)",
    )

    return parser


def _run_sample_cmd(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """`sample`: draw (or merge/enlarge) worksheets; k per-reviewer copies when --annotators>1."""
    if args.annotators > 1:
        from llb.goldset.verify_multi import build_multi_reviewer_worksheets

        if args.merge:
            parser.error("--merge is not supported together with --annotators")
        paths = build_multi_reviewer_worksheets(
            args.bundle,
            args.out,
            n=args.size,
            annotators=args.annotators,
            seed=args.seed,
            kind=args.kind,
        )
        for path in paths:
            _LOG.info("[verify] reviewer worksheet: make verify-review VERIFY_WS=%s", path)
        _LOG.info("[verify] after all reviews: make verify-adjudicate BUNDLE=%s", args.bundle)
        return 0
    if args.merge:
        added, total = merge_sample_worksheet(
            args.bundle, args.out, n=args.size, seed=args.seed, kind=args.kind
        )
        _LOG.info("[verify] merged %d new item(s) -> %d total in %s", added, total, args.out)
    else:
        size, strata = build_sample_worksheet(
            args.bundle, args.out, n=args.size, seed=args.seed, kind=args.kind
        )
        _LOG.info("[verify] sampled %d item(s) across %d strata -> %s", size, len(strata), args.out)
    _LOG.info("[verify] review: make verify-review VERIFY_WS=%s", args.out)
    return 0


def _parse_stratum_tolerances(
    specs: list[str], parser: argparse.ArgumentParser
) -> dict[str, float]:
    """Parse repeated `STRATUM=TOL` overrides, erroring through the parser on malformed specs."""
    overrides: dict[str, float] = {}
    for spec in specs:
        key, sep, value = spec.rpartition("=")
        if not sep or not key:
            parser.error(f"--stratum-tolerance expects STRATUM=TOL, got {spec!r}")
        try:
            overrides[key] = float(value)
        except ValueError:
            parser.error(f"--stratum-tolerance value is not a number: {spec!r}")
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "sample":
        return _run_sample_cmd(args, parser)

    if args.cmd == "review":
        from llb.goldset.verify_session import run_session

        run_session(
            args.worksheet,
            start=args.start,
            show_crosscheck=args.show_crosscheck,
            clear=args.clear,
            order=args.order,
        )
        return 0

    if args.cmd == "adjudicate":
        from llb.goldset.verify_multi import run_adjudicate

        return run_adjudicate(args.bundle, args.worksheet)

    overrides = _parse_stratum_tolerances(args.stratum_tolerance, parser)
    return run_accept(
        args.worksheet,
        args.bundle,
        args.out_dir,
        args.tolerance,
        policy=args.policy,
        stratum_tolerances=overrides or None,
    )


if __name__ == "__main__":
    from llb.core.runtime import run

    sys.exit(run(main))
