"""Command-line orchestration for sampling, review, adjudication, and acceptance."""

import argparse
import logging
from pathlib import Path

from llb.goldset.verify_acceptance import run_accept
from llb.goldset.verify_base import (
    ACCEPT_POLICIES,
    DEFAULT_TOLERANCE,
    KIND_AUTO,
    POLICY_GLOBAL,
    SAMPLE_KINDS,
)
from llb.goldset.verify_sampling.worksheet import build_sample_worksheet, merge_sample_worksheet

_LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="human verification gate human sample-verification of draft data."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sample = sub.add_parser("sample", help="draw a stratified sample -> worksheet")
    sample.add_argument("--bundle", required=True, type=Path)
    sample.add_argument("--out", required=True, type=Path)
    sample.add_argument("-n", "--size", type=int, default=30)
    sample.add_argument("--seed", type=int, default=13)
    sample.add_argument("--kind", choices=SAMPLE_KINDS, default=KIND_AUTO)
    sample.add_argument("--merge", action="store_true")
    sample.add_argument("--annotators", type=int, default=1)

    review = sub.add_parser("review", help="interactively verify sampled items")
    review.add_argument("--worksheet", required=True, type=Path)
    review.add_argument("--start", type=int, default=None)
    review.add_argument("--show-crosscheck", action="store_true")
    review.add_argument("--clear", action="store_true")
    review.add_argument("--order", choices=("worksheet", "confidence"), default="worksheet")

    adjudicate = sub.add_parser("adjudicate", help="report agreement and adjudicate disagreements")
    adjudicate.add_argument("--bundle", required=True, type=Path)
    adjudicate.add_argument("--worksheet", type=Path, default=None)

    accept = sub.add_parser("accept", help="report acceptance and emit the accepted ledger")
    accept.add_argument("--worksheet", required=True, type=Path)
    accept.add_argument("--bundle", required=True, type=Path)
    accept.add_argument("--out-dir", type=Path, default=None)
    accept.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    accept.add_argument("--policy", choices=ACCEPT_POLICIES, default=POLICY_GLOBAL)
    accept.add_argument("--stratum-tolerance", action="append", default=[], metavar="STRATUM=TOL")
    return parser


def run_sample(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.annotators > 1:
        from llb.goldset.verify_multi.sampling import build_multi_reviewer_worksheets

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


def parse_stratum_tolerances(specs: list[str], parser: argparse.ArgumentParser) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for spec in specs:
        key, separator, value = spec.rpartition("=")
        if not separator or not key:
            parser.error(f"--stratum-tolerance expects STRATUM=TOL, got {spec!r}")
        try:
            overrides[key] = float(value)
        except ValueError:
            parser.error(f"--stratum-tolerance value is not a number: {spec!r}")
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "sample":
        return run_sample(args, parser)
    if args.cmd == "review":
        from llb.goldset.verify_session.loop import run_session

        run_session(
            args.worksheet,
            start=args.start,
            show_crosscheck=args.show_crosscheck,
            clear=args.clear,
            order=args.order,
        )
        return 0
    if args.cmd == "adjudicate":
        from llb.goldset.verify_multi.adjudication import run_adjudicate

        return run_adjudicate(args.bundle, args.worksheet)
    overrides = parse_stratum_tolerances(args.stratum_tolerance, parser)
    return run_accept(
        args.worksheet,
        args.bundle,
        args.out_dir,
        args.tolerance,
        policy=args.policy,
        stratum_tolerances=overrides or None,
    )
