"""Judge calibration statistics (judge calibration statistics, codeable half).

Given paired (human_rating, judge_rating) over the calibration split, compute the
Spearman rank correlation and a bootstrap confidence interval, then decide whether the
judge is trustworthy (rho >= threshold). Producing the judge ratings needs a running
model (RAG core+); this module is pure stats so it can be built and tested now.

No third-party stats deps: Spearman is Pearson over average ranks; CI is a bootstrap.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.contracts.judging import WorksheetItem
from llb.judge.calibration_stats import DEFAULT_THRESHOLD, calibrate
from llb.judge.calibration_worksheet import _LOG, emit_worksheet


# Columns a human edits while rating (everything else is read-only context). Kept here,
# next to the worksheet I/O, so the interactive rater (`llb.judge.rate`) and the merge path
# share one source of truth. `human_status` is pending / rated / skipped.


def _load_ratings(path: Path) -> tuple[list[float], list[float]]:
    rows: list[JsonObject] = []
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = list(csv.DictReader(text.splitlines()))
    human, judge = [], []
    for r in rows:
        if r.get("human_rating") in (None, "") or r.get("judge_rating") in (None, ""):
            continue
        human.append(float(r["human_rating"]))
        judge.append(float(r["judge_rating"]))
    return human, judge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge calibration (worksheet + scoring).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ws = sub.add_parser("worksheet", help="emit a blank calibration worksheet from a gold set")
    ws.add_argument("--goldset", required=True, type=Path)
    ws.add_argument("--out", required=True, type=Path)

    sc = sub.add_parser("score", help="compute rho + CI + trust decision from filled ratings")
    sc.add_argument(
        "--ratings", required=True, type=Path, help="CSV/JSONL with human_rating, judge_rating"
    )
    sc.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    rt = sub.add_parser("rate", help="interactively fill the human columns of a worksheet")
    rt.add_argument("--worksheet", required=True, type=Path, help="pre-filled calibration CSV")
    rt.add_argument("--start", type=int, default=None, help="begin at this 1-based item")
    rt.add_argument(
        "--show-judge",
        action="store_true",
        help="reveal judge_rating (post-hoc only; anchors the rater -- off by default)",
    )
    rt.add_argument(
        "--clear",
        action="store_true",
        help="wipe ALL human columns first and start fresh (confirmation-gated)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "rate":
        from llb.judge.rate.session import run_session

        run_session(
            args.worksheet,
            start=args.start,
            show_judge=args.show_judge,
            clear=args.clear,
        )
        return 0

    if args.cmd == "worksheet":
        from llb.goldset.schema import load_goldset

        items: list[WorksheetItem] = [
            {
                "id": item.id,
                "split": item.split,
                "provenance": item.provenance,
                "question": item.question,
                "reference_answer": item.reference_answer,
            }
            for item in load_goldset(args.goldset)
        ]
        n = emit_worksheet(items, args.out)
        _LOG.info("[calibration] wrote worksheet: %d calibration rows -> %s", n, args.out)
        return 0

    human, judge = _load_ratings(args.ratings)
    if len(human) < 2:
        _LOG.error("[calibration] ERROR: need >= 2 filled rating pairs")
        return 1
    result = calibrate(human, judge, args.threshold)
    _LOG.info(
        "[calibration] rho={rho:.3f} ci=[{ci_low:.3f},{ci_high:.3f}] "
        "n={n} threshold={threshold} trusted={trusted}".format(**result),
    )
    if not result["trusted"]:
        _LOG.info("[calibration] judge NOT trusted -> demote to diagnostic; objective scores rank.")
    return 0


if __name__ == "__main__":
    from llb.core.runtime import run

    sys.exit(run(main))
