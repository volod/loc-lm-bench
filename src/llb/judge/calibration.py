"""Judge calibration statistics (M0.5, codeable half).

Given paired (human_rating, judge_rating) over the calibration split, compute the
Spearman rank correlation and a bootstrap confidence interval, then decide whether the
judge is trustworthy (rho >= threshold). Producing the judge ratings needs a running
model (Milestone 1+); this module is pure stats so it can be built and tested now.

No third-party stats deps: Spearman is Pearson over average ranks; CI is a bootstrap.
"""

import argparse
import csv
import json
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from llb.contracts import CalibrationResult, JsonObject, WorksheetItem
from llb.goldset.schema import GoldItem

DEFAULT_THRESHOLD = 0.6
_LOG = logging.getLogger(__name__)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
    den_b = sum((y - mean_b) ** 2 for y in b) ** 0.5
    if den_a == 0 or den_b == 0:
        return 0.0
    return float(num / (den_a * den_b))


def spearman_rho(human: list[float], judge: list[float]) -> float:
    if len(human) != len(judge):
        raise ValueError("human and judge ratings must be the same length")
    if len(human) < 2:
        raise ValueError("need >= 2 paired ratings")
    return _pearson(_average_ranks(human), _average_ranks(judge))


def bootstrap_ci(
    human: list[float],
    judge: list[float],
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = random.Random(seed)
    m = len(human)
    rhos: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(m) for _ in range(m)]
        sample_h = [human[i] for i in idx]
        sample_j = [judge[i] for i in idx]
        try:
            rhos.append(spearman_rho(sample_h, sample_j))
        except ValueError:
            continue
    if not rhos:
        return (0.0, 0.0)
    rhos.sort()
    lo = rhos[int((alpha / 2) * len(rhos))]
    hi = rhos[min(len(rhos) - 1, int((1 - alpha / 2) * len(rhos)))]
    return (lo, hi)


def calibrate(
    human: list[float], judge: list[float], threshold: float = DEFAULT_THRESHOLD
) -> CalibrationResult:
    rho = spearman_rho(human, judge)
    lo, hi = bootstrap_ci(human, judge)
    return {
        "rho": rho,
        "ci_low": lo,
        "ci_high": hi,
        "n": len(human),
        "threshold": threshold,
        "trusted": rho >= threshold,
    }


WORKSHEET_COLS = [
    "item_id",
    "split",
    "question",
    "reference_answer",
    "model_answer",
    "human_rating",
    "judge_rating",
]


def emit_worksheet(items: list[WorksheetItem], out_path: Path) -> int:
    """Write a blank CSV worksheet (one row per calibration item) for the human to fill.

    Columns model_answer / human_rating / judge_rating are blank. Use
    `write_filled_worksheet` instead to pre-fill model_answer from a `run-eval` run.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=WORKSHEET_COLS)
        writer.writeheader()
        for it in items:
            if it.get("split") != "calibration":
                continue
            writer.writerow(
                {
                    "item_id": it["id"],
                    "split": it["split"],
                    "question": it["question"],
                    "reference_answer": it["reference_answer"],
                    "model_answer": "",
                    "human_rating": "",
                    "judge_rating": "",
                }
            )
            n += 1
    return n


def write_filled_worksheet(
    answers: Sequence[tuple[GoldItem, str]],
    out_path: Path,
    judge_ratings: Sequence[float] | None = None,
) -> int:
    """Write a worksheet with model_answer pre-filled from a run; ratings left blank.

    `answers` is a list of `(gold_item, model_answer)` (gold_item duck-typed:
    `id / split / question / reference_answer`). Produced by `run-eval --worksheet` on the
    calibration split so the human only fills `human_rating`.

    When `judge_ratings` is supplied (aligned with `answers`), the `judge_rating` column is
    pre-filled with the JUDGE's score per item -- so the calibration worksheet carries both the
    judge rating and a blank human rating, and `calibration score` can compute rho(human, judge)
    once the human column is filled.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=WORKSHEET_COLS)
        writer.writeheader()
        for i, (item, answer) in enumerate(answers):
            judge = "" if judge_ratings is None else round(float(judge_ratings[i]), 4)
            writer.writerow(
                {
                    "item_id": item.id,
                    "split": item.split,
                    "question": item.question,
                    "reference_answer": item.reference_answer,
                    "model_answer": answer or "",
                    "human_rating": "",
                    "judge_rating": judge,
                }
            )
            n += 1
    return n


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

    args = parser.parse_args(argv)

    if args.cmd == "worksheet":
        from llb.goldset.schema import load_goldset

        items: list[WorksheetItem] = [
            {
                "id": item.id,
                "split": item.split,
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
    from llb.runtime import run

    sys.exit(run(main))
