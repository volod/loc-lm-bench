"""Judge calibration statistics (judge calibration statistics, codeable half).

Given paired (human_rating, judge_rating) over the calibration split, compute the
Spearman rank correlation and a bootstrap confidence interval, then decide whether the
judge is trustworthy (rho >= threshold). Producing the judge ratings needs a running
model (RAG core+); this module is pure stats so it can be built and tested now.

No third-party stats deps: Spearman is Pearson over average ranks; CI is a bootstrap.
"""

import argparse
import csv
import io
import json
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from llb.contracts import CalibrationResult, JsonObject, WorksheetItem
from llb.fsutil import atomic_write_text
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


# Columns a human edits while rating (everything else is read-only context). Kept here,
# next to the worksheet I/O, so the interactive rater (`llb.judge.rate`) and the merge path
# share one source of truth. `human_status` is pending / rated / skipped.
HUMAN_COLS = ["human_answer", "human_rating", "human_note", "human_status"]

WORKSHEET_COLS = [
    "item_id",
    "split",
    "provenance",
    "question",
    "reference_answer",
    "model_answer",
    "human_answer",
    "human_rating",
    "human_note",
    "human_status",
    "judge_rating",
]


def worksheet_fieldnames(existing: Sequence[str] | None = None) -> list[str]:
    """Canonical column order: keep any columns already in the header, then append any
    `WORKSHEET_COLS` that are missing. With no header yet, the order is `WORKSHEET_COLS`."""
    names = list(existing) if existing else []
    for col in WORKSHEET_COLS:
        if col not in names:
            names.append(col)
    return names


def load_worksheet(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load a worksheet CSV into `(rows, fieldnames)`.

    Any `WORKSHEET_COLS` column missing from the header is added blank, so callers can rely on
    every column being present; any extra columns are preserved in `fieldnames` so a round-trip
    never drops data.
    """
    text = Path(path).read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    fieldnames = worksheet_fieldnames(reader.fieldnames)
    rows = [{name: (raw.get(name) or "") for name in fieldnames} for raw in reader]
    return rows, fieldnames


def write_worksheet_rows(
    out_path: Path,
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str] | None = None,
) -> int:
    """Atomically (re)write the whole worksheet, preserving column order.

    The CSV is the worksheet's only state, so every edit rewrites it through a temp file +
    `os.replace` (`atomic_write_text`); a crash mid-write leaves the prior file intact.
    """
    columns = list(fieldnames) if fieldnames else list(WORKSHEET_COLS)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in columns})
    atomic_write_text(Path(out_path), buf.getvalue())
    return len(rows)


def emit_worksheet(items: list[WorksheetItem], out_path: Path) -> int:
    """Write a blank CSV worksheet (one row per calibration item) for the human to fill.

    `model_answer`, the human columns, and `judge_rating` are blank; `provenance` is copied
    from the item. Use `write_filled_worksheet` instead to pre-fill `model_answer` from a run.
    """
    rows = [
        {
            "item_id": it["id"],
            "split": it["split"],
            "provenance": it.get("provenance", "") or "",
            "question": it["question"],
            "reference_answer": it["reference_answer"],
        }
        for it in items
        if it.get("split") == "calibration"
    ]
    return write_worksheet_rows(out_path, rows)


def _existing_rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    """Index a prior worksheet by `item_id` for the merge-on-regenerate path (empty if none)."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        rows, _ = load_worksheet(path)
    except (OSError, csv.Error):
        return {}
    return {row["item_id"]: row for row in rows if row.get("item_id")}


def _merge_human_columns(new_row: dict[str, str], prev_row: dict[str, str]) -> None:
    """Carry a prior run's human columns into a freshly pre-filled row.

    Human work survives a re-run with the same deterministic candidate. If the regenerated
    `model_answer` CHANGED (a different candidate), the human rating no longer applies to the
    shown answer, so it is cleared with a warning; the human's OWN authored answer/note are
    kept (they do not depend on the candidate).
    """
    for col in HUMAN_COLS:
        prev_val = prev_row.get(col, "")
        if prev_val:
            new_row[col] = prev_val
    answer_changed = prev_row.get("model_answer", "") != new_row.get("model_answer", "")
    if answer_changed and new_row.get("human_rating"):
        _LOG.warning(
            "[calibration] item %s: model_answer changed since the last rating; clearing the "
            "stale human_rating (was %r) -- re-rate against the new answer.",
            new_row.get("item_id", "?"),
            new_row["human_rating"],
        )
        new_row["human_rating"] = ""
        if new_row.get("human_status") == "rated":
            new_row["human_status"] = "pending"


def write_filled_worksheet(
    answers: Sequence[tuple[GoldItem, str]],
    out_path: Path,
    judge_ratings: Sequence[float] | None = None,
) -> int:
    """Write a worksheet with model_answer pre-filled from a run; human columns left blank.

    `answers` is a list of `(gold_item, model_answer)` (gold_item duck-typed:
    `id / split / question / reference_answer / provenance`). Produced by `run-eval --worksheet`
    on the calibration split so the human authors `human_answer` + `human_rating`.

    When `judge_ratings` is supplied (aligned with `answers`), the `judge_rating` column is
    pre-filled with the JUDGE's score per item -- so the calibration worksheet carries both the
    judge rating and a blank human rating, and `calibration score` can compute rho(human, judge)
    once the human column is filled.

    Re-running MERGES any prior human columns by `item_id` (never clobbers them); a row whose
    regenerated `model_answer` changed has its stale rating cleared (see `_merge_human_columns`).
    """
    existing = _existing_rows_by_id(out_path)
    rows: list[dict[str, str]] = []
    for i, (item, answer) in enumerate(answers):
        judge = "" if judge_ratings is None else str(round(float(judge_ratings[i]), 4))
        row = {
            "item_id": item.id,
            "split": item.split,
            "provenance": getattr(item, "provenance", "") or "",
            "question": item.question,
            "reference_answer": item.reference_answer,
            "model_answer": answer or "",
            "human_answer": "",
            "human_rating": "",
            "human_note": "",
            "human_status": "",
            "judge_rating": judge,
        }
        prev = existing.get(str(item.id))
        if prev is not None:
            _merge_human_columns(row, prev)
        rows.append(row)
    return write_worksheet_rows(out_path, rows)


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
        from llb.judge.rate import run_session

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
    from llb.runtime import run

    sys.exit(run(main))
