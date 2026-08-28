"""Read the paired variance a confirmation run's screen size is priced from.

The reference is two ordinary scored run bundles -- any earlier pair whose per-case objective
scores describe the kind of model-to-model gap the confirmation is about to measure. Pairing is the
shared one (`llb.eval.paired_cases`), so a reference read here and a lane read anywhere else in the
repo agree on the item set or fail loudly instead of comparing different ones.
"""

from pathlib import Path

from llb.eval.paired_cases import SCORES_FILENAME, recorded_lane_rows, rows_by_item
from llb.optimize.joint_search.long_run.uncertainty import QUALITY_COLUMN


def paired_reference_deltas(candidate: Path, baseline: Path) -> list[float]:
    """Candidate-minus-baseline per-case objective scores over the items both bundles scored."""
    lanes = {
        "candidate": rows_by_item(recorded_lane_rows([candidate])),
        "baseline": rows_by_item(recorded_lane_rows([baseline])),
    }
    shared = sorted(set(lanes["candidate"]) & set(lanes["baseline"]))
    if len(shared) < 2:
        raise ValueError(
            f"the power reference needs at least two items scored by both {candidate} and "
            f"{baseline} (each must hold a {SCORES_FILENAME}); found {len(shared)}"
        )
    return [
        float(lanes["candidate"][item].get(QUALITY_COLUMN, 0.0) or 0.0)
        - float(lanes["baseline"][item].get(QUALITY_COLUMN, 0.0) or 0.0)
        for item in shared
    ]


__all__ = ["paired_reference_deltas"]
