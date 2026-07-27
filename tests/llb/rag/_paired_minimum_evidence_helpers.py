"""paired-reading-minimum-evidence-gate -- a reading no item count could support is not one.

A percentile bootstrap of a handful of DISCORDANT items happily prints `+1.000 [+1.000, +1.000]`
beside its own exact sign-test p of 0.5: resampling two differing items can only ever draw those
two. The gate refuses to call that a separation, and the bound it refuses at is derived from the
reporting confidence -- the fewest differing items the exact two-sided sign test could reach that
level with -- so nothing here is fitted.

Covered: the derived bound and its behaviour at the 5 / 6 / 7 boundary, that only the CLAIM is
gated (never the null reading, never the numbers), and the verdict guard in every lane that cuts a
`lo > 0` reading.

Pure: value vectors and dict rows, so the whole vertical runs in the lightweight CI install.
"""

from llb.rag.fusion_evidence.evidence_gate import minimum_discordant_pairs
from llb.rag.fusion_evidence.paired import paired_comparison
from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    bootstrap_index_sets,
)

RESAMPLES = 2000

SEED = 13

BOUND = minimum_discordant_pairs(DEFAULT_CONFIDENCE)


def _unanimous(wins: int, n: int = 40):
    """A paired row where `wins` items differ, ALL in the candidate's favour -- the best case.

    The most extreme arrangement there is, so if the reporting level is unreachable here it is
    unreachable for any data with that many differing items.
    """
    candidate = [1.0 if i < wins else 0.0 for i in range(n)]
    baseline = [0.0] * n
    return paired_comparison(
        candidate, baseline, bootstrap_index_sets(n, RESAMPLES, SEED), DEFAULT_CONFIDENCE
    )


def _lane_rows(wins: int, n: int = 40):
    """Per-item candidate/baseline vectors with `wins` unanimous differences."""
    return [1.0 if i < wins else 0.0 for i in range(n)], [0.0] * n
