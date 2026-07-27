"""re-decide-the-relabeled-fusion-and-bakeoff-readings -- a withdrawn reading says what it needs.

The minimum-evidence gate turns a claim that rests on too few differing items into
`insufficient_evidence`. That is a statement about the ITEM SET, not about the difference, so a
recommendation an operator may still be acting on must not be left reading like a measured tie.
Every withdrawn row is therefore priced from its own discordance rate: `d` differing items out of
`n` extrapolate to the item count at which the reporting level becomes reachable at all.

Covered: the arithmetic and its two boundaries (already reachable, nothing differing), that the
floor moves with the reporting convention exactly as the bound it inverts does, the shared clause
every lane appends, the per-row column in the boundary table, and the recorded open questions whose
prices the current docs quote -- including the encoder row, which no committed goldset can reach.

Pure: value vectors and dict rows, so the whole vertical runs in the lightweight CI install.
"""

from pathlib import Path


from llb.rag.fusion_evidence.evidence_gate import (
    minimum_discordant_pairs,
)


from llb.rag.fusion_evidence.stats import (
    DEFAULT_CONFIDENCE,
    bootstrap_index_sets,
)


from llb.rag.fusion_evidence.paired import (
    paired_comparison,
)


RESAMPLES = 2000


SEED = 13


BOUND = minimum_discordant_pairs(DEFAULT_CONFIDENCE)


GOLDSETS = Path(__file__).resolve().parents[3] / "samples" / "goldsets"


def _unanimous(wins: int, n: int = 40):
    """A paired row where `wins` items differ, all in the candidate's favour."""
    candidate = [1.0 if i < wins else 0.0 for i in range(n)]
    return paired_comparison(
        candidate, [0.0] * n, bootstrap_index_sets(n, RESAMPLES, SEED), DEFAULT_CONFIDENCE
    )
