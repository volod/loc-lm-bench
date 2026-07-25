"""How much repetition a built store still holds AFTER exact-duplicate collapse.

Exact collapse (`llb.rag.duplicates`) removes byte-identical chunks. Converted-PDF furniture is
not always byte-identical: the footer repeats with a page number in it, the table header repeats
with one changed cell, the heading repeats with different whitespace after conversion. Those
chunks keep their own index rows and score NEAR-ties instead of exact ties, so they still crowd
the top-k and can still sit inside the measurement floor (`llb.rag.noise_floor`).

This module measures that residue on the chunks a store actually indexed, along two independent
axes:

- TEXT: re-measure the duplicate rate at each coarser tier (`llb.rag.duplicate_tiers`). The
  residue a cheap normalizer would collapse is exactly `tier.collapsed`.
- EMBEDDING: count chunk pairs whose stored vectors sit above a cosine band, which is what the
  ranking actually sees, and cross-tabulate them against the text tiers. A band with many pairs
  the text tiers do NOT merge says a normalized tier cannot reach that residue at all.

The digit-masking tier merges texts that differ only in a number, which is a page footer on one
corpus and two rows of a rate table on another. `digit_merge_examples` samples exactly those
pairs so the false-merge rate is read rather than assumed.
"""

from typing import Any

from typing_extensions import TypedDict

from llb.core.contracts.rag import ChunkRecord
from llb.rag.duplicate_tiers import (
    COARSE_TIERS,
    DUPLICATE_TIERS,
    TIER_MASKED,
    TIER_NORMALIZED,
    duplicate_key,
)
from llb.rag.duplicates import DuplicateStats, duplicate_stats

# Cosine bands to report. 0.99+ is "the ranking cannot tell these apart"; 0.95 is still a very
# close neighbour for a sentence encoder over Ukrainian prose.
DEFAULT_THRESHOLDS = (0.999, 0.99, 0.95)

DEFAULT_EXAMPLES = 8

# Rows of the similarity matrix computed at once (a block is `block x n_indexed` float32).
DEFAULT_BLOCK = 512

_TEXT_PREVIEW_CHARS = 120


class NearDuplicateBand(TypedDict):
    """Chunk pairs above one cosine band, and how many of them a TEXT tier would merge."""

    threshold: float
    pairs: int
    chunks: int  # chunks with at least one neighbour in the band
    chunk_share: float
    normalized_pairs: int  # of `pairs`, those the `normalized` tier merges
    masked_pairs: int  # of `pairs`, those the `masked` tier merges


class ResiduePair(TypedDict):
    """One near-duplicate pair, for a human reading of what the residue actually is."""

    cosine: float
    same_document: bool
    normalized_equal: bool
    masked_equal: bool
    a: str  # "<doc_id>@<char_start>: <text preview>"
    b: str


class ResidueReport(TypedDict):
    """Post-collapse residue of one store: the text tiers, the cosine bands, and samples."""

    n_indexed: int
    store_tier: str
    tiers: dict[str, DuplicateStats]
    bands: list[NearDuplicateBand]
    near_duplicate_examples: list[ResiduePair]  # top-cosine pairs no text tier merges
    digit_merge_examples: list[ResiduePair]  # pairs ONLY the digit-masking tier merges


def measure_duplicate_residue(
    chunks: list[ChunkRecord],
    vectors: Any,
    store_tier: str,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    examples: int = DEFAULT_EXAMPLES,
    block: int = DEFAULT_BLOCK,
) -> ResidueReport:
    """Measure text-tier and embedding residue over a built store's indexed chunks."""
    import numpy as np

    # The exact tier's key IS the chunk text, so only the coarse tiers are keyed here.
    keys = {tier: [duplicate_key(chunk["text"], tier) for chunk in chunks] for tier in COARSE_TIERS}
    codes = {tier: _codes(values) for tier, values in keys.items()}
    unit = _unit_rows(np.asarray(vectors, dtype="float32"))
    bands, found = _scan_bands(unit, codes, sorted(thresholds, reverse=True), block, examples)
    return {
        "n_indexed": len(chunks),
        "store_tier": store_tier,
        "tiers": {tier: duplicate_stats(chunks, tier) for tier in DUPLICATE_TIERS},
        "bands": bands,
        "near_duplicate_examples": [_pair(chunks, keys, *hit) for hit in found],
        "digit_merge_examples": [
            _pair(chunks, keys, i, j, None) for i, j in _digit_merges(keys, examples)
        ],
    }


def _unit_rows(vectors: Any) -> Any:
    """L2-normalized rows, so a dot product IS the cosine (a store may hold unnormalized rows)."""
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _codes(keys: list[str]) -> Any:
    """Key strings as small ints, so pair-wise key equality is one vectorized comparison."""
    import numpy as np

    seen: dict[str, int] = {}
    return np.array([seen.setdefault(key, len(seen)) for key in keys], dtype="int64")


def _scan_bands(
    unit: Any,
    codes: dict[str, Any],
    thresholds: list[float],
    block: int,
    examples: int,
) -> tuple[list[NearDuplicateBand], list[tuple[int, int, float]]]:
    """One blocked pass over the upper triangle: per-band counts plus the top unmatched pairs."""
    import numpy as np

    n = unit.shape[0]
    counts = {t: [0, 0, 0] for t in thresholds}  # pairs, normalized-equal, masked-equal
    neighbours = {t: np.zeros(n, dtype=bool) for t in thresholds}
    top: list[tuple[float, int, int]] = []
    lowest = thresholds[-1]
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = unit[start:stop] @ unit.T
        rows = np.arange(start, stop)[:, None]
        upper = np.arange(n)[None, :] > rows  # each pair counted once
        same_normalized = codes[TIER_NORMALIZED][start:stop][:, None] == codes[TIER_NORMALIZED]
        same_masked = codes[TIER_MASKED][start:stop][:, None] == codes[TIER_MASKED]
        for threshold in thresholds:
            hits = upper & (sims >= threshold)
            counts[threshold][0] += int(hits.sum())
            counts[threshold][1] += int((hits & same_normalized).sum())
            counts[threshold][2] += int((hits & same_masked).sum())
            touched = neighbours[threshold]
            touched[start:stop] |= hits.any(axis=1)
            touched |= hits.any(axis=0)
        rows_idx, cols_idx = np.nonzero(upper & (sims >= lowest) & ~same_masked)
        if rows_idx.size:
            values = sims[rows_idx, cols_idx]
            best = np.argsort(-values)[:examples]
            top.extend((float(values[o]), int(start + rows_idx[o]), int(cols_idx[o])) for o in best)
            top = sorted(top, reverse=True)[:examples]
    bands: list[NearDuplicateBand] = [
        {
            "threshold": threshold,
            "pairs": counts[threshold][0],
            "chunks": int(neighbours[threshold].sum()),
            "chunk_share": round(float(neighbours[threshold].mean()), 4) if n else 0.0,
            "normalized_pairs": counts[threshold][1],
            "masked_pairs": counts[threshold][2],
        }
        for threshold in thresholds
    ]
    return bands, [(i, j, cosine) for cosine, i, j in top]


def _digit_merges(keys: dict[str, list[str]], limit: int) -> list[tuple[int, int]]:
    """Pairs the `masked` tier merges but `normalized` does not -- its whole added reach.

    One pair per masked group (its anchor plus the first member whose normalized key differs), so
    a footer repeated fifty times contributes one sample rather than fifty.
    """
    anchors: dict[str, int] = {}
    sampled: set[str] = set()
    pairs: list[tuple[int, int]] = []
    for position, masked in enumerate(keys[TIER_MASKED]):
        anchor = anchors.setdefault(masked, position)
        if anchor == position or masked in sampled:
            continue
        if keys[TIER_NORMALIZED][anchor] != keys[TIER_NORMALIZED][position]:
            sampled.add(masked)
            pairs.append((anchor, position))
            if len(pairs) >= limit:
                break
    return pairs


def _pair(
    chunks: list[ChunkRecord],
    keys: dict[str, list[str]],
    i: int,
    j: int,
    cosine: float | None,
) -> ResiduePair:
    return {
        "cosine": round(cosine, 4) if cosine is not None else -1.0,
        "same_document": chunks[i]["doc_id"] == chunks[j]["doc_id"],
        "normalized_equal": keys[TIER_NORMALIZED][i] == keys[TIER_NORMALIZED][j],
        "masked_equal": keys[TIER_MASKED][i] == keys[TIER_MASKED][j],
        "a": _preview(chunks[i]),
        "b": _preview(chunks[j]),
    }


def _preview(chunk: ChunkRecord) -> str:
    text = " ".join(str(chunk["text"]).split())[:_TEXT_PREVIEW_CHARS]
    return f"{chunk['doc_id']}@{chunk['char_start']}: {text}"


def format_residue_report(report: ResidueReport) -> str:
    """ASCII lines for the residue report (AGENTS.md: ASCII-only, no box-drawing)."""
    lines = [
        f"[duplicate-residue] {report['n_indexed']} indexed chunks "
        f"(store tier: {report['store_tier']})",
        "  text tiers (what each tier would collapse from here):",
    ]
    for tier, stats in report["tiers"].items():
        lines.append(
            f"    {tier.ljust(10)} {stats['collapsed']:6d} collapsible "
            f"({stats['duplicate_share']:6.1%} of chunks in {stats['groups']} groups, "
            f"{stats['intra_document_groups']} intra / {stats['cross_document_groups']} cross, "
            f"largest {stats['largest_group']})"
        )
    lines.append("  cosine bands (pairs the ranking can barely tell apart):")
    for band in report["bands"]:
        lines.append(
            f"    >={band['threshold']:.3f}   {band['pairs']:7d} pairs over "
            f"{band['chunks']:6d} chunks ({band['chunk_share']:.1%}); "
            f"normalized merges {band['normalized_pairs']}, masked merges {band['masked_pairs']}"
        )
    lines.extend(
        _example_lines(
            "near-duplicate pairs no text tier merges", report["near_duplicate_examples"]
        )
    )
    lines.extend(_example_lines("pairs ONLY digit masking merges", report["digit_merge_examples"]))
    return "\n".join(lines)


def _example_lines(title: str, pairs: list[ResiduePair]) -> list[str]:
    if not pairs:
        return [f"  {title}: none"]
    lines = [f"  {title}:"]
    for pair in pairs:
        cosine = f"cos {pair['cosine']:.4f}" if pair["cosine"] >= 0 else "cos n/a"
        lines.append(f"    [{cosine}] {pair['a']}")
        lines.append(f"              {pair['b']}")
    return lines
