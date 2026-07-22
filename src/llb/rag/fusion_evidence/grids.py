"""Parse the four sweep-grid specs `compare-graph-fusion` accepts on the command line.

One module per concern: `rows.py` decides which (weight, depth, identity, threshold) points are
worth a row, this one turns the operator's comma-separated text into those axes and rejects a
value the fusion knobs cannot honor -- so the CLI never has to validate a knob itself.
"""

from llb.rag.fusion_spans import (
    SPAN_IDENTITIES,
    resolve_merge_ratio,
    resolve_span_identity,
)


def parse_span_identities(spec: str) -> tuple[str, ...]:
    """Parse an `exact,overlap` span-identity grid; raises `ValueError` on an unknown policy."""
    identities = [
        resolve_span_identity(token.strip()) for token in spec.split(",") if token.strip()
    ]
    if not identities:
        raise ValueError(
            f"no span identity parsed from the grid spec (expected {', '.join(SPAN_IDENTITIES)})"
        )
    return tuple(dict.fromkeys(identities))


def parse_merge_ratios(spec: str) -> tuple[float, ...]:
    """Parse a `0.25,0.5,1.0` merge-threshold grid; `1.0` is the containment-only setting."""
    ratios = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ratio = float(token)
        except ValueError:
            raise ValueError(f"span merge ratio must be a number, got {token!r}") from None
        ratios.append(resolve_merge_ratio(ratio))
    if not ratios:
        raise ValueError("no span merge ratio parsed from the grid spec")
    return tuple(dict.fromkeys(ratios))


def parse_candidates(spec: str) -> tuple[int | None, ...]:
    """Parse a `10,50` candidate-depth grid; raises `ValueError` on a non-positive entry.

    `k` names the sweep's own cutoff (the historical depth), so `k,50` compares the shallow pool
    against a deeper one without the caller having to repeat the `--k` value.
    """
    depths: list[int | None] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "k":
            depths.append(None)
            continue
        try:
            depth = int(token)
        except ValueError:
            raise ValueError(f"candidate depth must be an integer or 'k', got {token!r}") from None
        if depth < 1:
            raise ValueError(f"fusion candidate depth must be at least 1, got {depth}")
        depths.append(depth)
    if not depths:
        raise ValueError("no candidate depth parsed from the grid spec")
    return tuple(dict.fromkeys(depths))


def parse_weights(spec: str) -> tuple[float, ...]:
    """Parse a `0,0.3,1` graph-weight grid; raises `ValueError` on an out-of-range entry."""
    weights = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        weight = float(token)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"graph weight must be within [0, 1], got {weight}")
        weights.append(weight)
    if not weights:
        raise ValueError("no graph weight parsed from the grid spec")
    return tuple(dict.fromkeys(weights))
