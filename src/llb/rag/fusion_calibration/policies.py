"""Threshold-grid parsing for deterministic fusion routing calibration."""

from llb.rag.fusion_routing import HeuristicPolicy


def policy_grid(
    long_question_words: tuple[int, ...], min_linked_entities: tuple[int, ...]
) -> tuple[HeuristicPolicy, ...]:
    """Cross and de-duplicate the two threshold axes in stable order."""
    return tuple(
        dict.fromkeys(
            HeuristicPolicy(words, entities)
            for words in long_question_words
            for entities in min_linked_entities
        )
    )


def parse_thresholds(spec: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    """Parse a comma-separated integer threshold grid."""
    values: list[int] = []
    minimum = 0 if allow_zero else 1
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            raise ValueError(f"threshold must be an integer, got {token!r}") from None
        if value < minimum:
            raise ValueError(f"threshold must be at least {minimum}, got {value}")
        values.append(value)
    if not values:
        raise ValueError("no threshold parsed from the grid spec")
    return tuple(dict.fromkeys(values))
