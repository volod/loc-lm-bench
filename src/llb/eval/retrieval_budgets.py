"""Retrieval budget selections shared by the lanes that sweep `top_k` END TO END.

Two lanes vary the same knob for opposite questions. The embedder adoption-bar sweep shrinks the
budget to ask whether a first-hit-RANK gain still binds when the context has no room to spare; the
answer-quality lane raises it to ask whether a multi-hop COVERAGE gain survives the context bill it
arrives with. Parsing the selection in one place is what keeps `10,50` meaning exactly one thing in
both artifacts.
"""


def parse_top_ks(spec: str) -> list[int]:
    """Parse `10,50` into retrieval budgets, de-duplicated in the order given."""
    values: list[int] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        try:
            budget = int(token)
        except ValueError:
            raise ValueError(f"top_k must be an integer, got {token!r}") from None
        if budget < 1:
            raise ValueError(f"top_k must be at least 1, got {budget}")
        if budget not in values:
            values.append(budget)
    if not values:
        raise ValueError("name at least one top_k budget")
    return values


__all__ = ["parse_top_ks"]
