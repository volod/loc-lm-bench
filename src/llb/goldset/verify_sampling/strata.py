"""Stratification keys, exact quotas, and deterministic sample draws."""

import random
from collections.abc import Sequence

from llb.goldset.chains import ChainItem, chain_stratum_key
from llb.goldset.schema import GoldItem


def stratum_key(item: GoldItem) -> str:
    """Return the provenance x split x source-document stratum."""
    return f"{item.provenance}|{item.split}|{item.source_doc_id}"


def stratify(items: Sequence[GoldItem]) -> dict[str, list[GoldItem]]:
    """Group items by stratum while preserving their input order."""
    strata: dict[str, list[GoldItem]] = {}
    for item in items:
        strata.setdefault(stratum_key(item), []).append(item)
    return strata


def stratum_quotas(strata_sizes: dict[str, int], n: int) -> dict[str, int]:
    """Allocate exactly min(n, population) slots with deterministic largest remainders."""
    total = sum(strata_sizes.values())
    budget = min(n, total)
    quotas = {key: 0 for key in strata_sizes}
    allocated = 0
    for key in sorted(strata_sizes, key=lambda k: (-strata_sizes[k], k)):
        if allocated >= budget:
            break
        if strata_sizes[key] > 0:
            quotas[key] = 1
            allocated += 1
    while allocated < budget:
        open_keys = [key for key in strata_sizes if quotas[key] < strata_sizes[key]]
        winner = max(sorted(open_keys), key=lambda k: (n * strata_sizes[k] / total) - quotas[k])
        quotas[winner] += 1
        allocated += 1
    return quotas


def draw_stratified_sample(items: Sequence[GoldItem], n: int, *, seed: int = 13) -> list[GoldItem]:
    """Draw an exact deterministic sample spread across gold-item strata."""
    if n >= len(items):
        return list(items)
    strata = stratify(items)
    rng = random.Random(seed)
    picked: set[int] = set()
    index = {id(item): position for position, item in enumerate(items)}
    quotas = stratum_quotas({key: len(group) for key, group in strata.items()}, n)
    for key, group in sorted(strata.items()):
        order = list(group)
        rng.shuffle(order)
        picked.update(index[id(item)] for item in order[: quotas[key]])
    return [items[position] for position in sorted(picked)]


def draw_chain_sample(chains: Sequence[ChainItem], n: int, *, seed: int = 13) -> list[ChainItem]:
    """Draw an exact deterministic sample spread across chain strata."""
    if n >= len(chains):
        return list(chains)
    strata: dict[str, list[ChainItem]] = {}
    for chain in chains:
        strata.setdefault(chain_stratum_key(chain), []).append(chain)
    rng = random.Random(seed)
    picked: set[int] = set()
    index = {id(chain): position for position, chain in enumerate(chains)}
    quotas = stratum_quotas({key: len(group) for key, group in strata.items()}, n)
    for key, group in sorted(strata.items()):
        order = list(group)
        rng.shuffle(order)
        picked.update(index[id(chain)] for chain in order[: quotas[key]])
    return [chains[position] for position in sorted(picked)]
