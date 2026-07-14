"""Focused verify ref ledger implementation."""

import json
from collections.abc import Callable
from pathlib import Path
from llb.goldset.chains import CHAINS_FILENAME, load_chains
from llb.goldset.schema import load_goldset
from llb.goldset.verify_base import (
    GOLDSET_FILENAME,
    VerificationRefStatus,
)
from llb.goldset.verify_ref_format import (
    _accepted_ledger_instruction,
)


def _ledger_ref_status(
    ledger: Path,
    kind: str,
    *,
    noun: str,
    stats_key: str,
    unreadable_label: str,
    load_unverified: Callable[[Path], tuple[int, list[str]]],
    instruction: str,
) -> VerificationRefStatus:
    """Shared flat/chain ledger check: readable, non-empty, and every entry verified."""
    try:
        total, unverified_ids = load_unverified(ledger)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return VerificationRefStatus(
            False, ledger, kind, f"unreadable {unreadable_label}: {exc}", instruction=instruction
        )
    stats: dict[str, object] = {
        stats_key: total,
        "verified": total - len(unverified_ids),
        "unverified": len(unverified_ids),
        "unverified_item_ids": unverified_ids[:10],
    }
    if not total:
        return VerificationRefStatus(
            False,
            ledger,
            kind,
            f"ledger has no {stats_key}",
            stats=stats,
            instruction=instruction,
        )
    if unverified_ids:
        return VerificationRefStatus(
            False,
            ledger,
            kind,
            f"ledger contains unverified {noun}(s)",
            stats=stats,
            instruction=instruction,
        )
    return VerificationRefStatus(True, ledger, kind, stats=stats)


def _load_unverified_chains(ledger: Path) -> tuple[int, list[str]]:
    chains = load_chains(ledger)
    return len(chains), [chain.chain_id for chain in chains if not chain.verified]


def _load_unverified_items(ledger: Path) -> tuple[int, list[str]]:
    items = load_goldset(ledger)
    return len(items), [item.id for item in items if not item.verified]


def _check_accepted_ledger_ref(path: Path) -> VerificationRefStatus:
    ledger = path / GOLDSET_FILENAME if path.is_dir() else path
    instruction = _accepted_ledger_instruction(path)
    chain_ledger = path / CHAINS_FILENAME if path.is_dir() else path
    if not ledger.is_file() and chain_ledger.name == CHAINS_FILENAME and chain_ledger.is_file():
        return _ledger_ref_status(
            chain_ledger,
            "accepted_chain_ledger",
            noun="chain",
            stats_key="chains",
            unreadable_label="chain ledger",
            load_unverified=_load_unverified_chains,
            instruction=instruction,
        )
    return _ledger_ref_status(
        ledger,
        "accepted_ledger",
        noun="item",
        stats_key="items",
        unreadable_label="ledger",
        load_unverified=_load_unverified_items,
        instruction=instruction,
    )
