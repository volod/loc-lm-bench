"""Sample worksheet creation, additive enlargement, and manifest persistence."""

import json
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.chains import chain_stratum_key, load_chains
from llb.goldset.schema import load_goldset
from llb.goldset.verify_base import (
    KIND_AUTO,
    KIND_CHAINS,
    KIND_GOLDSET,
    SAMPLE_MANIFEST,
    _worksheet_sample_kind,
    bundle_is_synthetic,
    find_chains,
    find_goldset,
    load_worksheet,
    resolve_sample_kind,
    write_worksheet_rows,
)
from llb.goldset.verify_sampling.rows import sample_chain_rows, sample_gold_rows
from llb.goldset.verify_sampling.strata import (
    draw_chain_sample,
    draw_stratified_sample,
    stratum_key,
)


def write_sample_manifest(
    out_path: Path,
    bundle: Path,
    manifest_update: dict[str, object],
    *,
    merge_existing: bool = False,
) -> None:
    manifest_path = Path(out_path).with_name(SAMPLE_MANIFEST)
    manifest: dict[str, object] = {}
    if merge_existing and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    manifest.update({"bundle": str(bundle), "worksheet": str(out_path)})
    manifest.update(manifest_update)
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))


def build_sample_worksheet(
    bundle: Path, out_path: Path, *, n: int, seed: int = 13, kind: str = KIND_AUTO
) -> tuple[int, dict[str, int]]:
    """Draw a stratified sample and persist its worksheet and manifest."""
    bundle = Path(bundle)
    resolved_kind = resolve_sample_kind(bundle, kind)
    synthetic = bundle_is_synthetic(bundle) if resolved_kind == KIND_GOLDSET else False
    strata_sizes: dict[str, int] = {}
    if resolved_kind == KIND_CHAINS:
        chains = load_chains(find_chains(bundle))
        chain_sample = draw_chain_sample(chains, n, seed=seed)
        rows = sample_chain_rows(bundle, chain_sample)
        keys = [chain_stratum_key(chain) for chain in chain_sample]
        sample_size = len(chain_sample)
        population_size = len(chains)
    else:
        items = load_goldset(find_goldset(bundle))
        gold_sample = draw_stratified_sample(items, n, seed=seed)
        rows = sample_gold_rows(bundle, gold_sample, synthetic=synthetic)
        keys = [stratum_key(item) for item in gold_sample]
        sample_size = len(gold_sample)
        population_size = len(items)
    for key in keys:
        strata_sizes[key] = strata_sizes.get(key, 0) + 1
    write_worksheet_rows(out_path, rows)
    write_sample_manifest(
        out_path,
        bundle,
        {
            "kind": resolved_kind,
            "synthetic": synthetic,
            "seed": seed,
            "requested": n,
            "sample_size": sample_size,
            "population": population_size,
            "strata": strata_sizes,
        },
    )
    return sample_size, strata_sizes


def merge_sample_worksheet(
    bundle: Path, out_path: Path, *, n: int, seed: int = 13, kind: str = KIND_AUTO
) -> tuple[int, int]:
    """Enlarge a worksheet additively while preserving all existing decisions."""
    out_path = Path(out_path)
    if not out_path.is_file():
        size, _ = build_sample_worksheet(bundle, out_path, n=n, seed=seed, kind=kind)
        return size, size
    bundle = Path(bundle)
    manifest_kind = _worksheet_sample_kind(out_path)
    resolved_kind = resolve_sample_kind(bundle, manifest_kind if kind == KIND_AUTO else kind)
    existing_rows, fieldnames = load_worksheet(out_path)
    existing_ids = {(row.get("item_id") or "").strip() for row in existing_rows}
    synthetic = bundle_is_synthetic(bundle) if resolved_kind == KIND_GOLDSET else False
    new_rows, population = new_sample_rows(
        bundle, resolved_kind, n, seed, existing_ids, synthetic=synthetic
    )
    if not new_rows:
        return 0, len(existing_rows)
    all_rows = [*existing_rows, *new_rows]
    write_worksheet_rows(out_path, all_rows, fieldnames)
    write_sample_manifest(
        out_path,
        bundle,
        {
            "kind": resolved_kind,
            "synthetic": synthetic,
            "seed": seed,
            "requested": n,
            "sample_size": len(all_rows),
            "population": population,
            "merged_added": len(new_rows),
        },
        merge_existing=True,
    )
    return len(new_rows), len(all_rows)


def new_sample_rows(
    bundle: Path,
    resolved_kind: str,
    n: int,
    seed: int,
    existing_ids: set[str],
    *,
    synthetic: bool,
) -> tuple[list[dict[str, str]], int]:
    if resolved_kind == KIND_CHAINS:
        chains = load_chains(find_chains(bundle))
        chain_sample = draw_chain_sample(chains, n, seed=seed)
        new_chains = [chain for chain in chain_sample if chain.chain_id not in existing_ids]
        return (sample_chain_rows(bundle, new_chains) if new_chains else []), len(chains)
    items = load_goldset(find_goldset(bundle))
    gold_sample = draw_stratified_sample(items, n, seed=seed)
    new_items = [item for item in gold_sample if item.id not in existing_ids]
    return (sample_gold_rows(bundle, new_items, synthetic=synthetic) if new_items else []), len(
        items
    )
