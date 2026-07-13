"""Shared-sample creation for independent reviewers."""

import json
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.chains import chain_stratum_key, load_chains
from llb.goldset.schema import load_goldset
from llb.goldset.verify_base import (
    KIND_CHAINS,
    REVIEWER_COL,
    SAMPLE_MANIFEST,
    bundle_is_synthetic,
    find_chains,
    find_goldset,
    resolve_sample_kind,
    write_worksheet_rows,
)
from llb.goldset.verify_multi.common import reviewer_id, reviewer_worksheet_path
from llb.goldset.verify_sampling.rows import sample_chain_rows, sample_gold_rows
from llb.goldset.verify_sampling.strata import (
    draw_chain_sample,
    draw_stratified_sample,
    stratum_key,
)


def build_multi_reviewer_worksheets(
    bundle: Path,
    out_path: Path,
    *,
    n: int,
    annotators: int,
    seed: int = 13,
    kind: str = "auto",
) -> list[Path]:
    """Draw one sample and write identical context rows for every reviewer."""
    if annotators < 2:
        raise ValueError("multi-reviewer sampling needs --annotators >= 2")
    bundle = Path(bundle)
    out_path = Path(out_path)
    resolved_kind = resolve_sample_kind(bundle, kind)
    synthetic = bundle_is_synthetic(bundle) if resolved_kind != KIND_CHAINS else False
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
    paths: list[Path] = []
    for index in range(1, annotators + 1):
        worksheet = reviewer_worksheet_path(out_path, index)
        write_worksheet_rows(
            worksheet,
            [{**row, REVIEWER_COL: reviewer_id(index)} for row in rows],
        )
        paths.append(worksheet)
    manifest = {
        "bundle": str(bundle),
        "kind": resolved_kind,
        "annotators": annotators,
        "worksheets": [str(path) for path in paths],
        "synthetic": synthetic,
        "seed": seed,
        "requested": n,
        "sample_size": sample_size,
        "population": population_size,
        "strata": strata_sizes,
    }
    atomic_write_text(
        out_path.with_name(SAMPLE_MANIFEST), json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return paths
