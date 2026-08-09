"""Shared-sample creation for independent reviewers."""

from dataclasses import dataclass
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
from llb.goldset.verify_sampling.planning import (
    DEFAULT_EXPECTED_REJECT_RATE,
    DEFAULT_SAMPLE_CONFIDENCE,
    DEFAULT_SAMPLE_PRECISION,
    SampleSizePlan,
    sample_size_plan,
)
from llb.goldset.verify_sampling.rows import sample_chain_rows, sample_gold_rows
from llb.goldset.verify_sampling.strata import (
    draw_chain_sample,
    draw_stratified_sample,
    stratum_key,
)


@dataclass(frozen=True, slots=True)
class _DrawnSample:
    """One drawn review sample: the rows reviewers see, and what the manifest has to state."""

    rows: list[dict[str, str]]
    keys: list[str]
    sample_size: int
    population_size: int
    plan: SampleSizePlan


def _draw_chain_sample(
    bundle: Path, *, n: int | None, seed: int, plan_options: dict[str, float]
) -> _DrawnSample:
    """Draw a multi-hop CHAIN sample: the unit under review is the chain, not the item."""
    chains = load_chains(find_chains(bundle))
    plan = sample_size_plan(
        len(chains),
        len({chain_stratum_key(chain) for chain in chains}),
        requested_size=n,
        **plan_options,
    )
    drawn = draw_chain_sample(chains, int(plan["selected_target"]), seed=seed)
    return _DrawnSample(
        rows=sample_chain_rows(bundle, drawn),
        keys=[chain_stratum_key(chain) for chain in drawn],
        sample_size=len(drawn),
        population_size=len(chains),
        plan=plan,
    )


def _draw_gold_sample(
    bundle: Path, *, n: int | None, seed: int, synthetic: bool, plan_options: dict[str, float]
) -> _DrawnSample:
    """Draw a stratified GOLD-ITEM sample."""
    items = load_goldset(find_goldset(bundle))
    plan = sample_size_plan(
        len(items),
        len({stratum_key(item) for item in items}),
        requested_size=n,
        **plan_options,
    )
    drawn = draw_stratified_sample(items, int(plan["selected_target"]), seed=seed)
    return _DrawnSample(
        rows=sample_gold_rows(bundle, drawn, synthetic=synthetic),
        keys=[stratum_key(item) for item in drawn],
        sample_size=len(drawn),
        population_size=len(items),
        plan=plan,
    )


def _write_reviewer_worksheets(
    out_path: Path, rows: list[dict[str, str]], annotators: int
) -> list[Path]:
    """Identical context rows per reviewer -- the whole point of a multi-annotator draw."""
    paths: list[Path] = []
    for index in range(1, annotators + 1):
        worksheet = reviewer_worksheet_path(out_path, index)
        write_worksheet_rows(worksheet, [{**row, REVIEWER_COL: reviewer_id(index)} for row in rows])
        paths.append(worksheet)
    return paths


def build_multi_reviewer_worksheets(
    bundle: Path,
    out_path: Path,
    *,
    n: int | None,
    annotators: int,
    seed: int = 13,
    kind: str = "auto",
    confidence: float = DEFAULT_SAMPLE_CONFIDENCE,
    precision: float = DEFAULT_SAMPLE_PRECISION,
    expected_reject_rate: float = DEFAULT_EXPECTED_REJECT_RATE,
) -> list[Path]:
    """Draw one sample and write identical context rows for every reviewer."""
    if annotators < 2:
        raise ValueError("multi-reviewer sampling needs --annotators >= 2")
    bundle = Path(bundle)
    out_path = Path(out_path)
    resolved_kind = resolve_sample_kind(bundle, kind)
    synthetic = bundle_is_synthetic(bundle) if resolved_kind != KIND_CHAINS else False
    plan_options = {
        "confidence": confidence,
        "precision": precision,
        "expected_reject_rate": expected_reject_rate,
    }
    sample = (
        _draw_chain_sample(bundle, n=n, seed=seed, plan_options=plan_options)
        if resolved_kind == KIND_CHAINS
        else _draw_gold_sample(
            bundle, n=n, seed=seed, synthetic=synthetic, plan_options=plan_options
        )
    )
    paths = _write_reviewer_worksheets(out_path, sample.rows, annotators)
    strata_sizes: dict[str, int] = {}
    for key in sample.keys:
        strata_sizes[key] = strata_sizes.get(key, 0) + 1
    manifest = {
        "bundle": str(bundle),
        "kind": resolved_kind,
        "annotators": annotators,
        "worksheets": [str(path) for path in paths],
        "synthetic": synthetic,
        "seed": seed,
        "requested": n,
        "sample_size": sample.sample_size,
        "population": sample.population_size,
        "strata": strata_sizes,
        "acceptance_gate": sample.plan,
    }
    atomic_write_text(
        out_path.with_name(SAMPLE_MANIFEST), json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return paths
