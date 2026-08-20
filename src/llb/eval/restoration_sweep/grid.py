"""Sweep the typo step's three restoration design constants over noisy Ukrainian queries.

The constraints in `llb.rag.query_prep.restore` ship with three hand-picked constants (the surface
budget, the short-token cutoff, and which ranking signal breaks a tie first). Each was chosen to be
conservative, and nothing measured what the conservatism costs. This module runs one RETRIEVAL-ONLY
pass per setting -- no generation, so a setting costs seconds rather than a model run -- over the
same seeded noise classes the query-robustness benchmark uses, and pairs each setting's recall and
MRR with the per-edit precision audit in `restoration_sweep_audit`.

Retrieval alone is the right scale here: the constants decide WHICH corpus surface a query token is
rewritten to, which moves the retrieved evidence. What a model does with that evidence is the
robustness benchmark's question, and it is measured there under the setting this sweep pins.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from itertools import product

from llb.core.config import RunConfig
from llb.eval.query_robustness.variants import KEYBOARD_TYPOS, TRANSLITERATION, generate_variant
from llb.eval.restoration_sweep.audit import CaseAlignment, EditRecord, audit_case
from llb.eval.restoration_sweep.lanes import (
    LaneAccumulator,
    LaneReading,
    RetrievalCache,
    SweepResult,
    score_prepared,
)
from llb.executor.cases import spans_as_dicts
from llb.goldset.schema import GoldItem
from llb.rag.query_prep.base import STEP_NORMALIZE, STEP_TYPOS
from llb.rag.query_prep.pipeline import QueryPrep
from llb.rag.query_prep.restore_policy import DEFAULT_RESTORATION_POLICY, RestorationPolicy

METHOD = "restoration-sweep"

# The classes whose noise the typo step is there to repair. Apostrophe and homoglyph noise is
# already fully inverted by the normalize step on this encoder, so a restoration constant cannot
# move them; running them would spend time to reprint the same saturated row per setting.
SWEEP_VARIANT_CLASSES = (TRANSLITERATION, KEYBOARD_TYPOS)

# Reference lanes every setting is read against: the clean question, the untouched noisy question,
# and safe normalization alone. None of the three consults the restoration constraints, so they are
# measured once and repeated in every setting's report block.
LANE_CLEAN = "clean"
LANE_OFF = "off"
LANE_NORMALIZE = "normalize"
REFERENCE_LANES = (LANE_CLEAN, LANE_OFF, LANE_NORMALIZE)

SWEEP_STEPS = (STEP_NORMALIZE, STEP_TYPOS)


def policy_grid(
    surface_values: Sequence[int],
    cutoff_values: Sequence[int],
    rank_values: Sequence[str],
    *,
    full: bool = False,
) -> tuple[RestorationPolicy, ...]:
    """The swept settings, always starting with the shipped default.

    One factor at a time by default: each alternative value is measured against the default with
    the other two constants held at theirs, which is what makes a per-constant verdict attributable.
    `full` measures the whole product instead, for reading interactions between two constants.
    """
    default = DEFAULT_RESTORATION_POLICY
    if full:
        candidates = [
            RestorationPolicy(surface, cutoff, rank)
            for surface, cutoff, rank in product(surface_values, cutoff_values, rank_values)
        ]
    else:
        candidates = [
            *(replace(default, surface_max_distance=value) for value in surface_values),
            *(replace(default, ambiguous_token_max_chars=value) for value in cutoff_values),
            *(replace(default, rank_order=value) for value in rank_values),
        ]
    ordered = [default, *(policy for policy in candidates if policy != default)]
    return tuple(dict.fromkeys(ordered))


def sweep_config(config: RunConfig, dense_case: bool = False) -> RunConfig:
    """The measured lane: safe normalization plus guarded corpus-vocabulary correction.

    `dense_case` routes the raw question's capitalization to the case-sensitive dense encoder
    (normalize-casefold-dense-lane-cost). It is set HERE rather than by the caller's config because
    it is refused at validation until the `normalize` step is present, which this lane supplies.
    """
    return config.with_overrides(
        query_prep=list(SWEEP_STEPS),
        query_prep_typo_guard=True,
        query_prep_dense_case=dense_case or None,
        run_name="restoration-sweep",
    )


def run_restoration_sweep(
    config: RunConfig,
    *,
    split: str = "final",
    limit: int | None = None,
    typo_rate: float = 0.08,
    variant_classes: Sequence[str] = SWEEP_VARIANT_CLASSES,
    policies: Sequence[RestorationPolicy] = (DEFAULT_RESTORATION_POLICY,),
    dense_case: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SweepResult:
    """Measure every setting's retrieval and edit precision over the same seeded noisy queries."""
    from llb.executor.runner_retrieval import _load_store, build_query_prep
    from llb.executor.runner_setup import _select_eval_items

    if not policies:
        raise ValueError("the sweep needs at least one restoration policy")
    if not variant_classes:
        raise ValueError("the sweep needs at least one noise class")
    lane_config = sweep_config(config, dense_case)
    items = _select_eval_items(lane_config, None, split, limit)
    if not items:
        raise ValueError(f"no verified '{split}' items in {lane_config.goldset_path}")
    store = _load_store(lane_config)
    prep = build_query_prep(lane_config, store, None)
    if prep is None:  # pragma: no cover - sweep_config always sets both steps
        raise ValueError("the restoration sweep needs the normalize,typos lane")
    cache = RetrievalCache(store, lane_config.top_k)
    raw_prep = QueryPrep.build(())
    normalize_prep = replace(prep, steps=(STEP_NORMALIZE,))
    references = _reference_readings(
        items, variant_classes, cache, raw_prep, normalize_prep, lane_config.seed, typo_rate
    )
    settings: list[LaneReading] = []
    edits: list[EditRecord] = []
    for index, policy in enumerate(policies, start=1):
        if progress is not None:
            progress(f"[restoration-sweep] setting {index}/{len(policies)}: {policy.label}")
        readings, records = _setting_readings(
            items,
            variant_classes,
            cache,
            replace(prep, restoration_policy=policy),
            normalize_prep,
            lane_config.seed,
            typo_rate,
        )
        settings.extend(readings)
        edits.extend(records)
    if progress is not None:
        progress(f"[restoration-sweep] {cache.calls} distinct queries retrieved")
    return SweepResult(
        policies=tuple(policies),
        variant_classes=tuple(variant_classes),
        settings=tuple(settings),
        references=tuple(references),
        edits=tuple(edits),
        item_ids=tuple(item.id for item in items),
        top_k=lane_config.top_k,
    )


def _variant(item: GoldItem, variant_class: str, seed: int, typo_rate: float) -> str:
    return generate_variant(
        item.question,
        variant_class,
        item_id=item.id,
        seed=seed,
        typo_rate=typo_rate,
    )


def _reference_readings(
    items: list[GoldItem],
    variant_classes: Sequence[str],
    cache: RetrievalCache,
    raw_prep: QueryPrep,
    normalize_prep: QueryPrep,
    seed: int,
    typo_rate: float,
) -> list[LaneReading]:
    """Clean, unmitigated, and normalize-only readings; none of them consults the constraints."""
    lanes = {lane: LaneAccumulator() for lane in REFERENCE_LANES}
    for item in items:
        spans = spans_as_dicts(item)
        for variant_class in variant_classes:
            noisy = _variant(item, variant_class, seed, typo_rate)
            lanes[LANE_CLEAN].add(
                variant_class, *score_prepared(cache, raw_prep.process(item.question), spans)
            )
            lanes[LANE_OFF].add(
                variant_class, *score_prepared(cache, raw_prep.process(noisy), spans)
            )
            lanes[LANE_NORMALIZE].add(
                variant_class, *score_prepared(cache, normalize_prep.process(noisy), spans)
            )
    return [reading for lane, acc in lanes.items() for reading in acc.readings(lane)]


def _setting_readings(
    items: list[GoldItem],
    variant_classes: Sequence[str],
    cache: RetrievalCache,
    prep: QueryPrep,
    normalize_prep: QueryPrep,
    seed: int,
    typo_rate: float,
) -> tuple[list[LaneReading], list[EditRecord]]:
    """One setting: retrieval per noise class plus the labeled corrections it made."""
    label = prep.restoration_policy.label
    accumulator = LaneAccumulator()
    records: list[EditRecord] = []
    for item in items:
        spans = spans_as_dicts(item)
        clean_normalized = normalize_prep.process(item.question).processed
        for variant_class in variant_classes:
            noisy = _variant(item, variant_class, seed, typo_rate)
            prepared = prep.process(noisy)
            case_records, counts = audit_case(
                setting=label,
                variant_class=variant_class,
                item_id=item.id,
                edits=prepared.edits,
                alignment=CaseAlignment.build(
                    clean_normalized, normalize_prep.process(noisy).processed
                ),
                vocabulary=prep.vocabulary,
            )
            records.extend(case_records)
            accumulator.add(variant_class, *score_prepared(cache, prepared, spans), counts)
    return accumulator.readings(label), records
