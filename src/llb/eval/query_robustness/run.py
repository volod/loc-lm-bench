"""Production wiring for clean baseline plus noisy query probe lanes."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import new_run_timestamp
from llb.board.io import read_case_rows
from llb.core.config import RESTORATION_DEFAULTS, RunConfig, restoration_fields
from llb.eval import graph as eval_graph
from llb.eval.query_robustness.evaluate import (
    MITIGATION_LANES,
    LANE_TRANSLATE,
    MitigationLane,
    QueryExecutor,
    RobustnessResult,
    evaluate_query_robustness,
    mitigation_lanes_for_class,
)
from llb.eval.query_robustness.languages import (
    FixtureTranslationPrep,
    LANGUAGE_VARIANT_CLASSES,
    fixture_translation_queries,
    infer_language_fixture,
    language_fixture_status,
    load_language_variants,
    select_ukrainian_baseline,
)
from llb.eval.query_robustness.report import write_robustness_artifacts
from llb.eval.query_robustness.variants import resolve_variant_classes
from llb.executor.cases import score_case, spans_as_dicts
from llb.executor.runner import run_eval
from llb.executor.runner_backend import _make_launcher
from llb.executor.runner_retrieval import _load_store, build_query_prep
from llb.executor.runner_setup import _score_options, _select_eval_items
from llb.rag.query_prep.base import STEP_TYPOS
from llb.goldset.schema import GoldItem

METHOD = "query-robustness"


@dataclass(frozen=True)
class QueryRobustnessRun:
    result: RobustnessResult
    clean_run_dir: Path
    out_dir: Path
    paths: Mapping[str, str]


def make_query_executor(
    config: RunConfig,
    store: Any,
    launcher: Any,
    translation_queries: Mapping[str, str] | None = None,
    lanes: Sequence[MitigationLane] | None = None,
    restoration: Mapping[str, Any] | None = None,
) -> QueryExecutor:
    """Build one graph lane per mitigation configuration over one injected store/endpoint pair.

    `restoration` carries the run's restoration constants
    (restoration-constraint-threshold-sweep) into the lanes that keep the `typos` step; a lane
    without that step resets them, because the constants are inert -- and refused -- there.
    """
    options = _score_options(config)
    constants = dict(restoration or RESTORATION_DEFAULTS)

    def build(lane: MitigationLane) -> Any:
        query_prep: Any | None
        if lane == LANE_TRANSLATE:
            if translation_queries is None:
                raise ValueError("translation lane requires a paired language fixture")
            query_prep = FixtureTranslationPrep(translation_queries)
        else:
            lane_config = config.with_overrides(
                query_prep=list(lane.steps),
                query_prep_typo_guard=lane.typo_guard,
                query_prep_dense_case=lane.dense_case,
                **(constants if STEP_TYPOS in lane.steps else RESTORATION_DEFAULTS),
            )
            query_prep = build_query_prep(lane_config, store, launcher) if lane.steps else None
        return eval_graph.build_rag_graph(
            store,
            launcher,
            config.top_k,
            config.max_tokens,
            config.temperature,
            config.request_timeout_s,
            context_order=config.context_order,
            query_prep=query_prep,
            cited=config.cited_answers,
        )

    selected_lanes = lanes or (
        MITIGATION_LANES + ((LANE_TRANSLATE,) if translation_queries is not None else ())
    )
    apps = {lane.id: build(lane) for lane in selected_lanes}

    def execute(item: GoldItem, question: str, lane: MitigationLane) -> Mapping[str, Any]:
        state = eval_graph.run_case(apps[lane.id], question, spans_as_dicts(item))
        return score_case(item, state, options=options)

    return execute


def _baseline_config(config: RunConfig) -> RunConfig:
    values = config.model_dump()
    values.update(
        **RESTORATION_DEFAULTS,
        run_name="query-robustness-clean",
        query_prep=[],
        query_prep_typo_guard=False,
        query_prep_dense_case=False,
        insufficient_context_probes=0,
        judge_model=None,
        score_semantic=False,
        measure_telemetry=False,
    )
    return RunConfig.model_validate(values)


def run_query_robustness(
    config: RunConfig,
    *,
    split: str = "final",
    limit: int | None = None,
    typo_rate: float = 0.08,
    variant_classes: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
    emit_clean: bool = True,
    language_fixture: Path | None = None,
    dense_case: bool = False,
) -> QueryRobustnessRun:
    """Persist an ordinary clean run, then the isolated noisy probe bundle."""
    if not 0 < typo_rate <= 1:
        raise ValueError("typo_rate must be greater than 0 and at most 1")
    classes = resolve_variant_classes(variant_classes)
    baseline_config = _baseline_config(config)
    items = _select_eval_items(baseline_config, None, split, limit)
    if not items:
        raise SystemExit(f"no verified '{split}' items in {baseline_config.goldset_path}")
    language_classes = tuple(name for name in classes if name in LANGUAGE_VARIANT_CLASSES)
    excluded_language_baseline_ids: list[str] = []
    if language_classes:
        items, excluded_language_baseline_ids = select_ukrainian_baseline(items)
        if not items:
            raise SystemExit("no Ukrainian-dominant baseline questions for the query-language lane")
    fixture_path = language_fixture or infer_language_fixture(baseline_config.goldset_path)
    language_variants = (
        load_language_variants(fixture_path, items, language_classes) if language_classes else {}
    )
    translation_queries = fixture_translation_queries(language_variants, items) or None
    lanes = tuple(
        dict.fromkeys(
            lane
            for variant_class in classes
            for lane in mitigation_lanes_for_class(variant_class, dense_case=dense_case)
        )
    )
    clean = run_eval(
        baseline_config,
        items=items,
        split=split,
        emit=emit_clean,
    )
    clean_run_dir = Path(str(clean["paths"]["manifest"])).parent
    clean_rows = read_case_rows(Path(str(clean["paths"]["scores"])))

    store = _load_store(baseline_config)
    launcher = _make_launcher(baseline_config)
    with launcher as backend:
        execute = make_query_executor(
            baseline_config,
            store,
            backend,
            translation_queries,
            lanes=lanes,
            restoration=restoration_fields(config),
        )
        result = evaluate_query_robustness(
            items,
            clean_rows,
            execute,
            seed=baseline_config.seed,
            typo_rate=typo_rate,
            variant_classes=classes,
            progress=progress,
            language_variants=language_variants,
            dense_case=dense_case,
        )

    _, stamp = new_run_timestamp()
    out_dir = baseline_config.data_dir / METHOD / stamp
    metadata: dict[str, object] = {
        "model": baseline_config.model,
        "backend": baseline_config.backend,
        "split": split,
        "seed": baseline_config.seed,
        "typo_rate": typo_rate,
        "query_prep_dense_case": dense_case,
        "restoration_constants": restoration_fields(config),
        "variant_classes": list(classes),
        "clean_run_dir": clean_run_dir,
        "language_fixture": str(fixture_path) if language_classes else None,
        "language_fixture_status": (
            language_fixture_status(fixture_path) if language_classes else None
        ),
        "language_baseline_excluded_ids": excluded_language_baseline_ids,
    }
    paths = write_robustness_artifacts(result, out_dir, metadata)
    return QueryRobustnessRun(result, clean_run_dir, out_dir, paths)
