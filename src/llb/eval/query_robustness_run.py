"""Production wiring for clean baseline plus noisy query probe lanes."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import new_run_timestamp
from llb.board.io import read_case_rows
from llb.core.config import RunConfig
from llb.eval import graph as eval_graph
from llb.eval.query_robustness import (
    MITIGATION_LANES,
    MitigationLane,
    QueryExecutor,
    RobustnessResult,
    evaluate_query_robustness,
)
from llb.eval.query_robustness_report import write_robustness_artifacts
from llb.executor.cases import score_case, spans_as_dicts
from llb.executor.runner import run_eval
from llb.executor.runner_backend import _make_launcher
from llb.executor.runner_retrieval import _load_store, build_query_prep
from llb.executor.runner_setup import _score_options, _select_eval_items
from llb.goldset.schema import GoldItem

METHOD = "query-robustness"


@dataclass(frozen=True)
class QueryRobustnessRun:
    result: RobustnessResult
    clean_run_dir: Path
    out_dir: Path
    paths: Mapping[str, str]


def make_query_executor(config: RunConfig, store: Any, launcher: Any) -> QueryExecutor:
    """Build one graph lane per mitigation configuration over one injected store/endpoint pair."""
    options = _score_options(config)

    def build(lane: MitigationLane) -> Any:
        lane_config = config.with_overrides(
            query_prep=list(lane.steps), query_prep_typo_guard=lane.typo_guard
        )
        return eval_graph.build_rag_graph(
            store,
            launcher,
            config.top_k,
            config.max_tokens,
            config.temperature,
            config.request_timeout_s,
            context_order=config.context_order,
            query_prep=build_query_prep(lane_config, store, launcher) if lane.steps else None,
            cited=config.cited_answers,
        )

    apps = {lane.id: build(lane) for lane in MITIGATION_LANES}

    def execute(item: GoldItem, question: str, lane: MitigationLane) -> Mapping[str, Any]:
        state = eval_graph.run_case(apps[lane.id], question, spans_as_dicts(item))
        return score_case(item, state, options=options)

    return execute


def _baseline_config(config: RunConfig) -> RunConfig:
    values = config.model_dump()
    values.update(
        run_name="query-robustness-clean",
        query_prep=[],
        query_prep_typo_guard=False,
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
    progress: Callable[[str], None] | None = None,
    emit_clean: bool = True,
) -> QueryRobustnessRun:
    """Persist an ordinary clean run, then the isolated noisy probe bundle."""
    if not 0 < typo_rate <= 1:
        raise ValueError("typo_rate must be greater than 0 and at most 1")
    baseline_config = _baseline_config(config)
    items = _select_eval_items(baseline_config, None, split, limit)
    if not items:
        raise SystemExit(f"no verified '{split}' items in {baseline_config.goldset_path}")
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
        execute = make_query_executor(baseline_config, store, backend)
        result = evaluate_query_robustness(
            items,
            clean_rows,
            execute,
            seed=baseline_config.seed,
            typo_rate=typo_rate,
            progress=progress,
        )

    _, stamp = new_run_timestamp()
    out_dir = baseline_config.data_dir / METHOD / stamp
    metadata: dict[str, object] = {
        "model": baseline_config.model,
        "backend": baseline_config.backend,
        "split": split,
        "seed": baseline_config.seed,
        "typo_rate": typo_rate,
        "clean_run_dir": clean_run_dir,
    }
    paths = write_robustness_artifacts(result, out_dir, metadata)
    return QueryRobustnessRun(result, clean_run_dir, out_dir, paths)
