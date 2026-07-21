"""Production wiring for clean baseline plus noisy query probe lanes."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llb.bench.common import new_run_timestamp
from llb.core.config import RunConfig
from llb.eval import graph as eval_graph
from llb.eval.query_robustness import (
    MITIGATION_STEPS,
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
    """Build raw and normalize+typos graph lanes over one injected store/endpoint pair."""
    mitigation_config = config.with_overrides(
        query_prep=list(MITIGATION_STEPS), query_prep_typo_guard=True
    )
    prep = build_query_prep(mitigation_config, store, launcher)
    options = _score_options(config)

    def build(prepared: bool) -> Any:
        return eval_graph.build_rag_graph(
            store,
            launcher,
            config.top_k,
            config.max_tokens,
            config.temperature,
            config.request_timeout_s,
            context_order=config.context_order,
            query_prep=prep if prepared else None,
            cited=config.cited_answers,
        )

    apps = {False: build(False), True: build(True)}

    def execute(item: GoldItem, question: str, mitigated: bool) -> Mapping[str, Any]:
        state = eval_graph.run_case(apps[mitigated], question, spans_as_dicts(item))
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


def load_clean_case_rows(path: Path) -> list[dict[str, Any]]:
    """Load canonical per-case rows; `run_eval()['rows']` contains aggregate board rows."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or "item_id" not in value:
                raise ValueError(f"{path}:{line_number}: expected a per-case score row")
            rows.append(value)
    return rows


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
    clean_rows = load_clean_case_rows(Path(str(clean["paths"]["scores"])))

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
