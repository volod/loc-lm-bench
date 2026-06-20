"""Minimal sequential eval runner -- the Milestone 1 walking skeleton.

Orchestrates one (model, config) end to end: load eval items -> retrieve+generate per
case through the LangGraph RAG flow -> score objective correctness + collect retrieval
hits -> aggregate one ranked row -> persist the canonical manifest+scores (then mirror).

Every heavy collaborator is injectable (`store`, `launcher`, `runner_fn`, `mirror`), so
the whole vertical runs end to end in a unit test with fakes -- no FAISS, langgraph,
Ollama, or GPU. The default path wires the real components and uses the compiled
LangGraph app.
"""

import uuid
from typing import Callable

from llb.config import RunConfig
from llb.eval import graph as eval_graph
from llb.goldset.schema import GoldItem, load_goldset
from llb.rag import retrieval
from llb.scoring import correctness
from llb.scoring.aggregate import ModelResult, format_table, rank_results
from llb.scoring.judge import judge_is_trusted
from llb.tracking.manifest import RunManifest, persist_run

RagState = eval_graph.RagState


def _load_eval_items(config: RunConfig, split: str, limit: int | None) -> list[GoldItem]:
    items = [it for it in load_goldset(config.goldset_path) if it.split == split]
    items.sort(key=lambda it: it.id)
    return items[:limit] if limit else items


def _spans_as_dicts(item: GoldItem) -> list[dict]:
    return [s.model_dump() for s in item.source_spans]


def score_case(item: GoldItem, state: RagState, embedder=None) -> dict:
    """One per-case score row from a terminal graph state."""
    answer = state.get("answer", "")
    status = state.get("status", eval_graph.OK)
    spans = _spans_as_dicts(item)
    retrieved = state.get("retrieved", [])
    corr = correctness.answer_correctness(answer, item.reference_answer, embedder=embedder)
    usage = state.get("usage", {})
    row = {
        "item_id": item.id,
        "split": item.split,
        "status": status,
        "objective_score": corr["score"],
        "token_f1": corr["token_f1"],
        "exact": corr["exact"],
        "contains": corr["contains"],
        "retrieval_hit": retrieval.recall_at_k(retrieved, spans, len(retrieved)),
        "first_hit_rank": retrieval.first_hit_rank(retrieved, spans),
        "tokens_per_s": usage.get("tokens_per_s", 0.0),
        "latency_s": usage.get("latency_s", 0.0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "answer_preview": (answer or "")[:280],
    }
    if "semantic" in corr:
        row["semantic"] = corr["semantic"]
    return row


def _make_launcher(config: RunConfig):
    if config.backend == "ollama":
        from llb.backends.ollama import OllamaLauncher

        return OllamaLauncher(config.model, host=config.ollama_host)
    raise SystemExit(
        f"backend '{config.backend}' is not wired in Milestone 1 (Ollama only)."
    )


def _default_runner_fn(config: RunConfig, store, launcher) -> Callable[[GoldItem], RagState]:
    app = eval_graph.build_rag_graph(
        store, launcher, config.top_k, config.max_tokens, config.temperature,
        config.request_timeout_s,
    )

    def run(item: GoldItem) -> RagState:
        return eval_graph.run_case(app, item.question, _spans_as_dicts(item))

    return run


def _aggregate(config: RunConfig, case_rows: list[dict], judge_rho: float | None,
               telemetry: dict) -> tuple[list[dict], dict]:
    n = len(case_rows)
    objective = sum(r["objective_score"] for r in case_rows) / n if n else 0.0
    ok = [r for r in case_rows if r["status"] == eval_graph.OK]
    reliability = len(ok) / n if n else 0.0
    tok_rates = [r["tokens_per_s"] for r in ok if r["tokens_per_s"] > 0]
    tokens_per_s = sum(tok_rates) / len(tok_rates) if tok_rates else 0.0
    result = ModelResult(
        model=config.model,
        backend=config.backend,
        objective_score=objective,
        n_cases=n,
        reliability=reliability,
        tokens_per_s=tokens_per_s,
        peak_vram_mb=telemetry.get("peak_vram_mb"),
        judge_score=None,
        feasible=True,
    )
    trusted = judge_is_trusted(judge_rho, config.judge_threshold)
    rows = rank_results([result], judge_trusted=trusted)
    metrics = {
        "objective_score": objective,
        "reliability": reliability,
        "tokens_per_s": tokens_per_s,
    }
    return rows, metrics


def run_eval(
    config: RunConfig,
    *,
    items: list[GoldItem] | None = None,
    store=None,
    launcher=None,
    runner_fn: Callable[[GoldItem], RagState] | None = None,
    mirror: Callable | None = None,
    judge_rho: float | None = None,
    limit: int | None = None,
    split: str = "final",
    worksheet=None,
    emit: bool = True,
) -> dict:
    """Run the skeleton and return {rows, metrics, paths, table}.

    `worksheet` (a path) emits a judge-calibration worksheet pre-filled with this run's
    model answers (the human only adds ratings); pair it with `split="calibration"`.
    """
    if items is None:
        items = _load_eval_items(config, split, limit)
    if not items:
        raise SystemExit(f"no '{split}' items in {config.goldset_path}")

    if launcher is None:
        launcher = _make_launcher(config)
    if runner_fn is None:
        if store is None:
            from llb.rag.store import RagStore

            store = RagStore.load(config.index_dir())
        runner_fn = _default_runner_fn(config, store, launcher)

    embedder = store.embedder if (config.score_semantic and hasattr(store, "embedder")) else None

    case_rows: list[dict] = []
    retrieval_pairs: list[tuple[list[dict], list[dict]]] = []
    answers: list[tuple[GoldItem, str]] = []
    with launcher:
        for item in items:
            state = runner_fn(item)
            case_rows.append(score_case(item, state, embedder=embedder))
            retrieval_pairs.append((state.get("retrieved", []), _spans_as_dicts(item)))
            answers.append((item, state.get("answer", "")))

    telemetry = launcher.telemetry() if hasattr(launcher, "telemetry") else {}
    rows, metrics = _aggregate(config, case_rows, judge_rho, telemetry)
    retrieval_metrics = retrieval.evaluate_retrieval(retrieval_pairs, config.top_k)

    manifest = RunManifest(
        run_id=uuid.uuid4().hex[:12],
        run_name=config.run_name,
        config=config.fingerprint(),
        metrics=metrics,
        retrieval=retrieval_metrics,
        judge={
            "calibration_rho": judge_rho,
            "threshold": config.judge_threshold,
            "trusted": judge_is_trusted(judge_rho, config.judge_threshold),
        },
        n_cases=len(case_rows),
    )
    paths = persist_run(manifest, case_rows, config.run_dir(), mirror=mirror)

    if worksheet is not None:
        from llb.judge.calibration import write_filled_worksheet

        n = write_filled_worksheet(answers, worksheet)
        paths["worksheet"] = str(worksheet)

    table = format_table(rows)
    if emit:
        print(f"[run-eval] model={config.model} backend={config.backend} "
              f"cases={len(case_rows)}")
        print(f"[run-eval] retrieval: recall@{config.top_k}="
              f"{retrieval_metrics['recall_at_k']:.3f} mrr={retrieval_metrics['mrr']:.3f}")
        print(table)
        print(f"[run-eval] manifest -> {paths['manifest']}")
        print(f"[run-eval] scores   -> {paths['scores']} (mirror: {paths['mirror']})")
        if worksheet is not None:
            print(f"[run-eval] worksheet -> {worksheet} ({n} rows; add human_rating)")
    return {"rows": rows, "metrics": metrics, "retrieval": retrieval_metrics,
            "paths": paths, "table": table, "manifest": manifest}
