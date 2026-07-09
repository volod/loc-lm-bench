"""Minimal sequential eval runner -- the RAG core walking skeleton.

Orchestrates one (model, config) end to end: load eval items -> retrieve+generate per
case through the LangGraph RAG flow -> score objective correctness + collect retrieval
hits -> aggregate one ranked row -> persist the canonical manifest+scores (then mirror).

Every heavy collaborator is injectable (`store`, `launcher`, `runner_fn`, `mirror`), so
the whole vertical runs end to end in a unit test with fakes -- no FAISS, langgraph,
Ollama, or GPU. The default path wires the real components and uses the compiled
LangGraph app.
"""

import logging
import shutil
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.core.contracts import (
    BackendMetadata,
    CaseScoreRow,
    ContentionReport,
    EvalResult,
    JudgeInputRecord,
    JudgeScore,
    JudgeStatus,
    LeaderboardRow,
    RunMetrics,
    TelemetryReport,
)
from llb.eval import common as eval_common
from llb.eval import graph as eval_graph
from llb.executor import durability
from llb.executor.cases import CaseBatch, batch_retrieval_records, spans_as_dicts
from llb.executor.reporting import emit_summary
from llb.goldset.schema import GoldItem, load_goldset
from llb.rag import retrieval
from llb.scoring.aggregate import ModelResult, format_table, rank_results
from llb.scoring.judge import judge_is_trusted, run_judge
from llb.tracking.manifest import RunManifest, persist_run

if TYPE_CHECKING:
    from llb.eval.insufficient_context import InsufficientContextReport
    from llb.executor.cases import ScoreOptions

JudgeScorer = Callable[[list[JudgeInputRecord], str], list[JudgeScore]]

RagState = eval_graph.RagState
_RUN_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"
_LOG = logging.getLogger(__name__)


def _preserve_backend_log(launcher: BackendLauncher, config: RunConfig) -> None:
    """Copy a failed backend's startup log out of the staging dir (which is about to be
    removed) into the persistent logs dir, so a launch failure stays diagnosable instead of
    vanishing with the staging bundle (e.g. a vLLM engine that dies during startup)."""
    log_path = getattr(launcher, "log_path", None)
    src = Path(log_path) if log_path else None
    if src is None or not src.exists():
        return
    dest_dir = config.data_dir / "llb" / "logs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"failed-{src.stem}-{stamp}.log"
    try:
        shutil.copyfile(src, dest)
    except OSError:
        return
    _LOG.error("[run-eval] backend failed to start; startup log preserved -> %s", dest)


def _load_eval_items(config: RunConfig, split: str, limit: int | None) -> list[GoldItem]:
    if not config.goldset_path.exists():
        raise SystemExit(
            f"gold set not found: {config.goldset_path}\n"
            "  use the committed fixture with --goldset "
            "samples/goldsets/ua_squad_postedited_v1/goldset.jsonl,\n"
            "  or create unverified development material with `make ingest-uk-squad`."
        )
    items = [
        item for item in load_goldset(config.goldset_path) if item.split == split and item.verified
    ]
    items.sort(key=lambda it: it.id)
    return items[:limit] if limit is not None else items


def _select_eval_items(
    config: RunConfig,
    items: list[GoldItem] | None,
    split: str,
    limit: int | None,
) -> list[GoldItem]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if items is None:
        return _load_eval_items(config, split, limit)
    selected = sorted(
        (item for item in items if item.split == split and item.verified),
        key=lambda item: item.id,
    )
    return selected[:limit] if limit is not None else selected


def _make_launcher(config: RunConfig, log_dir: Path | None = None) -> BackendLauncher:
    if config.adapter_path is not None and config.backend != "vllm":
        raise SystemExit(
            f"[run-eval] adapter serving is wired for vLLM LoRA modules; backend "
            f"{config.backend!r} needs a merged model artifact first"
        )
    if config.backend == "ollama":
        from llb.backends.ollama import OllamaLauncher

        return OllamaLauncher(config.model, host=config.ollama_host)
    if config.backend == "vllm":
        from llb.backends.vllm import VllmLauncher

        return VllmLauncher(
            config.model,
            host=config.vllm_host,
            port=config.vllm_port,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            cpu_offload_gb=config.cpu_offload_gb,
            kv_offloading_size_gb=config.kv_offloading_size_gb,
            dtype=config.dtype,
            quantization=config.quantization,
            adapter_path=config.adapter_path,
            log_dir=log_dir,
        )
    if config.backend == "llamacpp":
        from llb.backends.llamacpp import LlamaCppLauncher, resolve_llama_server_binary

        return LlamaCppLauncher(
            config.model,
            host=config.llamacpp_host,
            n_gpu_layers=config.n_gpu_layers,
            ctx_size=config.max_model_len,
            log_dir=log_dir,
            binary=resolve_llama_server_binary(config.data_dir),
        )
    raise SystemExit(f"backend '{config.backend}' is not wired (ollama, vllm, llamacpp supported).")


def _vram_reader() -> Callable[[], int] | None:
    """Best-effort NVML reader for telemetry (None when the [telemetry] extra/GPU is absent)."""
    try:
        from llb.executor.vram import nvml_reader

        return nvml_reader()
    except (Exception, SystemExit):  # nvml_reader raises SystemExit when [telemetry] is absent
        return None


def _pid_usage_reader() -> Callable[[], dict[int, int]] | None:
    """Best-effort NVML per-PID VRAM reader (for the VRAM contention guard contention guard's resident attribution)."""
    try:
        from llb.executor.vram import nvml_process_reader

        return nvml_process_reader()
    except (Exception, SystemExit):
        return None


def _guard_vllm_contention(
    config: RunConfig, launcher: BackendLauncher, *, evict: bool, wait: bool
) -> "ContentionReport | None":
    """Pre-launch VRAM-contention guard for vLLM (VRAM contention guard): derate gpu-memory-utilization to the
    actually-free VRAM, or abort if even that cannot hold the model. No-op without a GPU."""
    from llb.backends.vllm import VllmLauncher
    from llb.executor.contention import (
        ACTION_ABORT,
        apply_contention_guard,
        default_gpu_reader,
        model_kv_headroom_mb,
        model_weight_floor_mb,
    )

    report = apply_contention_guard(
        requested_util=config.gpu_memory_utilization,
        weight_floor_mb=model_weight_floor_mb(config.model),
        gpu_reader=default_gpu_reader,
        process_reader=_pid_usage_reader(),
        evict=evict,
        wait=wait,
        ollama_host=config.ollama_host,
        min_kv_headroom_mb=model_kv_headroom_mb(config.model),
    )
    if report is None:
        return None
    if report["action"] == ACTION_ABORT:
        raise SystemExit(f"[run-eval] pre-launch VRAM guard: {report['note']}")
    if report["derated"] and isinstance(launcher, VllmLauncher):
        _LOG.warning("[run-eval] %s", report["note"])
        launcher.gpu_memory_utilization = report["safe_util"]
        launcher.meta["gpu_memory_utilization"] = report["safe_util"]
    else:
        _LOG.info("[run-eval] pre-launch VRAM guard: %s", report["note"])
    return report


def _default_runner_fn(
    config: RunConfig, store: Any, launcher: BackendLauncher, prompt_package: Any | None = None
) -> Callable[[GoldItem], RagState]:
    chunk_filter = None
    if config.acl_label is not None:
        from llb.rag.filters import metadata_filter

        chunk_filter = metadata_filter(acl_label=config.acl_label)
    app = eval_graph.build_rag_graph(
        store,
        launcher,
        config.top_k,
        config.max_tokens,
        config.temperature,
        config.request_timeout_s,
        prompt_package=prompt_package,
        context_order=config.context_order,
        query_prep=build_query_prep(config, store, launcher),
        chunk_filter=chunk_filter,
        cited=config.cited_answers,
    )

    def run(item: GoldItem) -> RagState:
        return eval_graph.run_case(app, item.question, spans_as_dicts(item))

    return run


def _score_options(config: RunConfig) -> "ScoreOptions":
    """The opt-in answer-side scoring toggles for this run (groundedness-citation-metrics)."""
    from llb.executor.cases import ScoreOptions

    return ScoreOptions(
        score_groundedness=config.score_groundedness,
        cited_answers=config.cited_answers,
        context_order=config.context_order,
    )


def _maybe_run_probes(
    config: RunConfig, items: list[GoldItem], store: Any, backend: Any
) -> "InsufficientContextReport | None":
    """Run the insufficient-context abstention probe over a seeded sample, if configured.

    The gold evidence is excluded from retrieval for each probed item; correct behavior is an
    explicit abstention. Probe rows are scored separately and never enter the correctness batch."""
    if config.insufficient_context_probes <= 0:
        return None
    from llb.eval.insufficient_context import run_insufficient_context_probe

    def chat(messages: Any) -> tuple[str, str | None]:
        result = backend.chat(
            messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.request_timeout_s,
        )
        return result.text or "", result.error

    return run_insufficient_context_probe(
        items,
        store,
        chat,
        model=config.model,
        backend=config.backend,
        k=config.top_k,
        n=config.insufficient_context_probes,
        seed=config.seed,
        cited=config.cited_answers,
        context_order=config.context_order,
    )


def _launcher_rewriter(config: RunConfig, launcher: Any) -> Callable[[str], str]:
    """Local-LLM query rewriter over the run's backend endpoint seam (uk-query-processing)."""
    from llb.prompts import render_chat

    def rewrite(query: str) -> str:
        messages = render_chat("eval.rag.query_rewrite", {"query": query})
        result = launcher.chat(
            messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.request_timeout_s,
        )
        return result.text or ""

    return rewrite


def build_query_prep(config: RunConfig, store: Any, launcher: Any | None) -> Any | None:
    """Build the opt-in query-side lane for this run, resolving each step's dependency.

    Returns None when no steps are configured (the lane is an exact no-op). The typo step reads
    the corpus vocabulary from the loaded store's chunks; the glossary step loads
    `config.query_glossary_path`; the rewrite step wraps the backend launcher. Missing
    dependencies raise a clear SystemExit rather than a bare error mid-run."""
    from llb.rag import query_prep as qp

    steps = list(config.query_prep)
    if not steps:
        return None
    vocabulary = None
    glossary = None
    rewriter = None
    if qp.STEP_TYPOS in steps:
        chunks = getattr(store, "chunks", None) or []
        vocabulary = qp.build_vocabulary(str(chunk.get("text", "")) for chunk in chunks)
    if qp.STEP_GLOSSARY in steps:
        if config.query_glossary_path is None:
            raise SystemExit(
                "[run-eval] query_prep 'glossary' step needs query_glossary_path "
                "(build one with `llb build-query-glossary`)."
            )
        if not Path(config.query_glossary_path).is_file():
            raise SystemExit(f"[run-eval] query glossary not found: {config.query_glossary_path}")
        glossary = qp.Glossary.load(config.query_glossary_path)
    if qp.STEP_REWRITE in steps:
        if launcher is None:
            raise SystemExit("[run-eval] query_prep 'rewrite' step needs a backend launcher")
        rewriter = _launcher_rewriter(config, launcher)
    try:
        return qp.QueryPrep.build(
            steps, vocabulary=vocabulary, glossary=glossary, rewriter=rewriter
        )
    except ValueError as exc:
        raise SystemExit(f"[run-eval] invalid query_prep: {exc}") from None


def _judge_records(batch: CaseBatch) -> list[JudgeInputRecord]:
    """The (question, answer, retrieved-contexts) record per case the judge scores."""
    return [
        {
            "question": item.question,
            "answer": answer,
            "contexts": [str(chunk.get("text", "")) for chunk in retrieved],
        }
        for (item, answer), (retrieved, _spans) in zip(batch.answers, batch.retrieval_pairs)
    ]


def _judge_value(score: JudgeScore) -> float:
    """One scalar judge rating per case: the mean of faithfulness + answer-relevancy."""
    return (score["faithfulness"] + score["answer_relevancy"]) / 2.0


def _configured_judge_scorer(config: RunConfig, scorer: JudgeScorer | None) -> JudgeScorer:
    """Bind the configured endpoint while preserving the injectable scorer seam."""
    if scorer is not None:
        return scorer
    from llb.scoring.judge import deepeval_scorer

    def score(records: list[JudgeInputRecord], model: str) -> list[JudgeScore]:
        return deepeval_scorer(records, model, base_url=config.judge_base_url)

    return score


def _judge_cases(
    config: RunConfig,
    batch: CaseBatch,
    judge_rho: float | None,
    scorer: JudgeScorer | None,
) -> float | None:
    """Score answers with the GATED judge (Premise 2) and attach per-case judge scores.

    Returns the mean per-case judge score ONLY when the judge is configured AND trusted
    (calibration rho >= threshold); otherwise the judge stays a demoted diagnostic and objective
    correctness ranks alone. The per-case judge value is the mean of faithfulness + relevancy.
    """
    if config.judge_model is None:
        return None
    outcome = run_judge(
        _judge_records(batch),
        config.judge_model,
        judge_rho,
        config.judge_threshold,
        scorer=_configured_judge_scorer(config, scorer),
    )
    if not outcome.trusted or not outcome.scores:
        _LOG.info("[run-eval] judge demoted (%s); objective ranks alone", outcome.reason)
        return None
    per_case = [_judge_value(s) for s in outcome.scores]
    for row, value in zip(batch.rows, per_case):
        row["judge_score"] = round(value, 4)
    return sum(per_case) / len(per_case) if per_case else None


def _judge_ratings(
    config: RunConfig, batch: CaseBatch, scorer: JudgeScorer | None
) -> list[float] | None:
    """Run the judge UNGATED and return one rating per case (judge calibration gate calibration scaffolding).

    Calibration measures whether the judge AGREES with humans, so the judge runs regardless of
    its (not-yet-known) trust -- the gate is irrelevant here. Returns None when no judge is
    configured; raises if the judge backend is unavailable (so the worksheet path can warn).
    """
    if config.judge_model is None:
        return None
    score_fn = _configured_judge_scorer(config, scorer)
    scores = score_fn(_judge_records(batch), config.judge_model)
    return [_judge_value(s) for s in scores]


def _aggregate(
    config: RunConfig,
    case_rows: list[CaseScoreRow],
    judge_rho: float | None,
    telemetry: Mapping[str, object],
    judge_score: float | None = None,
) -> tuple[list[LeaderboardRow], RunMetrics]:
    n = len(case_rows)
    objective = sum(r["objective_score"] for r in case_rows) / n if n else 0.0
    ok = [r for r in case_rows if r["status"] == eval_common.OK]
    reliability = len(ok) / n if n else 0.0
    tok_rates = [r["tokens_per_s"] for r in ok if r["tokens_per_s"] > 0]
    observed_tokens_per_s = sum(tok_rates) / len(tok_rates) if tok_rates else 0.0
    steady_rate = telemetry.get("steady_tokens_per_s")
    tokens_per_s = (
        float(steady_rate)
        if isinstance(steady_rate, int | float) and steady_rate > 0
        else observed_tokens_per_s
    )
    peak_vram = telemetry.get("peak_vram_mb")
    result = ModelResult(
        model=config.model,
        backend=config.backend,
        objective_score=objective,
        n_cases=n,
        reliability=reliability,
        tokens_per_s=tokens_per_s,
        peak_vram_mb=float(peak_vram) if isinstance(peak_vram, int | float) else None,
        judge_score=judge_score,
        feasible=True,
    )
    # The judge is trusted only when calibrated AND it actually produced a score this run.
    trusted = judge_is_trusted(judge_rho, config.judge_threshold) and judge_score is not None
    rows = rank_results([result], judge_trusted=trusted)
    metrics: RunMetrics = {
        "objective_score": objective,
        "reliability": reliability,
        "tokens_per_s": tokens_per_s,
    }
    mean_power = telemetry.get("mean_power_w")
    if isinstance(mean_power, int | float) and mean_power > 0:
        metrics["mean_power_w"] = round(float(mean_power), 2)
        metrics["tokens_per_watt"] = round(tokens_per_s / float(mean_power), 4)
        metrics["quality_per_watt"] = round(objective * tokens_per_s / float(mean_power), 4)
    stage = _stage_latency(case_rows)
    if stage:
        metrics["stage_latency"] = stage
    if judge_score is not None:
        metrics["judge_score"] = round(judge_score, 4)
    _attach_answer_side_metrics(metrics, case_rows)
    return rows, metrics


def _attach_answer_side_metrics(metrics: RunMetrics, case_rows: list[CaseScoreRow]) -> None:
    """Mean per-case groundedness / citation signals (groundedness-citation-metrics), when present."""
    for key in ("groundedness", "citation_validity", "hallucinated_citation_rate"):
        values = [float(row[key]) for row in case_rows if key in row]
        if values:
            metrics[key] = round(sum(values) / len(values), 4)


def _stage_latency(case_rows: list[CaseScoreRow]) -> dict[str, float]:
    """Mean per-case stage wall-clock (rerank-context-order): retrieve / rerank / generate.

    Retrieve and rerank means cover the cases that recorded them (rerank only exists when a
    reranker is configured); generate is the mean backend latency. Empty when nothing was
    measured, so pre-existing bundles keep their shape."""

    def mean_of(key: str) -> float | None:
        values = [float(row[key]) for row in case_rows if key in row]  # type: ignore[literal-required]
        return round(sum(values) / len(values), 4) if values else None

    stage: dict[str, float] = {}
    retrieve_s = mean_of("retrieve_latency_s")
    if retrieve_s is not None:
        stage["retrieve_s"] = retrieve_s
    rerank_s = mean_of("rerank_latency_s")
    if rerank_s is not None:
        stage["rerank_s"] = rerank_s
    if stage:
        generate = [float(row["latency_s"]) for row in case_rows if row.get("latency_s")]
        stage["generate_s"] = round(sum(generate) / len(generate), 4) if generate else 0.0
    return stage


def _run_timestamp(run_id: str) -> str:
    now = datetime.now(timezone.utc).strftime(_RUN_TIMESTAMP_FORMAT)
    return f"{now}-{run_id}"


def _collect_optional_telemetry(
    config: RunConfig, launcher: BackendLauncher
) -> TelemetryReport | None:
    if not config.measure_telemetry:
        return None
    from llb.backends.telemetry import collect_telemetry, nvidia_smi_power_reader

    return collect_telemetry(
        launcher,
        requested_context=config.max_model_len,
        timeout=config.request_timeout_s,
        vram_reader=_vram_reader(),
        power_reader=nvidia_smi_power_reader(),
    )


def _build_judge_metadata(config: RunConfig, judge_rho: float | None) -> JudgeStatus:
    judge_metadata: JudgeStatus = {
        "calibration_rho": judge_rho,
        "threshold": config.judge_threshold,
        "trusted": judge_is_trusted(judge_rho, config.judge_threshold),
    }
    if config.judge_model is None:
        return judge_metadata
    from llb.scoring.judge import judge_experiment_metadata

    experiment_metadata = judge_experiment_metadata(config.judge_model, config.judge_base_url)
    judge_metadata["provider"] = experiment_metadata["provider"]
    judge_metadata["model"] = experiment_metadata["model"]
    judge_metadata["base_url"] = experiment_metadata["base_url"]
    judge_metadata["prompt_language"] = experiment_metadata["prompt_language"]
    judge_metadata["metrics"] = experiment_metadata["metrics"]
    return judge_metadata


def _resolve_eval_runner(
    config: RunConfig,
    *,
    store: Any,
    launcher: BackendLauncher | None,
    runner_fn: Callable[[GoldItem], RagState] | None,
    prompt_package: Any | None,
    staging_dir: Path,
    evict: bool,
    wait: bool,
) -> tuple[BackendLauncher, Callable[[GoldItem], RagState], Any, ContentionReport | None]:
    contention: ContentionReport | None = None
    if launcher is None:
        launcher = _make_launcher(config, log_dir=staging_dir / "vllm")
        if config.backend == "vllm":
            contention = _guard_vllm_contention(config, launcher, evict=evict, wait=wait)
    if runner_fn is None:
        if store is None:
            store = _load_store(config)
        runner_fn = _default_runner_fn(config, store, launcher, prompt_package)
    return launcher, runner_fn, store, contention


def _load_store(config: RunConfig) -> Any:
    """Load the configured retrieval store: the GraphRAG backend (GraphRAG backend) or the default FAISS store.

    Both expose the same `.retrieve(question, k) -> list[ChunkRecord]` seam, so the eval graph,
    scoring, isolation, and board are unchanged regardless of backend. With `config.reranker`
    set, the loaded store is wrapped in the cross-encoder rerank stage (rerank-context-order);
    the wrapper honors the same retrieve seam, so every backend gains reranking identically."""
    from llb.rag.rerank import maybe_wrap_reranker

    if config.retrieval_backend == "graph":
        from llb.graph.store import GraphStore

        return maybe_wrap_reranker(
            GraphStore.load(
                config.graph_dir(),
                strategy=config.retrieval_strategy,
                khop_depth=config.graph_khop_depth,
            ),
            config,
        )
    from llb.rag.store import MODE_HYBRID, RagStore, stale_store_message, store_embedder_mismatch

    store = RagStore.load(config.index_dir())
    stale = stale_store_message(store.meta, config.corpus_root, config.index_dir())
    if stale is not None:
        raise SystemExit(stale)
    built = store_embedder_mismatch(store.meta, config.embedding_model)
    if built is not None:
        raise SystemExit(
            f"[run-eval] embedder mismatch: the store at {config.index_dir()} was built with "
            f"'{built}' but config.embedding_model is '{config.embedding_model}'. Rebuild the "
            f"index (build-index --embedding-model {config.embedding_model}) or set the config to "
            f"match; a store is embedded and queried by one encoder, so they must agree."
        )
    if config.retrieval_mode == MODE_HYBRID:
        if getattr(store, "lexical", None) is None:
            raise SystemExit(
                f"[run-eval] --retrieval-mode hybrid needs a lexical index, but the store at "
                f"{config.index_dir()} was built '{store.meta.get('mode')}' (dense-only). "
                f"Rebuild it with `build-index --retrieval-mode hybrid`."
            )
        store.fusion_weight = config.fusion_weight
        store.fusion_candidates = config.fusion_candidates
    elif getattr(store, "lexical", None) is not None:
        # A hybrid store can always serve dense-only: drop the lexical side for this run.
        _LOG.info(
            "[run-eval] retrieval_mode=%s over a hybrid store; lexical fusion disabled",
            config.retrieval_mode,
        )
        store.lexical = None
    return maybe_wrap_reranker(store, config)


def _write_calibration_worksheet(
    config: RunConfig,
    batch: CaseBatch,
    worksheet: Path | str,
    judge_scorer: JudgeScorer | None,
) -> int:
    from llb.judge.calibration import write_filled_worksheet

    judge_ratings: list[float] | None = None
    if config.judge_model is not None:
        try:
            judge_ratings = _judge_ratings(config, batch, judge_scorer)
        except (Exception, SystemExit) as exc:
            _LOG.warning(
                "[run-eval] judge unavailable for the worksheet (%s); judge_rating left blank "
                "-- pick the judge (OQ2) and install its backend to calibrate.",
                exc,
            )
    return write_filled_worksheet(batch.answers, Path(worksheet), judge_ratings=judge_ratings)


def run_eval(
    config: RunConfig,
    *,
    items: list[GoldItem] | None = None,
    store: Any = None,
    launcher: BackendLauncher | None = None,
    runner_fn: Callable[[GoldItem], RagState] | None = None,
    prompt_package: Any | None = None,
    prompt_system_provenance: Mapping[str, object] | None = None,
    mirror: Callable[[RunManifest, Path], None] | None = None,
    judge_rho: float | None = None,
    judge_scorer: JudgeScorer | None = None,
    limit: int | None = None,
    split: str = "final",
    worksheet: Path | str | None = None,
    evict: bool = False,
    wait: bool = False,
    emit: bool = True,
    resume: Path | str | None = None,
    max_case_retries: int = 2,
    retry_backoff_s: float = 1.0,
    max_backend_relaunches: int = 1,
    sleep: Callable[[float], None] | None = None,
) -> EvalResult:
    """Run the skeleton and return {rows, metrics, paths, table}.

    `worksheet` (a path) emits a judge-calibration worksheet pre-filled with this run's
    model answers (the human only adds ratings); pair it with `split="calibration"`.

    The run is durable (the durable-eval-runner): completed cases journal to
    `cases.progress.jsonl` in the staging dir, transient per-case transport failures retry
    (`max_case_retries` / `retry_backoff_s`), and a crashed launcher-owned backend relaunches up to
    `max_backend_relaunches` times. `resume=<run-dir>` continues an interrupted run from its journal
    instead of re-spending model calls; the config fingerprint and goldset digest must match.
    """
    items = _select_eval_items(config, items, split, limit)
    if not items:
        raise SystemExit(
            f"no verified '{split}' items in {config.goldset_path} "
            "(only items with verified=true are scored; public-reused sets ship "
            "verified=false pending human review)"
        )
    adapter_manifest = None
    if config.adapter_path is not None:
        from llb.finetune.guard import validate_adapter_for_eval

        adapter_manifest = validate_adapter_for_eval(
            adapter_path=config.adapter_path,
            items=items,
            model=config.model,
            judge_model=config.judge_model,
        )

    config_payload = config.fingerprint()
    if adapter_manifest is not None:
        config_payload["adapter"] = adapter_manifest
        label = adapter_manifest.get("adapter_label")
        if isinstance(label, str) and label:
            config_payload["model"] = label
    if prompt_system_provenance is not None:
        config_payload["prompt_system"] = prompt_system_provenance["prompt_system_id"]

    if resume is not None:
        run_timestamp, run_id, run_dir, staging_dir = durability.resume_target(
            config.run_dir, config.run_staging_dir, resume
        )
        if run_dir.exists():
            raise SystemExit(f"[run-eval] {run_dir} is already finalized; nothing to resume")
        if not staging_dir.exists():
            raise SystemExit(f"[run-eval] no interrupted run to resume at {staging_dir}")
        durability.verify_resume_meta(
            staging_dir, config_fingerprint=config_payload, items=items, split=split
        )
    else:
        run_id = uuid.uuid4().hex[:12]
        run_timestamp = _run_timestamp(run_id)
        run_dir = config.run_dir(run_timestamp)
        staging_dir = config.run_staging_dir(run_timestamp)
        staging_dir.mkdir(parents=True, exist_ok=True)
        durability.write_journal_meta(
            staging_dir,
            config_fingerprint=config_payload,
            items=items,
            run_id=run_id,
            split=split,
        )

    active_launcher: BackendLauncher | None = None
    counters = durability.DurabilityCounters()
    try:
        active_launcher, runner_fn, store, contention = _resolve_eval_runner(
            config,
            store=store,
            launcher=launcher,
            runner_fn=runner_fn,
            prompt_package=prompt_package,
            staging_dir=staging_dir,
            evict=evict,
            wait=wait,
        )
        embedder = (
            store.embedder if (config.score_semantic and hasattr(store, "embedder")) else None
        )
        score_options = _score_options(config)
        policy = durability.RetryPolicy(
            max_case_retries=max_case_retries,
            retry_backoff_s=retry_backoff_s,
            max_backend_relaunches=max_backend_relaunches,
        )
        with active_launcher as backend:

            def relaunch() -> None:
                backend.stop()
                backend.start()

            batch, counters = durability.execute_cases_durable(
                items,
                runner_fn,
                embedder,
                journal=durability.CaseJournal(durability.journal_path(staging_dir)),
                policy=policy,
                relaunch=relaunch,
                sleep=sleep if sleep is not None else time.sleep,
                counters=counters,
                options=score_options,
            )
            telemetry_report = _collect_optional_telemetry(config, backend)
            probe_report = _maybe_run_probes(config, items, store, backend)
    except KeyboardInterrupt:
        if active_launcher is not None:
            _preserve_backend_log(active_launcher, config)
        _LOG.warning(
            "[run-eval] interrupted; staging preserved -- resume with --resume %s", run_dir
        )
        raise
    except BaseException:
        if active_launcher is not None:
            _preserve_backend_log(active_launcher, config)
        if resume is None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        else:
            _LOG.warning("[run-eval] resume failed; staging kept for another --resume %s", run_dir)
        raise

    backend_telemetry: BackendMetadata = (
        active_launcher.telemetry() if hasattr(active_launcher, "telemetry") else {}
    )
    effective_telemetry = {**backend_telemetry, **(telemetry_report or {})}
    judge_score = _judge_cases(config, batch, judge_rho, judge_scorer)
    rows, metrics = _aggregate(config, batch.rows, judge_rho, effective_telemetry, judge_score)
    if probe_report is not None:
        metrics["abstention_accuracy"] = round(probe_report.abstention_accuracy, 4)
        metrics["n_probes"] = probe_report.n_probes
    retrieval_metrics = retrieval.evaluate_retrieval(batch.retrieval_pairs, config.top_k)

    manifest = RunManifest(
        run_id=run_id,
        run_name=config.run_name,
        split=split,
        config=config_payload,
        metrics=metrics,
        retrieval=retrieval_metrics,
        judge=_build_judge_metadata(config, judge_rho),
        telemetry=telemetry_report,
        contention=contention,
        durability=counters.as_status(),
        prompt_system_provenance=dict(prompt_system_provenance)
        if prompt_system_provenance is not None
        else None,
        n_cases=len(batch.rows),
    )
    durability.drop_journal(staging_dir)
    paths = persist_run(
        manifest,
        batch.rows,
        run_dir,
        mirror=mirror,
        staging_dir=staging_dir,
        retrieval_rows=batch_retrieval_records(batch),
    )

    worksheet_rows = 0
    if worksheet is not None:
        worksheet_rows = _write_calibration_worksheet(config, batch, worksheet, judge_scorer)
        paths["worksheet"] = str(worksheet)

    if probe_report is not None:
        from llb.eval.insufficient_context import write_probe

        paths.update(write_probe(probe_report, run_dir))  # type: ignore[typeddict-item]

    table = format_table(rows)
    if emit:
        emit_summary(
            config,
            len(batch.rows),
            retrieval_metrics,
            table,
            telemetry_report,
            paths,
            worksheet,
            worksheet_rows,
            metrics,
        )
    return {
        "rows": rows,
        "metrics": metrics,
        "retrieval": retrieval_metrics,
        "paths": paths,
        "table": table,
        "telemetry": telemetry_report,
        "manifest": manifest,
        "run_timestamp": run_timestamp,
    }
