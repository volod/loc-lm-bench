"""Eval inputs for the runner: gold items, retrieval store, default per-case runner fn, the opt-in
query-prep lane, and the abstention probe. `runner_backend.py` owns the backend lifecycle.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llb.backends.base import BackendLauncher
from llb.backends.prompt_window import PromptWindow
from llb.core.config import RunConfig
from llb.eval import graph as eval_graph
from llb.executor.cases import spans_as_dicts
from llb.goldset.schema import GoldItem, load_goldset

if TYPE_CHECKING:
    from llb.eval.insufficient_context import InsufficientContextReport
    from llb.executor.cases import ScoreOptions
from llb.executor.runner_retrieval import build_query_prep

from llb.eval.graph_contracts import RagState


def _load_eval_items(
    config: RunConfig, split: str, limit: int | None, verified_only: bool = True
) -> list[GoldItem]:
    if not config.goldset_path.exists():
        raise SystemExit(
            f"gold set not found: {config.goldset_path}\n"
            "  use the committed fixture with --goldset "
            "samples/goldsets/ua_squad_postedited_v1/goldset.jsonl,\n"
            "  or create unverified development material with `make ingest-uk-squad`."
        )
    items = [
        item
        for item in load_goldset(config.goldset_path)
        if item.split == split and (item.verified or not verified_only)
    ]
    items.sort(key=lambda it: it.id)
    return items[:limit] if limit is not None else items


def _select_eval_items(
    config: RunConfig,
    items: list[GoldItem] | None,
    split: str,
    limit: int | None,
    verified_only: bool = True,
) -> list[GoldItem]:
    """The scored item selection.

    `verified_only=False` scores a DRAFTED ledger whose items no reviewer has accepted. It exists
    for diagnostic lanes that must measure the same set a retrieval sweep measured, never for a
    leaderboard run, and the run that uses it records the grounding in its manifest so a
    drafted-grounded score can never be mistaken for a verified one.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if items is None:
        return _load_eval_items(config, split, limit, verified_only)
    selected = sorted(
        (item for item in items if item.split == split and (item.verified or not verified_only)),
        key=lambda item: item.id,
    )
    return selected[:limit] if limit is not None else selected


def _default_runner_fn(
    config: RunConfig, store: Any, launcher: BackendLauncher, prompt_package: Any | None = None
) -> tuple[Callable[[GoldItem], RagState], PromptWindow]:
    """The per-case runner for this config's context strategy, and the run's usable prompt window.

    A non-`rag` strategy (rag-vs-long-context-ablation) either swaps the retrieve node's store
    lookup for its own context source (`closed_book`, `long_context`) or refines what ordinary
    retrieval produced (`retrieved_document`, which still retrieves). `closed_book` additionally
    brings its own generation prompt. The store is always loaded and passed: it carries the
    embedder the optional semantic correctness signal scores with, so every lane scores its
    answers identically.

    The window is resolved ONCE per run and owned here rather than by a lane, because every
    strategy shares it: the document lanes skip an item against it, and the `rag` lane is checked
    against it before the run spends anything. It reads the MINIMUM of the declared window and the
    one `launcher` is actually serving, and the run manifest records which side bound it.
    """
    from llb.eval.context_ablation.sources import build_context_lane

    chunk_filter = None
    if config.acl_label is not None:
        from llb.rag.filters import metadata_filter

        chunk_filter = metadata_filter(acl_label=config.acl_label)
    header_restorer = None
    if config.restore_table_headers:
        from llb.eval.table_headers import corpus_header_restorer

        header_restorer = corpus_header_restorer(config.corpus_root)
    window = PromptWindow(config, launcher=launcher)
    lane = build_context_lane(config, window.fits)
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
        context_source=lane.source if lane is not None else None,
        template_id=lane.template_id if lane is not None else None,
        context_refiner=lane.refiner if lane is not None else None,
        header_restorer=header_restorer,
        answer_format=config.answer_format,
        suppress_reasoning=config.suppress_reasoning_prompt,
    )

    def run(item: GoldItem) -> RagState:
        return eval_graph.run_case(app, item.question, spans_as_dicts(item))

    return run, window


def check_rag_prompt_window(config: RunConfig, window: PromptWindow | None) -> str | None:
    """The warning a `rag` run earns when its retrieved prompt cannot fit the served window.

    A `rag` overflow is a CONFIGURATION error, not a per-item outcome: `top_k * chunk_size` is the
    same on every item, so either every prompt fits or none do, and skipping items would report a
    truncated configuration as a corpus finding. The run is not refused either -- `top_k *
    chunk_size` is an UPPER bound on a context that short chunks, ACL filters, and reranking
    routinely make smaller, so refusing on it would block runs that do fit. What it must not be is
    silent: the backend truncates without saying so, and the score would read as that
    configuration's quality. Returns None when the prompt fits or nothing can bound it.
    """
    # Imported lazily: `llb.eval.context_ablation` pulls in the comparison stack, and this module
    # is imported while `llb.executor.runner` is still initializing.
    from llb.eval.context_ablation.models import LANE_RAG
    from llb.optimize.tuning_space import estimate_prompt_tokens

    if window is None or config.context_strategy != LANE_RAG:
        return None
    if window.fits(config.top_k * config.chunk_size):
        return None
    budget = window.resolve()
    return (
        f"[run-eval] the retrieved prompt may not fit: top_k {config.top_k} x chunk_size "
        f"{config.chunk_size} estimates ~{estimate_prompt_tokens(config)} tokens against a "
        f"{budget.budget_source} window of {budget.bound_max_model_len} "
        f"(declared {budget.declared_max_model_len}, served {budget.served_max_model_len}). "
        "The backend truncates an over-long prompt silently, so read this run's scores as measured "
        "on a shortened context; lower --top-k / --chunk-size, or raise --max-model-len so the "
        "backend serves the window the config declares."
    )


def _score_options(config: RunConfig) -> "ScoreOptions":
    """The opt-in answer-side scoring toggles for this run (groundedness-citation-metrics)."""
    from llb.executor.cases import ScoreOptions

    return ScoreOptions(
        score_groundedness=config.score_groundedness,
        cited_answers=config.cited_answers,
        context_order=config.context_order,
        answer_format=config.answer_format,
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
