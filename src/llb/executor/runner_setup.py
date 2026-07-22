"""Eval inputs for the runner: gold items, retrieval store, default per-case runner fn, the opt-in
query-prep lane, and the abstention probe. `runner_backend.py` owns the backend lifecycle.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.eval import graph as eval_graph
from llb.executor.cases import spans_as_dicts
from llb.goldset.schema import GoldItem, load_goldset

if TYPE_CHECKING:
    from llb.eval.insufficient_context import InsufficientContextReport
    from llb.executor.cases import ScoreOptions
from llb.executor.runner_retrieval import build_query_prep

RagState = eval_graph.RagState


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
) -> Callable[[GoldItem], RagState]:
    """The per-case runner for this config's context strategy.

    A non-`rag` strategy (rag-vs-long-context-ablation) swaps the retrieve node's store lookup for
    its own context source and, for `closed_book`, its own generation prompt. The store is still
    loaded and passed: it carries the embedder the optional semantic correctness signal scores
    with, so every lane scores its answers identically.
    """
    from llb.eval.context_ablation.sources import build_context_lane

    chunk_filter = None
    if config.acl_label is not None:
        from llb.rag.filters import metadata_filter

        chunk_filter = metadata_filter(acl_label=config.acl_label)
    lane = build_context_lane(config)
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
