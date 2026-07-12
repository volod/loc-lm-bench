"""Streamlit board over canonical run bundles."""

from pathlib import Path
from typing import Any

from llb.board.categories import load_category_composite, load_category_records
from llb.board.chain_context import chain_context_comparison, load_chain_context_records
from llb.board.harnesses import harness_comparison, load_agentic_harness_records
from llb.board.prompt_systems import (
    load_rag_prompt_system_records,
    rag_prompt_system_comparison,
)
from llb.board.runs import best_per_model, config_summary, load_run_records, load_screen_reports
from llb.core.paths import resolve_data_dir
from llb.scoring.aggregate import rank_board, ranking_policy_note


def render(run_root: Path | str | None = None, screen_root: Path | str | None = None) -> None:
    import streamlit as st

    data_dir = resolve_data_dir(None)
    root = Path(run_root) if run_root is not None else data_dir / "run-eval"
    screens = Path(screen_root) if screen_root is not None else data_dir / "screen"
    st.set_page_config(page_title="loc-lm-bench", layout="wide")
    st.title("loc-lm-bench -- local Ukrainian LLM leaderboard")

    _render_private_leaderboard(st, root)
    _render_category_boards(st, data_dir)
    _render_harness_comparison(st, data_dir)
    _render_chain_context_comparison(st, data_dir)
    _render_prompt_system_comparison(st, data_dir)
    _render_category_composite(st, data_dir)
    _render_public_screens(st, screens)
    st.caption("Deep per-run inspection: the MLflow UI (`make mlflow`).")


def _render_private_leaderboard(st: Any, root: Path) -> None:
    records = best_per_model(load_run_records(root))
    if not records:
        st.info(f"No Tier-2 runs under {root}. Run `llb run-eval` (or `llb sweep`) first.")
        return

    results = [r.result for r in records]
    rows = rank_board(results)
    st.subheader("Tier-2 private leaderboard (final split)")
    st.caption(ranking_policy_note(results, judge_trusted=False))
    st.caption("`*` = Pareto-optimal (quality / speed / VRAM); `~` = CI overlaps the model above.")
    st.dataframe(rows, use_container_width=True)

    st.subheader("Best config per model")
    by_model = {r.result.model: r for r in records}
    for row in rows:
        rec = by_model.get(row["model"])
        if rec is not None:
            st.write(f"**{row['model']}** ({row['backend']})", config_summary(rec.config))


def _render_category_boards(st: Any, data_dir: Path) -> None:
    category_by_tier = load_category_records(data_dir)
    if not category_by_tier:
        return
    st.subheader("Category boards (each own Tier, not cross-ranked)")
    for tier in sorted(category_by_tier):
        results = category_by_tier[tier]
        st.caption(f"{tier} -- {ranking_policy_note(results, judge_trusted=False)}")
        st.dataframe(rank_board(results), use_container_width=True)


def _render_harness_comparison(st: Any, data_dir: Path) -> None:
    harness_records = load_agentic_harness_records(data_dir)
    if not harness_records:
        return
    st.subheader("Agentic harness comparison (LangGraph vs CrewAI vs loop)")
    st.caption("Per model, the same task set + scoring + judge are held fixed; harness varies.")
    for model in sorted({r.model for r in harness_records}):
        rows, _table, harnesses = harness_comparison(data_dir, model)
        if rows:
            st.caption(f"**{model}** -- harnesses: {', '.join(sorted(set(harnesses)))}")
            st.dataframe(rows, use_container_width=True)


def _render_chain_context_comparison(st: Any, data_dir: Path) -> None:
    policy_records = load_chain_context_records(data_dir)
    if not policy_records:
        return
    st.subheader("Context-policy comparison (fresh vs history vs summary vs roles)")
    st.caption("Per model, the chain set + scoring stay fixed; only the context policy varies.")
    for model in sorted({r.model for r in policy_records}):
        rows, _table, policies = chain_context_comparison(data_dir, model)
        if rows:
            st.caption(f"**{model}** -- policies: {', '.join(sorted(set(policies)))}")
            st.dataframe(rows, use_container_width=True)


def _render_prompt_system_comparison(st: Any, data_dir: Path) -> None:
    rag_prompt_records = load_rag_prompt_system_records(data_dir)
    if not rag_prompt_records:
        return
    st.subheader("RAG prompt-system comparison")
    st.caption("Per model, retrieval/scoring stay fixed; only the prepended prompt system varies.")
    for model in sorted({r.model for r in rag_prompt_records}):
        rows, _table, ids = rag_prompt_system_comparison(data_dir, model)
        if rows:
            st.caption(f"**{model}** -- prompt systems: {', '.join(ids)}")
            st.dataframe(rows, use_container_width=True)


def _render_category_composite(st: Any, data_dir: Path) -> None:
    composite_rows, composite_issues = load_category_composite(data_dir)
    if composite_rows:
        st.subheader("Category composite headline (verified suite)")
        st.caption(
            "Shown only when every required category is present, CI-capable, and stamped as verified."
        )
        st.dataframe(composite_rows, use_container_width=True)
    elif composite_issues:
        st.info(
            "Category composite headline is gated until every required category has verified data "
            "and a reloadable per-case CI series."
        )


def _render_public_screens(st: Any, screens: Path) -> None:
    reports = load_screen_reports(screens)
    if not reports:
        return
    st.subheader("Tier-1 public screen (NOT comparable to Tier-2)")
    for track in sorted({r["track"] for r in reports}):
        st.caption(f"track: {track}")
        st.dataframe([r for r in reports if r["track"] == track], use_container_width=True)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
