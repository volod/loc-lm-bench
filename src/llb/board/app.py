"""Streamlit board (M3.7) -- a thin page over the canonical run bundles.

Deliberately minimal: the average-rank leaderboard (with Pareto + CI-overlap markers from
M3.6) plus the best config per model. Deep per-run inspection lives in the MLflow UI
(`make mlflow`). Run with `llb board` (which shells out to `streamlit run` on this file).
"""

from pathlib import Path

from llb.board.data import (
    best_per_model,
    config_summary,
    harness_comparison,
    load_agentic_harness_records,
    load_category_records,
    load_m5_composite,
    load_run_records,
    load_screen_reports,
)
from llb.paths import resolve_data_dir
from llb.scoring.aggregate import rank_board, ranking_policy_note


def render(run_root: Path | str | None = None, screen_root: Path | str | None = None) -> None:
    import streamlit as st

    data_dir = resolve_data_dir(None)
    root = Path(run_root) if run_root is not None else data_dir / "run-eval"
    screens = Path(screen_root) if screen_root is not None else data_dir / "screen"
    st.set_page_config(page_title="loc-lm-bench", layout="wide")
    st.title("loc-lm-bench -- local Ukrainian LLM leaderboard")

    records = best_per_model(load_run_records(root))
    if records:
        results = [r.result for r in records]
        rows = rank_board(results)
        st.subheader("Tier-2 private leaderboard (final split)")
        st.caption(ranking_policy_note(results, judge_trusted=False))
        st.caption(
            "`*` = Pareto-optimal (quality / speed / VRAM); `~` = CI overlaps the model above."
        )
        st.dataframe(rows, use_container_width=True)

        st.subheader("Best config per model")
        by_model = {r.result.model: r for r in records}
        for row in rows:
            rec = by_model.get(row["model"])
            if rec is not None:
                st.write(f"**{row['model']}** ({row['backend']})", config_summary(rec.config))
    else:
        st.info(f"No Tier-2 runs under {root}. Run `llb run-eval` (or `llb sweep`) first.")

    # M5 category boards -- each renders under its OWN Tier and is NEVER cross-ranked with the
    # Tier-2 RAG board or with another category (the aggregate guard refuses a mixed-tier board).
    category_by_tier = load_category_records(data_dir)
    if category_by_tier:
        st.subheader("M5 category boards (each its own Tier, not cross-ranked)")
        for tier in sorted(category_by_tier):
            results = category_by_tier[tier]
            st.caption(f"{tier} -- {ranking_policy_note(results, judge_trusted=False)}")
            st.dataframe(rank_board(results), use_container_width=True)

    # M7.1 agentic harness comparison -- ranks ONE model across {loop, langgraph, crewai} under
    # TIER_AGENTIC, so the harness effect is isolated without cross-ranking models.
    harness_records = load_agentic_harness_records(data_dir)
    if harness_records:
        st.subheader("M7.1 agentic harness comparison (LangGraph vs CrewAI vs loop)")
        st.caption("Per model, the same task set + scoring + judge are held fixed; harness varies.")
        for model in sorted({r.model for r in harness_records}):
            rows, _table, harnesses = harness_comparison(data_dir, model)
            if rows:
                st.caption(f"**{model}** -- harnesses: {', '.join(sorted(set(harnesses)))}")
                st.dataframe(rows, use_container_width=True)

    composite_rows, composite_issues = load_m5_composite(data_dir)
    if composite_rows:
        st.subheader("M5 composite headline (verified category suite)")
        st.caption(
            "Shown only when every required M5 category is present, CI-capable, and stamped "
            "as verified."
        )
        st.dataframe(composite_rows, use_container_width=True)
    elif composite_issues:
        st.info(
            "M5 composite headline is gated until every required category has verified data "
            "and a reloadable per-case CI series."
        )

    # Tier-1 public screens are shown SEPARATELY -- loglikelihood/generation tracks are not
    # comparable to Tier-2 private metrics and are never ranked together.
    reports = load_screen_reports(screens)
    if reports:
        st.subheader("Tier-1 public screen (NOT comparable to Tier-2)")
        for track in sorted({r["track"] for r in reports}):
            st.caption(f"track: {track}")
            st.dataframe([r for r in reports if r["track"] == track], use_container_width=True)

    st.caption("Deep per-run inspection: the MLflow UI (`make mlflow`).")


def main() -> None:
    render()


if __name__ == "__main__":
    main()
